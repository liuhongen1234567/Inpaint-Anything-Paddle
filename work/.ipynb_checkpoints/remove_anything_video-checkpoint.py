import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from PIL import Image 
from pathlib import Path
import paddle
import cv2
from typing import Union, Optional, List, Tuple, Text, BinaryIO
import numpy as np
from segment_anything.predictor import SamPredictor
from segment_anything.build_sam import sam_model_registry
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points,array_to_img
import math
import warnings
warnings.filterwarnings("ignore")
import logging
import imageio
from tqdm import tqdm
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0' 

label_list = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

model_link = {
    'vit_h':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_h/model.pdparams",
    'vit_l':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_l/model.pdparams",
    'vit_b':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_b/model.pdparams"
}


def get_args():
    parser = argparse.ArgumentParser(
        description='Segment image with point promp or box')
    # Parameters
    parser.add_argument(
        "--input_video", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--remove_type", type=list,nargs='+', required=True,
        help="Path to remove type",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b'],
        help="The type of sam model to load. Default: 'vit_h"
    )

    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--predict_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )

    return parser.parse_args()


def ResizePad(img, target_size):
    img = np.array(img)
    h, w = img.shape[:2]
    m = max(h, w)
    ratio = target_size / m
    new_w, new_h = int(ratio * w), int(ratio * h)
    img = cv2.resize(img, (new_w, new_h), cv2.INTER_LINEAR)
    top = (target_size - new_h) // 2
    bottom = (target_size - new_h) - top
    left = (target_size - new_w) // 2
    right = (target_size - new_w) - left
    img1 = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return img1

def process_mutilabel(img,  resize_with_pad):

    img = np.array(img)
    img = ResizePad(img, target_size=resize_with_pad)
    img = np.array(img).astype('float32').transpose((2, 0, 1)) 
    img = img /255.0

    return img

def generate_scale(im, resize_shape, keep_ratio):
    """
    Args:
        im (np.ndarray): image (np.ndarray)
    Returns:
        im_scale_x: the resize ratio of X
        im_scale_y: the resize ratio of Y
    """
    target_size = (resize_shape[0], resize_shape[1])
    origin_shape = im.shape[:2]

    if keep_ratio:
        im_size_min = np.min(origin_shape)
        im_size_max = np.max(origin_shape)
        target_size_min = np.min(target_size)
        target_size_max = np.max(target_size)
        im_scale = float(target_size_min) / float(im_size_min)
        if np.round(im_scale * im_size_max) > target_size_max:
            im_scale = float(target_size_max) / float(im_size_max)
        im_scale_x = im_scale
        im_scale_y = im_scale
    else:
        resize_h, resize_w = target_size
        im_scale_y = resize_h / float(origin_shape[0])
        im_scale_x = resize_w / float(origin_shape[1])
    return im_scale_y, im_scale_x


def resize(im, im_info, resize_shape, keep_ratio):
    interp = 2
    im_scale_y, im_scale_x = generate_scale(im, resize_shape, keep_ratio)
    im = cv2.resize(
        im,
        None,
        None,
        fx=im_scale_x,
        fy=im_scale_y,
        interpolation=interp)
    # print("scale,", im_scale_y, im_scale_x)
    im_info['im_shape'] = np.array(im.shape[:2]).astype('float32')
    im_info['scale_factor'] = np.array(
        [im_scale_y, im_scale_x]).astype('float32')

    return im, im_info

def process_yoloe(im, im_info, resize_shape):
    im_info['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
    im_info['scale_factor'] = np.array([1., 1.], dtype=np.float32)
    # print(im)

    im, im_info = resize(im, im_info, resize_shape, False)
    h_n, w_n = im.shape[:-1]
    im = im / 255.0
    im = im.transpose((2, 0, 1)).copy()

    im = paddle.to_tensor(im, dtype='float32')
    im = im.unsqueeze(0)
    factor = paddle.to_tensor(im_info['scale_factor']).reshape((1, 2)).astype('float32')
    im_shape = paddle.to_tensor(im_info['im_shape'].reshape((1, 2)), dtype='float32')
    return im, im_shape, factor

@paddle.no_grad()
def make_grid(
    tensor: Union[paddle.Tensor, List[paddle.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    **kwargs
) -> paddle.Tensor:

    if not (paddle.is_tensor(tensor) or
            (isinstance(tensor, list) and all(paddle.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = paddle.stack(tensor, axis=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = paddle.concat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = paddle.concat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = paddle.full([num_channels, height * ymaps + padding, width * xmaps + padding], pad_value)

    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid

def save_videos_grid(videos: paddle.Tensor, path: str, rescale=False, n_rows=4, fps=8):

    videos = videos.transpose([2, 0, 1, 3, 4])
    outputs = []
    for x in videos:
        x = make_grid(x, nrow=n_rows)
  
        # [1,3, 512, 512] 
        x = x.squeeze(0).transpose([1, 2, 0])
        # x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

def select_desired_box(pre,remove_type):
    box = []
    maxS = 0
    box = []
    max_item = None
    for item in pre[0].numpy(): 
        cls, value, xmin, ymin, xmax, ymax = list(item)
        cls, xmin, ymin, xmax, ymax = [int(x) for x in [cls, xmin, ymin, xmax, ymax]]
        curS = (ymax-ymin)*(xmax-xmin)
        label = label_list[cls]
        # if value>0.5:
        #    print(label)

        if value>0.5 and label!="person" and (label in remove_type):
            # print(cls,"cls")
            box.append( np.array([[xmin, ymin], [xmax, ymax]]))

        if value>0.5 and label=="person" and (label in remove_type):
            if curS>maxS:
                maxS=curS
                max_item = item
    
    if max_item is not None:
        cls, value, xmin, ymin, xmax, ymax = list( max_item )
        box.append( np.array([[xmin, ymin], [xmax, ymax]]))
    return box



def main(args):
    if paddle.is_compiled_with_cuda():
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")
    input_path = args.input_video

    remove_type = args.remove_type
    remove_type = [ "".join(r) for r in remove_type]
    # print("remove",remove_type)

    src_video_dir =  input_path 
    video_object = cv2.VideoCapture(src_video_dir)
    fps = video_object.get(cv2.CAP_PROP_FPS)
    frame_paths_list = []
    detector = paddle.jit.load('/home/aistudio/ppyoloe_plus_crn_l_80e_coco/model')
    detector.eval()
    model = sam_model_registry[args.sam_model_type](
        checkpoint=model_link[args.sam_model_type])

    frame_count = int(video_object.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=frame_count)

    for item in  remove_type:
        if item not in label_list:
            raise ValueError('the remove object type is not in COCO 80 class ')

    while True:

        ret, frame = video_object.read()
        
        if ret == False:
            print("predict_rbox_frame_from_video({})".format(src_video_dir))
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
          
        im_info = {
            'scale_factor': np.array(
                [1., 1.], dtype=np.float32),
            'im_shape': None,
        }

        h1, w1 = frame.shape[:-1]
        im, im_shape, factor = process_yoloe(frame, im_info, [640, 640])

        with paddle.no_grad():
            pre = detector(im, factor)
        box = select_desired_box(pre,remove_type)
        if len(box)>0:
            init_mask = np.zeros([h1,w1])
            for b in box:
                predictor = SamPredictor(model)
                predictor.set_image(frame)
                masks1, _, _ = predictor.predict(
                point_coords=None,
                point_labels=1,
                box=b,
                multimask_output=True, )
                masks1 = masks1.astype(np.uint8) * 255
                if args.dilate_kernel_size is not None:
                    masks1 = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks1]
                idx = np.array(masks1[0]==255)
                # print("init", init_mask.shape,masks[0].shape,np.array(masks[0]==255).shape)
                init_mask[idx]=255
            img_inpainted = inpaint_img_with_lama(
                frame,  init_mask, args.lama_config, args.lama_ckpt, args.predict_config)
            img_inpainted = array_to_img(img_inpainted)

            frame_remove  = np.array(img_inpainted)
            frame_remove = cv2.resize( frame_remove,dsize=None, fx=0.4, fy= 0.4 )
            orginal_frame = cv2.resize( frame,dsize=None, fx=0.4, fy= 0.4)
            concat_frame = np.hstack(( orginal_frame,frame_remove))
            frame_remove = paddle.to_tensor(concat_frame /255.0, dtype='float32').unsqueeze(0)
            frame_paths_list.append( frame_remove)
        # break
        progress_bar.update(1)
            
    video_seq = paddle.concat(frame_paths_list, axis= 0)
    video_seq = paddle.to_tensor(video_seq).transpose([3, 0, 1, 2 ]).unsqueeze(0)

    img_stem = Path(args.input_video).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    git_img_p ="{}/ {}".format(out_dir,os.path.basename(input_path).replace(".mp4",".gif"))
    save_videos_grid(video_seq,    git_img_p,fps=fps )


    # for idx, mask in enumerate(masks):
    #     mask_p = out_dir / f"mask_{idx}.jpg"
    #     img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
    #     save_array_to_img(mask, img_inpainted_p)

if __name__ == "__main__":
    args = get_args()
    main(args)

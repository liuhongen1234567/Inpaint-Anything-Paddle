import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from PIL import Image 
from pathlib import Path
import paddle
import cv2
import numpy as np
from segment_anything.predictor import SamPredictor
from segment_anything.build_sam import sam_model_registry
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0' 

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
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=True,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=True,
        help="The labels of the point prompt, 1 or 0.",
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

def main(args):
    if paddle.is_compiled_with_cuda():
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")
    input_path = args.input_img

    point, box = args.point_coords, None
    if point is not None:
        point = np.array([point])
        input_label = np.array([1])
    else:
        input_label = None
    if box is not None:
        box = np.array([[box[0], box[1]], [box[2], box[3]]])

    img  = cv2.imread(input_path)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model = sam_model_registry[args.sam_model_type](
        checkpoint=model_link[args.sam_model_type])
    predictor = SamPredictor(model)
    predictor.set_image(img)

    # print("point",point,box)

    masks, _, _ = predictor.predict(
        point_coords=point,
        point_labels=input_label,
        box=box,
        multimask_output=True, )

    masks = masks.astype(np.uint8) * 255
    if args.dilate_kernel_size is not None:
        masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        # path to the results
        mask_p = out_dir / f"mask_{idx}.jpg"
        img_points_p = out_dir / f"with_points.jpg"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        # save the mask
        save_array_to_img(mask, mask_p)

        # save the pointed and masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), [args.point_coords], args.point_labels,
                    size=(width*0.04)**2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()

    for idx, mask in enumerate(masks):
        mask_p = out_dir / f"mask_{idx}.jpg"
        img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
        img_inpainted = inpaint_img_with_lama(
            img, mask, args.lama_config, args.lama_ckpt, args.predict_config)
        save_array_to_img(img_inpainted, img_inpainted_p)


if __name__ == "__main__":
    args = get_args()
    main(args)

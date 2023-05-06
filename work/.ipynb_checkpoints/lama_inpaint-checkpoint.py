
import cv2
import os
import sys
import numpy as np
import paddle
import yaml
import glob
import argparse
from PIL import Image
from omegaconf import OmegaConf
from saicinpainting.training.modules import make_generator
import paddle.nn.functional as F

def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)

def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


def pad_tensor_to_modulo(img, mod):
    batch_size, channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return F.pad(img, pad=(0, out_width - width, 0, out_height - height), mode='reflect')


def inpaint_img_with_lama(
        img: np.ndarray,
        mask: np.ndarray,
        config_p: str,
        ckpt_p: str,
        predict_config: str="./lama/configs/prediction/default.yaml",
        mod=8,
):
    assert len(mask.shape) == 2
    if np.max(mask) == 1:
        mask = mask * 255
    img = paddle.to_tensor(img/255.0,dtype='float32')
    mask = paddle.to_tensor(mask,dtype="float32")

    predict_config = OmegaConf.load(predict_config)
    predict_config.model.path = ckpt_p

    with open(config_p, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'
    
    model = make_generator(train_config, **train_config.generator)
    path = ckpt_p
    state = paddle.load(path)
    model.set_state_dict(state)
    model.eval()

    batch = {}
    batch['image'] = img.transpose([2, 0, 1]).unsqueeze(0)
    batch['mask'] = mask[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
    batch['mask'] = (batch['mask'] > 0).cast('float32')

    img = batch['image']
    mask = batch['mask']
    img = paddle.to_tensor(img) 
    mask = paddle.to_tensor(mask)  
    masked_img = img * (1 - mask)
    masked_img = paddle.concat([masked_img, mask], axis =1)

    with paddle.no_grad():
        batch['predicted_image'] = model(masked_img)
    batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']

    cur_res = batch[predict_config.out_key][0].transpose([1, 2, 0])
    cur_res = cur_res.detach().cpu().numpy()

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res

# img = cv2.imread("/home/aistudio/cat.jpg")
# mask = cv2.imread("/home/aistudio/cat_mask.png")
# mask = np.array(mask[:,:,0]).squeeze()
# lama_ckpt = "/home/aistudio/work/lamn/big_lanm/"
# lama_config ="/home/aistudio/work/lamn/config/default.yml"
# img_inpainted_p = "/home/aistudio/cat_delete.png"
# img_inpainted  = inpaint_img_with_lama(img, mask, lama_config, lama_ckpt)
# save_array_to_img(img_inpainted, img_inpainted_p)
# plt.imshow(img)
# plt.show()
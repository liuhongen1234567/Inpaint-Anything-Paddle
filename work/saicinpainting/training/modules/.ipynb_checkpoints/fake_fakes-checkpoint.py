import paddle
from kornia import SamplePadding
# from kornia.augmentation import RandomAffine, CenterCrop


class FakeFakesGenerator:
    def __init__(self, aug_proba=0.5, img_aug_degree=30, img_aug_translate=0.2):
        # self.grad_aug = RandomAffine(degrees=360,
        #                              translate=0.2,
        #                              padding_mode=SamplePadding.REFLECTION,
        #                              keepaxis=False,
        #                              p=1)
        # self.img_aug = RandomAffine(degrees=img_aug_degree,
        #                             translate=img_aug_translate,
        #                             padding_mode=SamplePadding.REFLECTION,
        #                             keepaxis=True,
        #                             p=1)
        self.aug_proba = aug_proba

    def __call__(self, input_images, masks):
        blend_masks = self._fill_masks_with_gradient(masks)
        blend_target = self._make_blend_target(input_images)
        result = input_images * (1 - blend_masks) + blend_target * blend_masks
        return result, blend_masks

    def _make_blend_target(self, input_images):
        batch_size = input_images.shape[0]
        permuted = input_images[paddle.randperm(batch_size)]
        augmented = self.img_aug(input_images)
        is_aug = (paddle.rand(batch_size)[:, None, None, None] < self.aug_proba).float()
        result = augmented * is_aug + permuted * (1 - is_aug)
        return result

    def _fill_masks_with_gradient(self, masks):
        batch_size, _, height, width = masks.shape
        grad = paddle.linspace(0, 1, num=width * 2, dtype=masks.dtype) \
            .reshape([1, 1, 1, -1]).expand(batch_size, 1, height * 2, width * 2)
        # grad = self.grad_aug(grad)
        # grad = CenterCrop((height, width))(grad)
        grad *= masks

        grad_for_min = grad + (1 - masks) * 10
        grad -= grad_for_min.view(batch_size, -1).min(-1).values[:, None, None, None]
        grad /= grad.view(batch_size, -1).max(-1).values[:, None, None, None] + 1e-6
        grad.clamp_(min=0, max=1)

        return grad

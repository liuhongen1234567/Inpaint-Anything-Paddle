from typing import Optional, Tuple, Union

import paddle
import paddle.nn as nn
from paddle import Tensor
import paddle.nn.functional as F
# utilities to compute affine matrices

pi = paddle.to_tensor(3.14159265358979323846)

def deg2rad(tensor: paddle.Tensor) -> paddle.Tensor:

    if not isinstance(tensor, paddle.Tensor):
        raise TypeError(f"Input type is not a paddle.Tensor. Got {type(tensor)}")

    return tensor * paddle.to_tensor(pi,dtype=type(tensor.dtype)) / 180.0

def angle_to_rotation_matrix(angle: paddle.Tensor) -> paddle.Tensor:

    ang_rad = deg2rad(angle)
    cos_a: paddle.Tensor = paddle.cos(ang_rad)
    sin_a: paddle.Tensor = paddle.sin(ang_rad)
    return paddle.stack([cos_a, sin_a, -sin_a, cos_a], axis=-1).reshape([*angle.shape, 2, 2])

def get_rotation_matrix2d(center: Tensor, angle: Tensor, scale: Tensor) -> Tensor:

    if not isinstance(center, Tensor):
        raise TypeError(f"Input center type is not a Tensor. Got {type(center)}")

    if not isinstance(angle, Tensor):
        raise TypeError(f"Input angle type is not a Tensor. Got {type(angle)}")

    if not isinstance(scale, Tensor):
        raise TypeError(f"Input scale type is not a Tensor. Got {type(scale)}")

    if not (len(center.shape) == 2 and center.shape[1] == 2):
        raise ValueError(f"Input center must be a Bx2 tensor. Got {center.shape}")

    if not len(angle.shape) == 1:
        raise ValueError(f"Input angle must be a B tensor. Got {angle.shape}")

    if not (len(scale.shape) == 2 and scale.shape[1] == 2):
        raise ValueError(f"Input scale must be a Bx2 tensor. Got {scale.shape}")

    if not (center.shape[0] == angle.shape[0] == scale.shape[0]):
        raise ValueError(
            "Inputs must have same batch size dimension. Got center {}, angle {} and scale {}".format(
                center.shape, angle.shape, scale.shape
            )
        )

    if not (center.device == angle.device == scale.device) or not (center.dtype == angle.dtype == scale.dtype):
        raise ValueError(
            "Inputs must have same device Got center ({}, {}), angle ({}, {}) and scale ({}, {})".format(
                center.device, center.dtype, angle.device, angle.dtype, scale.device, scale.dtype
            )
        )

    shift_m = eye_like(3, center)
    shift_m[:, :2, 2] = center

    shift_m_inv = eye_like(3, center)
    shift_m_inv[:, :2, 2] = -center

    scale_m = eye_like(3, center)
    scale_m[:, 0, 0] *= scale[:, 0]
    scale_m[:, 1, 1] *= scale[:, 1]

    rotat_m = eye_like(3, center)
    rotat_m[:, :2, :2] = angle_to_rotation_matrix(angle)

    affine_m = shift_m @ rotat_m @ scale_m @ shift_m_inv
    return affine_m[:, :2, :]  # Bx2x3

def _compute_tensor_center(tensor: paddle.Tensor) -> paddle.Tensor:
    """Compute the center of tensor plane for (H, W), (C, H, W) and (B, C, H, W)."""
    if not 2 <= len(tensor.shape) <= 4:
        raise AssertionError(f"Must be a 3D tensor as HW, CHW and BCHW. Got {tensor.shape}.")
    height, width = tensor.shape[-2:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    center: paddle.Tensor = paddle.to_tensor([center_x, center_y], dtype=tensor.dtype)
    return center


def _compute_tensor_center3d(tensor: paddle.Tensor) -> paddle.Tensor:
    """Compute the center of tensor plane for (D, H, W), (C, D, H, W) and (B, C, D, H, W)."""
    if not 3 <= len(tensor.shape) <= 5:
        raise AssertionError(f"Must be a 3D tensor as DHW, CDHW and BCDHW. Got {tensor.shape}.")
    depth, height, width = tensor.shape[-3:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    center_z: float = float(depth - 1) / 2
    center: paddle.Tensor = paddle.to_tensor([center_x, center_y, center_z])
    return center


def _compute_rotation_matrix(angle: paddle.Tensor, center: paddle.Tensor) -> paddle.Tensor:
    """Compute a pure affine rotation matrix."""
    scale: paddle.Tensor = paddle.ones_like(center)
    matrix: paddle.Tensor = get_rotation_matrix2d(center, angle, scale)
    return matrix


def angle_axis_to_rotation_matrix(angle_axis: paddle.Tensor) -> paddle.Tensor:

    if not isinstance(angle_axis, paddle.Tensor):
        raise TypeError(f"Input type is not a paddle.Tensor. Got {type(angle_axis)}")

    if not angle_axis.shape[-1] == 3:
        raise ValueError(f"Input size must be a (*, 3) tensor. Got {angle_axis.shape}")

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = paddle.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = paddle.chunk(wxyz, 3, axis=1)
        cos_theta = paddle.cos(theta)
        sin_theta = paddle.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = paddle.concat([r00, r01, r02, r10, r11, r12, r20, r21, r22], axis=1)
        return rotation_matrix.reshape([-1, 3, 3])

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = paddle.chunk(angle_axis, 3, axis=1)
        k_one = paddle.ones_like(rx)
        rotation_matrix = paddle.concat([k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], axis=1)
        return rotation_matrix.reshape([-1, 3, 3])

    # stolen from ceres/rotation.h

    _angle_axis = paddle.unsqueeze(angle_axis,axis=1)
    theta2 = paddle.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = paddle.squeeze(theta2, axis=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (~mask).type_as(theta2)

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = paddle.eye(3).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.reshape([1, 3, 3]).tile([batch_size, 1, 1])
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx3x3


def projection_from_Rt(rmat: Tensor, tvec: Tensor) -> Tensor:
    r"""Compute the projection matrix from Rotation and translation.

    .. warning::
        This API signature it is experimental and might suffer some changes in the future.

    Concatenates the batch of rotations and translations such that :math:`P = [R | t]`.

    Args:
       rmat: the rotation matrix with shape :math:`(*, 3, 3)`.
       tvec: the translation vector with shape :math:`(*, 3, 1)`.

    Returns:
       the projection matrix with shape :math:`(*, 3, 4)`.
    """
    if not (len(rmat.shape) >= 2 and rmat.shape[-2:] == (3, 3)):
        raise AssertionError(rmat.shape)
    if not (len(tvec.shape) >= 2 and tvec.shape[-2:] == (3, 1)):
        raise AssertionError(tvec.shape)

    return paddle.concat([rmat, tvec], -1)  # Bx3x4

def _convert_affinematrix_to_homography_impl(A: Tensor) -> Tensor:
    H: Tensor = F.pad(A, [0, 0, 0, 1], "constant", value=0.0)
    H[..., -1, -1] += 1.0
    return H

def convert_affinematrix_to_homography3d(A: Tensor) -> Tensor:

    if not isinstance(A, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(A)}")

    if not (len(A.shape) == 3 and A.shape[-2:] == (3, 4)):
        raise ValueError(f"Input matrix must be a Bx3x4 tensor. Got {A.shape}")

    return _convert_affinematrix_to_homography_impl(A)

def get_projective_transform(center: Tensor, angles: Tensor, scales: Tensor) -> Tensor:

    if not (len(center.shape) == 2 and center.shape[-1] == 3):
        raise AssertionError(center.shape)
    if not (len(angles.shape) == 2 and angles.shape[-1] == 3):
        raise AssertionError(angles.shape)
    if center.device != angles.device:
        raise AssertionError(center.device, angles.device)
    if center.dtype != angles.dtype:
        raise AssertionError(center.dtype, angles.dtype)

    # create rotation matrix
    angle_axis_rad: Tensor = deg2rad(angles)
    rmat: Tensor = angle_axis_to_rotation_matrix(angle_axis_rad)  # Bx3x3
    scaling_matrix: Tensor = eye_like(3, rmat)
    scaling_matrix = scaling_matrix * scales.unsqueeze(dim=1)
    rmat = rmat @ scaling_matrix.to(rmat)

    # define matrix to move forth and back to origin
    from_origin_mat = eye_like(4, rmat, shared_memory=False)  # Bx4x4
    from_origin_mat[..., :3, -1] += center

    to_origin_mat = from_origin_mat.clone()
    to_origin_mat = _paddle_inverse_cast(from_origin_mat)

    # append translation with zeros
    proj_mat = projection_from_Rt(rmat, paddle.zeros_like(center)[..., None])  # Bx3x4

    # chain 4x4 transforms
    proj_mat = convert_affinematrix_to_homography3d(proj_mat)  # Bx4x4
    proj_mat = from_origin_mat @ proj_mat @ to_origin_mat

    return proj_mat[..., :3, :]  # Bx3x4

def _compute_rotation_matrix3d(
    yaw: paddle.Tensor, pitch: paddle.Tensor, roll: paddle.Tensor, center: paddle.Tensor
) -> paddle.Tensor:
    """Compute a pure affine rotation matrix."""
    if len(yaw.shape) == len(pitch.shape) == len(roll.shape) == 0:
        yaw = yaw.unsqueeze(axis=0)
        pitch = pitch.unsqueeze(axis=0)
        roll = roll.unsqueeze(axis=0)

    if len(yaw.shape) == len(pitch.shape) == len(roll.shape) == 1:
        yaw = yaw.unsqueeze(axis=1)
        pitch = pitch.unsqueeze(axis=1)
        roll = roll.unsqueeze(axis=1)

    if not (len(yaw.shape) == len(pitch.shape) == len(roll.shape) == 2):
        raise AssertionError(f"Expected yaw, pitch, roll to be (B, 1). Got {yaw.shape}, {pitch.shape}, {roll.shape}.")

    angles: paddle.Tensor = paddle.concat([yaw, pitch, roll], axis=1)
    scales: paddle.Tensor = paddle.ones_like(yaw)
    matrix: paddle.Tensor = get_projective_transform(center, angles, scales)
    return matrix


def eye_like(n: int, input: Tensor, shared_memory: bool = False) -> Tensor:
    r"""Return a 2-D tensor with ones on the diagonal and zeros elsewhere with the same batch size as the input.

    Args:
        n: the number of rows :math:`(N)`.
        input: image tensor that will determine the batch size of the output matrix.
          The expected shape is :math:`(B, *)`.
        shared_memory: when set, all samples in the batch will share the same memory.

    Returns:
       The identity matrix with the same batch size as the input :math:`(B, N, N)`.

    Notes:
        When the dimension to expand is of size 1, using paddle.expand(...) yields the same tensor as paddle.repeat(...)
        without using extra memory. Thus, when the tensor obtained by this method will be later assigned -
        use this method with shared_memory=False, otherwise, prefer using it with shared_memory=True.
    """
    if n <= 0:
        raise AssertionError(type(n), n)
    if len(input.shape) < 1:
        raise AssertionError(input.shape)

    identity = paddle.eye(n).type(input.dtype)

    return identity[None].expand(input.shape[0], n, n) if shared_memory else identity[None].title([input.shape[0], 1, 1])


def _compute_translation_matrix(translation: paddle.Tensor) -> paddle.Tensor:
    """Compute affine matrix for translation."""
    matrix: paddle.Tensor = eye_like(3, translation, shared_memory=False)

    dx, dy = paddle.chunk(translation, chunks=2, axis=-1)
    matrix[..., 0, 2:3] += dx
    matrix[..., 1, 2:3] += dy
    return matrix


def _compute_scaling_matrix(scale: paddle.Tensor, center: paddle.Tensor) -> paddle.Tensor:
    """Compute affine matrix for scaling."""
    angle: paddle.Tensor = paddle.zeros(scale.shape[:1],dtype=scale.dtype)
    matrix: paddle.Tensor = get_rotation_matrix2d(center, angle, scale)
    return matrix


def _compute_shear_matrix(shear: paddle.Tensor) -> paddle.Tensor:
    """Compute affine matrix for shearing."""
    matrix: paddle.Tensor = eye_like(3, shear, shared_memory=False)

    shx, shy = paddle.chunk(shear, chunks=2, axis=-1)
    matrix[..., 0, 1:2] += shx
    matrix[..., 1, 0:1] += shy
    return matrix


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L166

def _paddle_inverse_cast(input: Tensor) -> Tensor:
    """Helper function to make paddle.inverse work with other than fp32/64.

    The function paddle.inverse is only implemented for fp32/64 which makes impossible to be used by fp16 or others. What
    this function does, is cast input data type to fp32, apply paddle.inverse, and cast back to the input dtype.
    """
    if not isinstance(input, Tensor):
        raise AssertionError(f"Input must be Tensor. Got: {type(input)}.")
    dtype: paddle.dtype = input.dtype
    if dtype not in (paddle.float32, paddle.float64):
        dtype = paddle.float32
    return paddle.inverse(input.casr(dtype)).cast(input.dtype)

def convert_affinematrix_to_homography(A: Tensor) -> Tensor:

    if not isinstance(A, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(A)}")

    if not (len(A.shape) == 3 and A.shape[-2:] == (2, 3)):
        raise ValueError(f"Input matrix must be a Bx2x3 tensor. Got {A.shape}")

    return _convert_affinematrix_to_homography_impl(A)


def normal_transform_pixel(
    height: int,
    width: int,
    eps: float = 1e-14,
    dtype: Optional[paddle.dtype] = None,
) -> Tensor:
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].

    Args:
        height image height.
        width: image width.
        eps: epsilon to prevent divide-by-zero errors

    Returns:
        normalized transform with shape :math:`(1, 3, 3)`.
    """
    tr_mat = paddle.to_tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]], dtype=dtype)  # 3x3

    # prevent divide by zero bugs
    width_denom: float = eps if width == 1 else width - 1.0
    height_denom: float = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

    return tr_mat.unsqueeze(0)  # 1x3x3

def _torch_inverse_cast(input: paddle.Tensor) -> paddle.Tensor:
    """Helper function to make torch.inverse work with other than fp32/64.

    The function torch.inverse is only implemented for fp32/64 which makes impossible to be used by fp16 or others. What
    this function does, is cast input data type to fp32, apply torch.inverse, and cast back to the input dtype.
    """
    if not isinstance(input, paddle.Tensor):
        raise AssertionError(f"Input must be torch.Tensor. Got: {type(input)}.")
    dtype: paddle.dtype = input.dtype
    if dtype not in (paddle.float32, paddle.float64):
        dtype = paddle.float32
    return paddle.inverse(input.cast(dtype)).cast(input.dtype)

def normalize_homography(
    dst_pix_trans_src_pix: Tensor, dsize_src: Tuple[int, int], dsize_dst: Tuple[int, int]
) -> Tensor:
    r"""Normalize a given homography in pixels to [-1, 1].

    Args:
        dst_pix_trans_src_pix: homography/ies from source to destination to be
          normalized. :math:`(B, 3, 3)`
        dsize_src: size of the source image (height, width).
        dsize_dst: size of the destination image (height, width).

    Returns:
        the normalized homography of shape :math:`(B, 3, 3)`.
    """
    if not isinstance(dst_pix_trans_src_pix, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(dst_pix_trans_src_pix)}")

    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError(f"Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {dst_pix_trans_src_pix.shape}")

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: Tensor = normal_transform_pixel(src_h, src_w).to(dst_pix_trans_src_pix)

    src_pix_trans_src_norm = _torch_inverse_cast(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix: Tensor = normal_transform_pixel(dst_h, dst_w).to(dst_pix_trans_src_pix)

    # compute chain transformations
    dst_norm_trans_src_norm: Tensor = dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm


def warp_affine(
    src,
    M,
    dsize,
    mode = 'bilinear',
    padding_mode= 'zeros',
    align_corners = True,
    fill_value = paddle.zeros([3]),  # needed for jit
) -> Tensor:
  
    if not isinstance(src, Tensor):
        raise TypeError(f"Input src type is not a Tensor. Got {type(src)}")

    if not isinstance(M, Tensor):
        raise TypeError(f"Input M type is not a Tensor. Got {type(M)}")

    if not len(src.shape) == 4:
        raise ValueError(f"Input src must be a BxCxHxW tensor. Got {src.shape}")

    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError(f"Input M must be a Bx2x3 tensor. Got {M.shape}")

    # fill padding is only supported for 3 channels because we can't set fill_value default
    # to None as this gives jit issues.
    if padding_mode == "fill" and fill_value.shape != paddle.Size([3]):
        raise ValueError(f"Padding_tensor only supported for 3 channels. Got {fill_value.shape}")

    B, C, H, W = src.shape

    # we generate a 3x3 transformation matrix from 2x3 affine
    M_3x3: Tensor = convert_affinematrix_to_homography(M)
    dst_norm_trans_src_norm: Tensor = normalize_homography(M_3x3, (H, W), dsize)

    # src_norm_trans_dst_norm = paddle.inverse(dst_norm_trans_src_norm)
    src_norm_trans_dst_norm = _paddle_inverse_cast(dst_norm_trans_src_norm)

    grid = F.affine_grid(src_norm_trans_dst_norm[:, :2, :], [B, C, dsize[0], dsize[1]], align_corners=align_corners)

    if padding_mode == "fill":
        return _fill_and_warp(src, grid, align_corners=align_corners, mode=mode, fill_value=fill_value)
    return F.grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode)


def _fill_and_warp(src: Tensor, grid: Tensor, mode: str, align_corners: bool, fill_value: Tensor) -> Tensor:
    r"""Warp a mask of ones, then multiple with fill_value and add to default warp.

    Args:
        src: input tensor of shape :math:`(B, 3, H, W)`.
        grid: grid tensor from `transform_points`.
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        align_corners: interpolation flag.
        fill_value: tensor of shape :math:`(3)` that fills the padding area. Only supported for RGB.

    Returns:
        the warped and filled tensor with shape :math:`(B, 3, H, W)`.
    """
    ones_mask = paddle.ones_like(src)
    fill_value = fill_value.to(ones_mask)[None, :, None, None]  # cast and add dimensions for broadcasting
    inv_ones_mask = 1 - F.grid_sample(ones_mask, grid, align_corners=align_corners, mode=mode, padding_mode="zeros")
    inv_color_mask = inv_ones_mask * fill_value
    return F.grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode="zeros") + inv_color_mask


def affine(
    tensor: paddle.Tensor,
    matrix: paddle.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = True,
) -> paddle.Tensor:
    r"""Apply an affine transformation to the image.

    .. image:: _static/img/warp_affine.png

    Args:
        tensor: The image tensor to be warped in shapes of
            :math:`(H, W)`, :math:`(D, H, W)` and :math:`(B, C, H, W)`.
        matrix: The 2x3 affine transformation matrix.
        mode: interpolation mode to calculate output values ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The warped image with the same shape as the input.

    Example:

        paddle.Size([1, 2, 3, 5])
    """
    # warping needs data in the shape of BCHW
    is_unbatched: bool = tensor.ndimension() == 3
    if is_unbatched:
        tensor = paddle.unsqueeze(tensor, axis=0)

    # we enforce broadcasting since by default grid_sample it does not
    # give support for that
    matrix = matrix.expand(tensor.shape[0], -1, -1)

    # warp the input tensor
    height: int = tensor.shape[-2]
    width: int = tensor.shape[-1]
    warped: paddle.Tensor = warp_affine(tensor, matrix, (height, width), mode, padding_mode, align_corners)

    # return in the original shape
    if is_unbatched:
        warped = paddle.squeeze(warped, axis=0)

    return warped


def rotate(
    tensor: paddle.Tensor,
    angle: paddle.Tensor,
    center: Union[None, paddle.Tensor] = None,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = True,
) -> paddle.Tensor:
    r"""Rotate the tensor anti-clockwise about the center.

    .. image:: _static/img/rotate.png

    Args:
        tensor: The image tensor to be warped in shapes of :math:`(B, C, H, W)`.
        angle: The angle through which to rotate. The tensor
          must have a shape of (B), where B is batch size.
        center: The center through which to rotate. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The rotated tensor with shape as input.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       rotate_affine.html>`__.

    Example:
        paddle.Size([1, 3, 4, 4])
    """
    if not isinstance(tensor, paddle.Tensor):
        raise TypeError(f"Input tensor type is not a paddle.Tensor. Got {type(tensor)}")

    if not isinstance(angle, paddle.Tensor):
        raise TypeError(f"Input angle type is not a paddle.Tensor. Got {type(angle)}")

    if center is not None and not isinstance(center, paddle.Tensor):
        raise TypeError(f"Input center type is not a paddle.Tensor. Got {type(center)}")

    if len(tensor.shape) not in (3, 4):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. " "Got: {}".format(tensor.shape))

    # compute the rotation center
    if center is None:
        center = _compute_tensor_center(tensor)

    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    angle = angle.expand(tensor.shape[0])
    center = center.expand(tensor.shape[0], -1)
    rotation_matrix: paddle.Tensor = _compute_rotation_matrix(angle, center)

    # warp using the affine transform
    return affine(tensor, rotation_matrix[..., :2, :3], mode, padding_mode, align_corners)


def translate(
    tensor: paddle.Tensor,
    translation: paddle.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = True,
) -> paddle.Tensor:
    r"""Translate the tensor in pixel units.

    .. image:: _static/img/translate.png

    Args:
        tensor: The image tensor to be warped in shapes of :math:`(B, C, H, W)`.
        translation: tensor containing the amount of pixels to
          translate in the x and y direction. The tensor must have a shape of
          (B, 2), where B is batch size, last dimension contains dx dy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The translated tensor with shape as input.

    Example:

        paddle.Size([1, 3, 4, 4])
    """
    if not isinstance(tensor, paddle.Tensor):
        raise TypeError(f"Input tensor type is not a paddle.Tensor. Got {type(tensor)}")

    if not isinstance(translation, paddle.Tensor):
        raise TypeError(f"Input translation type is not a paddle.Tensor. Got {type(translation)}")

    if len(tensor.shape) not in (3, 4):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. " "Got: {}".format(tensor.shape))

    # compute the translation matrix
    translation_matrix: paddle.Tensor = _compute_translation_matrix(translation)

    # warp using the affine transform
    return affine(tensor, translation_matrix[..., :2, :3], mode, padding_mode, align_corners)


def scale(
    tensor: paddle.Tensor,
    scale_factor: paddle.Tensor,
    center: Union[None, paddle.Tensor] = None,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = True,
) -> paddle.Tensor:
    r"""Scale the tensor by a factor.

    .. image:: _static/img/scale.png

    Args:
        tensor: The image tensor to be warped in shapes of :math:`(B, C, H, W)`.
        scale_factor: The scale factor apply. The tensor
          must have a shape of (B) or (B, 2), where B is batch size.
          If (B), isotropic scaling will perform.
          If (B, 2), x-y-direction specific scaling will perform.
        center: The center through which to scale. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The scaled tensor with the same shape as the input.

    Example:
)
        paddle.Size([1, 3, 4, 4])
    """
    if not isinstance(tensor, paddle.Tensor):
        raise TypeError(f"Input tensor type is not a paddle.Tensor. Got {type(tensor)}")

    if not isinstance(scale_factor, paddle.Tensor):
        raise TypeError(f"Input scale_factor type is not a paddle.Tensor. Got {type(scale_factor)}")

    if len(scale_factor.shape) == 1:
        # convert isotropic scaling to x-y direction
        scale_factor = scale_factor.repeat(1, 2)

    # compute the tensor center
    if center is None:
        center = _compute_tensor_center(tensor)

    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    center = center.expand(tensor.shape[0], -1)
    scale_factor = scale_factor.expand(tensor.shape[0], 2)
    scaling_matrix: paddle.Tensor = _compute_scaling_matrix(scale_factor, center)

    # warp using the affine transform
    return affine(tensor, scaling_matrix[..., :2, :3], mode, padding_mode, align_corners)


def shear(
    tensor: paddle.Tensor,
    shear: paddle.Tensor,
    mode: str = 'bilinear',
    padding_mode: str = 'zeros',
    align_corners: bool = False,
) -> paddle.Tensor:
    r"""Shear the tensor.

    .. image:: _static/img/shear.png

    Args:
        tensor: The image tensor to be skewed with shape of :math:`(B, C, H, W)`.
        shear: tensor containing the angle to shear
          in the x and y direction. The tensor must have a shape of
          (B, 2), where B is batch size, last dimension contains shx shy.
        mode: interpolation mode to calculate output values
          ``'bilinear'`` | ``'nearest'``.
        padding_mode: padding mode for outside grid values
          ``'zeros'`` | ``'border'`` | ``'reflection'``.
        align_corners: interpolation flag.

    Returns:
        The skewed tensor with shape same as the input.

    Example:

        paddle.Size([1, 3, 4, 4])
    """
    if not isinstance(tensor, paddle.Tensor):
        raise TypeError(f"Input tensor type is not a paddle.Tensor. Got {type(tensor)}")

    if not isinstance(shear, paddle.Tensor):
        raise TypeError(f"Input shear type is not a paddle.Tensor. Got {type(shear)}")

    if len(tensor.shape) not in (3, 4):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. " "Got: {}".format(tensor.shape))

    # compute the translation matrix
    shear_matrix: paddle.Tensor = _compute_shear_matrix(shear)

    # warp using the affine transform
    return affine(tensor, shear_matrix[..., :2, :3], mode, padding_mode, align_corners)


def _side_to_image_size(side_size: int, aspect_ratio: float, side: str = "short") -> Tuple[int, int]:
    if side not in ("short", "long", "vert", "horz"):
        raise ValueError(f"side can be one of 'short', 'long', 'vert', and 'horz'. Got '{side}'")
    if side == "vert":
        return side_size, int(side_size * aspect_ratio)
    if side == "horz":
        return int(side_size / aspect_ratio), side_size
    if (side == "short") ^ (aspect_ratio < 1.0):
        return side_size, int(side_size * aspect_ratio)
    return int(side_size / aspect_ratio), side_size


class Rotate(nn.Layer):


    def __init__(
        self,
        angle: paddle.Tensor,
        center: Union[None, paddle.Tensor] = None,
        mode: str = 'bilinear',
        padding_mode: str = 'zeros',
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        self.angle: paddle.Tensor = angle
        self.center: Union[None, paddle.Tensor] = center
        self.mode: str = mode
        self.padding_mode: str = padding_mode
        self.align_corners: bool = align_corners

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        return rotate(input, self.angle, self.center, self.mode, self.padding_mode, self.align_corners)

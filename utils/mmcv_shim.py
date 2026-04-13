import os
import cv2
import numpy as np


def imread(filename, flag='color'):
    if flag in ('unchanged', 'unchanged_ignore_orientation'):
        cv_flag = cv2.IMREAD_UNCHANGED
    elif flag in ('grayscale', 'gray'):
        cv_flag = cv2.IMREAD_GRAYSCALE
    else:
        cv_flag = cv2.IMREAD_COLOR
    img = cv2.imread(filename, cv_flag)
    if img is None:
        raise FileNotFoundError(filename)
    return img


def _interp(interpolation):
    if interpolation == 'nearest':
        return cv2.INTER_NEAREST
    if interpolation == 'area':
        return cv2.INTER_AREA
    if interpolation == 'bicubic':
        return cv2.INTER_CUBIC
    return cv2.INTER_LINEAR


def imresize(img, size, return_scale=False, interpolation='bilinear', backend='cv2'):
    target_w, target_h = size
    h, w = img.shape[:2]
    out = cv2.resize(img, (int(target_w), int(target_h)), interpolation=_interp(interpolation))
    if return_scale:
        w_scale = float(target_w) / float(w)
        h_scale = float(target_h) / float(h)
        return out, w_scale, h_scale
    return out


def imrescale(img, scale, return_scale=False, interpolation='bilinear', backend='cv2'):
    target_w, target_h = scale
    h, w = img.shape[:2]
    scale_factor = min(float(target_w) / float(w), float(target_h) / float(h))
    new_w = max(int(w * scale_factor + 0.5), 1)
    new_h = max(int(h * scale_factor + 0.5), 1)
    out = cv2.resize(img, (new_w, new_h), interpolation=_interp(interpolation))
    if return_scale:
        return out, scale_factor
    return out


def imflip(img, direction='horizontal'):
    if direction == 'horizontal':
        return cv2.flip(img, 1)
    if direction == 'vertical':
        return cv2.flip(img, 0)
    if direction == 'diagonal':
        return cv2.flip(img, -1)
    raise ValueError(f'Unsupported flip direction: {direction}')


def imrotate(img, angle, center=None, scale=1.0, border_value=0, interpolation='bilinear', auto_bound=False):
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    if auto_bound:
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]
        out_size = (new_w, new_h)
    else:
        out_size = (w, h)
    return cv2.warpAffine(img, matrix, out_size, flags=_interp(interpolation), borderValue=border_value)


def impad(img, shape=None, padding=None, pad_val=0):
    if padding is not None:
        if isinstance(padding, int):
            left = right = top = bottom = padding
        elif len(padding) == 2:
            left = right = int(padding[0])
            top = bottom = int(padding[1])
        elif len(padding) == 4:
            left, top, right, bottom = map(int, padding)
        else:
            raise ValueError('padding must be int, (w,h), or (l,t,r,b)')
        h, w = img.shape[:2]
        target_h = h + top + bottom
        target_w = w + left + right
        result = np.full((target_h, target_w, *img.shape[2:]), pad_val, dtype=img.dtype) if img.ndim == 3 else np.full((target_h, target_w), pad_val, dtype=img.dtype)
        result[top:top + h, left:left + w, ...] = img
        return result
    if shape is None:
        return img
    target_h, target_w = shape
    h, w = img.shape[:2]
    result = np.full((target_h, target_w, *img.shape[2:]), pad_val, dtype=img.dtype) if img.ndim == 3 else np.full((target_h, target_w), pad_val, dtype=img.dtype)
    result[:h, :w, ...] = img
    return result


def impad_to_multiple(img, divisor, pad_val=0):
    h, w = img.shape[:2]
    target_h = int(np.ceil(h / divisor)) * divisor
    target_w = int(np.ceil(w / divisor)) * divisor
    return impad(img, shape=(target_h, target_w), pad_val=pad_val)


def imnormalize(img, mean, std, to_rgb=True):
    img = img.astype(np.float32)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    if img.ndim == 3 and to_rgb:
        img = img[..., ::-1]
    return (img - mean) / std


def mkdir_or_exist(path):
    os.makedirs(path, exist_ok=True)

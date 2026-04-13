import re
import torch
from torch import Tensor
from typing import Optional, List
from collections import defaultdict


np_str_obj_array_pattern = re.compile(r'[SaUO]')


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def _normalize_to_3_channels(tensor: Tensor) -> Tensor:
    """Görüntü tensorunu 3 kanala getirir: 2 kanal -> pad, >3 kanal -> kes."""
    if tensor.dim() < 3 or tensor.shape[0] == 3:
        return tensor
    c = tensor.shape[0]
    if c == 2:
        return torch.cat([tensor, tensor[-1:]], dim=0)
    return tensor[:3]


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # if torchvision._is_tracing():
    #     # nested_tensor_from_tensor_list() does not export well to ONNX
    #     # call _onnx_nested_tensor_from_tensor_list() instead
    #     return _onnx_nested_tensor_from_tensor_list(tensor_list)
    # Sadece kanal-benzeri (C,H,W) tensorları 3 kanala getir; (H,W,L) mask'lara dokunma.
    # Karışık batch'ta ilk eleman mask (512,...) olabilir; sadece dim0 in (2,3,4) olanları normalize et.
    if tensor_list and len(tensor_list[0].shape) >= 3:
        tensor_list = [_normalize_to_3_channels(t) if t.shape[0] in (2, 3, 4) else t for t in tensor_list]
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    batch_shape = [len(tensor_list)] + max_size
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    for i, img in enumerate(tensor_list):
        if len(img.shape) == 1:
            tensor[i, ..., :img.shape[-1]].copy_(img)
        else:
            # Tüm boyutları açıkça dilimle; "..." (3,C,H,W) ile (3,H,W,L) karışınca yanlış hedef veriyordu
            dest_slice = (i,) + tuple(slice(0, s) for s in img.shape)
            tensor[dest_slice].copy_(img)
    return tensor


def collate_fn(data_list):
    result = defaultdict(list)
    for data in data_list:
        for key, value in data.items():
            result[key].append(value)
    return result


def collate_fn_padded(data_list, keys=['img', 'img_mask']):
    result = defaultdict(list)
    for data in data_list:
        for key, value in data.items():
            result[key].append(value)
    for key in result:
        if key not in keys:
            continue
        if not all([isinstance(x, torch.Tensor) for x in result[key]]):
            continue
        # result[key] = torch.stack(result[key], dim=0)
        result[key] = nested_tensor_from_tensor_list(result[key])
    return result


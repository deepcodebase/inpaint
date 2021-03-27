import hashlib
import os
import time
import sys
from pathlib import Path
from typing import Union, Text, List, BinaryIO, Optional

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid


def time2str(time_used):
    gaps = [
        ('days', 86400000),
        ('h', 3600000),
        ('min', 60000),
        ('s', 1000),
        ('ms', 1)
    ]
    time_used *= 1000
    time_str = []
    for unit, gap in gaps:
        val = time_used // gap
        if val > 0:
            time_str.append('{}{}'.format(int(val), unit))
            time_used -= val * gap
    if len(time_str) == 0:
        time_str.append('0ms')
    return ' '.join(time_str)


def get_date():
    return time.strftime('%Y-%m-%d', time.localtime(time.time()))

def get_time(t=None):
    if t is None:
        t = time.time()
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))


def hash_seed(*items, width=32):
    # width: range of seed: [0, 2**width)
    sha = hashlib.sha256()
    for item in items:
        sha.update(str(item).encode('utf-8'))
    return int(sha.hexdigest()[23:23+width//4], 16)


def resize_like(x, target, mode='bilinear'):
    if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
        return F.interpolate(
            x, target.shape[-2:], mode=mode, align_corners=False)
    else:
        return F.interpolate(x, target.shape[-2:], mode=mode)


def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs
) -> None:
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format, quality=95)
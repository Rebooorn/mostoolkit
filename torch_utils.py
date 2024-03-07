import torch.nn.functional as f
import numpy as np
import torch.nn as nn
from monai.losses import DiceLoss, DiceCELoss

def pad_or_crop_to_target(source, target):
    # Only HWL will be padded
    source_shape = source.size()[2:]
    target_shape = target.size()[2:]

    # Calculate the required padding or cropping for each dimension
    pad = [0, 0, 0, 0]
    crop = [slice(None), slice(None)]
    for s, t in zip(source_shape, target_shape):
        diff = t - s
        if diff > 0:
            # Pad the source tensor
            pad_ = [diff // 2, diff - diff // 2]
            pad.extend(pad_)
            crop.append(slice(None))
        elif diff < 0:
            # Crop the source tensor
            crop_ = slice(-diff // 2, diff+(-diff // 2))
            pad.extend([0, 0])
            crop.append(crop_) 
        else:
            # No padding or cropping needed for this dimension
            pad.extend([0, 0])
            crop.append(slice(None))

    # Pad or crop the source tensor
    # print(padding_or_cropping)
    cropped_source = source[crop]
    modified_source = f.pad(cropped_source, tuple(pad[::-1]))
    return modified_source

def parse_loss_function_by_name(name, **kwargs) -> nn.Module:
    if name == 'dice':
        return DiceLoss(include_background=False, to_onehot_y=True, sigmoid=True, **kwargs)
    if name == 'dicece':
        return DiceCELoss(include_background=False, to_onehot_y=True, sigmoid=True, **kwargs)
    if name == 'ce':
        return DiceCELoss(include_background=False, to_onehot_y=True, sigmoid=True, lambda_dice=0.0, **kwargs)
    


if __name__ == '__main__':
    pass
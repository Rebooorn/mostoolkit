import pandas as pd
from pathlib import Path
import SimpleITK as sitk
from skimage.transform import resize
import numpy as np
import h5py
import shutil
import scipy
from mostoolkit.io_utils import load_json, load_pickle
from mostoolkit.nnunet.nnunet_utils import nnunet_normalize_np

def resample(image, spacing, new_spacing=[1,1,1], order=3):
    # .mhd image order : z, y, x
    # order is used in interpolate
    if not isinstance(spacing, np.ndarray):
        spacing = np.array(spacing)
    if not isinstance(new_spacing, np.ndarray):
        new_spacing = np.array(new_spacing)
    spacing = spacing[::-1]
    new_spacing = new_spacing[::-1]

    for i in range(3):
        if new_spacing[i] < 0:
            new_spacing[i] = spacing[i]
    # print(new_spacing, spacing)

    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=order, mode='nearest')
    return image

def crop_to_standard(scan, scale: list):
    z, y, x = scan.shape    
    zscale, yscale, xscale = scale

    if zscale < 0:
        # keep original shape
        ret_scan = scan
    elif z >= zscale:
        ret_scan = scan[(z-zscale)//2:(z+zscale)//2, :, :]
    else:        
        temp1 = np.zeros(((zscale-z)//2, y, x))
        temp2 = np.zeros(((zscale-z)-(zscale-z)//2, y, x))
        ret_scan = np.concatenate((temp1, scan, temp2), axis=0)

    z, y, x = ret_scan.shape
    if yscale < 0:
        ret_scan = ret_scan
    elif y >= yscale:
        ret_scan = ret_scan[:, (y-yscale)//2:(y+yscale)//2, :]
    else:
        temp1 = np.zeros((z, (yscale-y)//2, x))
        temp2 = np.zeros((z, (yscale-y)-(yscale-y)//2, x))
        ret_scan = np.concatenate((temp1, ret_scan, temp2), axis=1)

    z, y, x = ret_scan.shape
    if xscale < 0:
        ret_scan = ret_scan
    elif x >= xscale:
        ret_scan = ret_scan[:, :, (x-xscale)//2:(x+xscale)//2]
    else:
        temp1 = np.zeros((z, y, (xscale-x)//2))
        temp2 = np.zeros((z, y, (xscale-x)-(xscale-x)//2))
        ret_scan = np.concatenate((temp1, ret_scan, temp2), axis=2)
    return ret_scan

### Utils for segmentation map.
def map_seg_mask(seg_mask: np.ndarray, map_dict):
    ret = np.zeros_like(seg_mask)
    for k in map_dict.keys():
        ret[seg_mask==k] = map_dict[k]
    return ret
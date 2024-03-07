from PIL import Image
import numpy as np
import SimpleITK as sitk
import monai
from argparse import Namespace
import json, yaml
import h5py
import pickle

def dict_to_args(dict_in):
    args_out = Namespace(**dict_in)
    return args_out


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def load_yaml(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_yaml(path, data):
    with open(path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)

def normalize_to_png(array):
    # normalize [min, max] to [0,255]
    array = array - np.amin(array)
    array = array / np.amax(array) * 255
    return array.astype(np.uint8)
    
## SITK utils for nifti save and load
def sitk_load_with_metadata(fname):
    '''
    ret: img, direction, spacing, origin, size
    '''
    img_itk = sitk.ReadImage(fname)
    img = sitk.GetArrayFromImage(img_itk)
    return img, img_itk.GetDirection(), img_itk.GetSpacing(), img_itk.GetOrigin(), img_itk.GetSize()

def sitk_save(fname, volume, spacing, origin, direction):
    itkimage = sitk.GetImageFromArray(volume, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    itkimage.SetDirection(direction)
    sitk.WriteImage(itkimage, fname, True)
    return True

## load and (save) h5py
def h5py_load(fname, key):
    data = h5py.File(fname, 'r')
    return np.asarray(data[key])

def save_to_png(array, filename):
    # print(array.shape, array.dtype, np.amin(array), np.amax(array))
    im = Image.fromarray(array)
    im.save(filename)

def normalize_and_save_to_png(array, filename):
    array = normalize_to_png(array)
    save_to_png(array, filename)

def combine_and_save_to_png(array, filename):
    n, _, w, h = array.shape
    nw = int(np.sqrt(n))
    nh = int(np.ceil(n / nw))
    im_w = nw * w
    im_h = nh * h
    all_array = np.zeros([3, im_w, im_h], dtype=np.float32)

    w_loc = 0
    h_loc = 0
    for i in range(n):
        im = array[i]
        w_start = w * (w_loc)
        h_start = h * (h_loc)
        all_array[:, w_start:w_start+w, h_start:h_start+h] = im
        h_loc += 1
        if h_loc > (nh-1):
            # go to next row
            h_loc = 0
            w_loc += 1
        # print(i)
    all_array = all_array.transpose(1,2,0)
    save_to_png(normalize_to_png(all_array), filename)


'''io utils'''
def monai_save_nifti(img, fname, from_meta_dict):
    affine = from_meta_dict['affine']
    original_affine = from_meta_dict['original_affine']
    output_spatial_shape = from_meta_dict['dim'][1:4]
    monai.data.write_nifti(img,
                           file_name=fname,
                           affine=affine,
                           target_affine=original_affine,
                           resample=True,
                           mode='bilinear',
                           output_spatial_shape=output_spatial_shape,
                           output_dtype=np.float32
                           )

if __name__ == '__main__':
    n, _, w, h = 3, 1, 320, 120
    nw = int(np.sqrt(n))
    nh = int(np.ceil(n / nw))
    im_w = nw * w
    im_h = nh * h
    all_array = np.zeros([3, im_w, im_h], dtype=np.float32)

    w_loc = 0
    h_loc = 0
    for i in range(n):
        
        # all_array[:, w_start:w_start+w, h_start:h_start+h] = im
        print(w_loc, h_loc)
        h_loc += 1
        if h_loc > (nh-1):
            # go to next row
            h_loc = 0
            w_loc += 1

    # all_array = all_array.transpose(1,2,0)
    # save_to_png(normalize_to_png(all_array), filename)
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
from pathlib import Path


'''
Utils to help visulize CT images
'''
colormap = {
    0:     [0,     0,    0],
    1:   [255,    0,    0],  
    2:     [0,  255,    0],  
    3:     [0,    0,  255],  
    4:   [255,  255,    0],  
    5:     [0,  255,  255],  
    6:   [255,    0,  255],  
    7:   [255,  239,  213],  
    8:   [245,  245,  220], 
    9:   [186,   85,  211], 
   10:   [255,  228,  181], 
   11:   [210,  105,   30], 
   12:   [255,  248,  220], 
   13:     [0,  139,  139], 
   14:    [46,  139,   87], 
   15:   [255,  228,  225], 
   16:   [106,   90,  205], 
   17:   [221,  160,  221], 
   20:   [255,  250,  250], 
   21:   [147,  112,  219], 
   23:    [75,    0,  130], 
   40:   [238,  130,  238], 
   47:    [25,   25,  112], 
   49:    [34,  139,   34], 
   50:   [248,  248,  255], 
   52:   [255,  160,  122], 
   55:    [65,  105,  225], 
   57:   [250,  240,  230], 
   58:   [128,    0,    0], 
   59:    [50,  205,   50], 
   60:   [244,  164,   96], 
   61:   [255,  255,  240], 
   62:   [123,  104,  238], 
   63:   [255,  165,    0], 
   64:   [173,  216,  230], 
   65:   [255,  192,  203], 
   66:   [127,  255,  212], 
   67:   [255,  140,    0], 
   68:   [143,  188,  143], 
   70:   [253,  245,  230], 
   71:   [255,  250,  240], 
   72:     [0,  206,  209], 
   73:     [0,  255,  127], 
   74:   [128,    0,  128], 
   75:   [255,  250,  205], 
   76:   [250,  128,  114], 
   77:   [148,    0,  211], 
   78:   [178,   34,   34], 
   79:   [255,  127,   80], 
   80:   [135,  206,  235], 
   82:   [240,  230,  140], 
   84:   [255,  245,  238], 
   85:   [107,  142,   35], 
   86:   [135,  206,  250], 
   87:     [0,    0,  139], 
   88:   [139,    0,  139], 
   89:   [245,  245,  220], 
   90:   [186,   85,  211], 
   91:   [255,  228,  181], 
   94:   [210,  105,   30], 
   95:   [255,  248,  220], 
   97:    [72,   61,  139], 
   99:   [128,  128,    0], 
  100:   [176,  224,  230], 
  101:   [255,  240,  245], 
    }

def map_label_to_rgb(idx):
    return colormap[idx]


def contrast_adapt(img, contrast_after, crop=True):
    # ret will be [0, 1]
    # contrast_after = (min, max)
    vmin, vmax = contrast_after
    if crop:
        img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin)
    return img

def map_segmask_rgb(seg, nclasses=-1):
    ''' map segmentation mask to rgb for visualization, if nclasses<0, it will be inferred by the input '''
    shape = seg.shape
    ret = np.zeros([*shape, 3], dtype=np.uint8)
    if nclasses < 0:
        nclasses = np.max(seg)+1
    
    for i in range(nclasses):
        ret[seg == i] = map_label_to_rgb(i)
    # print(ret.shape)
    return ret

def map_segmask(seg, map_dict: dict):
    ''' map 0 to [0,0,0], 1 to [255,0,0], 2 to [0,255,0], 3 to [0,0,255] '''
    shape = seg.shape
    ret = np.zeros([*shape, 3], dtype=np.uint8)
    for k in map_dict.keys():
        ret[seg==k] = map_dict[k]
    return ret

def plt_save_pdf(img, save_fname, save_root):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(str(save_root / save_fname), bbox_inches='tight', pad_inches=0)

def plt_save_pdf_with_mask(img, seg, map_dict, save_fname, save_root):
    plt.imshow(img, cmap='gray')
    plt.imshow(map_segmask(seg, map_dict), alpha=0.7)
    plt.axis('off')
    plt.savefig(str(save_root / save_fname), bbox_inches='tight', pad_inches=0)

def plt_nice_imshow(img, save=None):
    assert len(img.shape) == 2
    plt.imshow(img, cmap='gray')
    plt.axis(False)
    if not save is None:
        plt.savefig(str(save), bbox_inches='tight', pad_inches=0)
    plt.show()

def gray_to_rgb(image):
    assert len(image.shape) == 2
    return np.stack([image, image, image], axis=-1)

def plt_nice_imshow_seg(img, seg, nclasses=-1, map_dict=None, map_fn=None, save=None, alpha=0.5):
    assert len(img.shape) == 2
    img = gray_to_rgb(img)
    if nclasses > 0:
        seg_ = map_segmask_rgb(seg, nclasses)
    elif map_dict is not None:
        seg_ = map_segmask(seg, map_dict)
    elif map_fn is not None:
        seg_ = map_fn(seg)
    else:
        NotImplementedError('mapping method missing')
    overlaid_image = np.copy(img)
    overlaid_image[seg > 0] = (1 - alpha) * img[seg > 0] + alpha * seg_[seg > 0]
    plt.imshow(overlaid_image)
    plt.axis(False)
    if not save is None:
        plt.savefig(str(save), bbox_inches='tight', pad_inches=0)
    plt.show()
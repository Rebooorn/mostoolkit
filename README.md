
---

<div align="center">    
 
# Multi Organ Segmentation Toolkit

</div>
 
## Description   
This toolkit is part of my PhD research. I gathered some utils for more convenient of:  
- load/save nifti images
- visualize (pretty) 3D images 
- basic intensity/shape preprocessing on CT images

## Installation
First, install dependencies   
```bash
# Unzip the project
cd mostoolkit

# Make sure PyTorch is already installed 
# Use pip to install this package, and the dependencies to your python environment.
pip install -r requirements.txt
pip install -e .   
 ```   

## Examples
1. Load/save a CT image
```python
from mostoolkit.io_utils import sitk_load_with_metadata, sitk_save
ct = r'example.nii.gz'
img, d, s, o, _ = sitk_load_with_metadata(ct)
'''
Something happens here
'''
save_name = r'target.nii.gz'
sitk_save(save_name, img, s, o, d)
```
2. Visualize a 3D CT image in a sliding window
```python
from mostoolkit.vis_utils import slice_visualize_X
from mostoolkit.io_utils import sitk_load_with_metadata
ct = r'example.nii.gz'
img = sitk_load_with_metadata(ct)[0]
slice_visualize_X(img)
```

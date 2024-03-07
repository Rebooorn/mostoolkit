from contextlib import closing
from functools import partial
import multiprocessing
import numpy as np
import scipy.ndimage
import os
from pathlib import Path
from mostoolkit.io_utils import sitk_load_with_metadata, h5py_load

# neighbour_code_to_normals is a lookup table.
# For every binary neighbour code 
# (2x2x2 neighbourhood = 8 neighbours = 8 bits = 256 codes) 
# it contains the surface normals of the triangles (called "surfel" for 
# "surface element" in the following). The length of the normal 
# vector encodes the surfel area.
#
# created by compute_surface_area_lookup_table.ipynb using the 
# marching_cube algorithm, see e.g. https://en.wikipedia.org/wiki/Marching_cubes
# https://arxiv.org/pdf/1809.04430.pdf
#
neighbour_code_to_normals = [
  [[0,0,0]],
  [[0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125]],
  [[-0.25,-0.25,0.0],[0.25,0.25,-0.0]],
  [[0.125,-0.125,0.125]],
  [[-0.25,-0.0,-0.25],[0.25,0.0,0.25]],
  [[0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[0.5,0.0,-0.0],[0.25,0.25,0.25],[0.125,0.125,0.125]],
  [[-0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,0.125,0.125]],
  [[-0.25,0.0,0.25],[-0.25,0.0,0.25]],
  [[0.5,0.0,0.0],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[0.5,0.0,0.0],[0.25,-0.25,0.25],[-0.125,0.125,-0.125]],
  [[-0.5,0.0,0.0],[-0.25,0.25,0.25],[-0.125,0.125,0.125]],
  [[0.5,0.0,0.0],[0.5,0.0,0.0]],
  [[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25]],
  [[-0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,-0.5,0.0],[0.25,0.25,0.25],[0.125,0.125,0.125]],
  [[0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,0.0,-0.5],[0.25,0.25,0.25],[-0.125,-0.125,-0.125]],
  [[-0.125,-0.125,0.125],[0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[-0.125,-0.125,-0.125],[-0.25,-0.25,-0.25],[0.25,0.25,0.25],[0.125,0.125,0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25],[-0.125,0.125,0.125]],
  [[-0.25,0.0,0.25],[-0.25,0.0,0.25],[0.125,-0.125,-0.125]],
  [[0.125,0.125,0.125],[0.375,0.375,0.375],[0.0,-0.25,0.25],[-0.25,0.0,0.25]],
  [[0.125,-0.125,-0.125],[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[0.375,0.375,0.375],[0.0,0.25,-0.25],[-0.125,-0.125,-0.125],[-0.25,0.25,0.0]],
  [[-0.5,0.0,0.0],[-0.125,-0.125,-0.125],[-0.25,-0.25,-0.25],[0.125,0.125,0.125]],
  [[-0.5,0.0,0.0],[-0.125,-0.125,-0.125],[-0.25,-0.25,-0.25]],
  [[0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125]],
  [[0.0,-0.25,0.25],[0.0,0.25,-0.25]],
  [[0.0,-0.5,0.0],[0.125,0.125,-0.125],[0.25,0.25,-0.25]],
  [[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.125,-0.125,0.125],[-0.25,-0.0,-0.25],[0.25,0.0,0.25]],
  [[0.0,-0.25,0.25],[0.0,0.25,-0.25],[0.125,-0.125,0.125]],
  [[-0.375,-0.375,0.375],[-0.0,0.25,0.25],[0.125,0.125,-0.125],[-0.25,-0.0,-0.25]],
  [[-0.125,0.125,0.125],[0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125],[-0.125,0.125,0.125]],
  [[-0.0,0.0,0.5],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.25,0.25,-0.25],[0.25,0.25,-0.25],[0.125,0.125,-0.125],[-0.125,-0.125,0.125]],
  [[0.125,-0.125,0.125],[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[0.5,0.0,0.0],[0.25,-0.25,0.25],[-0.125,0.125,-0.125],[0.125,-0.125,0.125]],
  [[0.0,0.25,-0.25],[0.375,-0.375,-0.375],[-0.125,0.125,0.125],[0.25,0.25,0.0]],
  [[-0.5,0.0,0.0],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.25,-0.25,0.0],[-0.25,0.25,0.0]],
  [[0.0,0.5,0.0],[-0.25,0.25,0.25],[0.125,-0.125,-0.125]],
  [[0.0,0.5,0.0],[0.125,-0.125,0.125],[-0.25,0.25,-0.25]],
  [[0.0,0.5,0.0],[0.0,-0.5,0.0]],
  [[0.25,-0.25,0.0],[-0.25,0.25,0.0],[0.125,-0.125,0.125]],
  [[-0.375,-0.375,-0.375],[-0.25,0.0,0.25],[-0.125,-0.125,-0.125],[-0.25,0.25,0.0]],
  [[0.125,0.125,0.125],[0.0,-0.5,0.0],[-0.25,-0.25,-0.25],[-0.125,-0.125,-0.125]],
  [[0.0,-0.5,0.0],[-0.25,-0.25,-0.25],[-0.125,-0.125,-0.125]],
  [[-0.125,0.125,0.125],[0.25,-0.25,0.0],[-0.25,0.25,0.0]],
  [[0.0,0.5,0.0],[0.25,0.25,-0.25],[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.375,0.375,-0.375],[-0.25,-0.25,0.0],[-0.125,0.125,-0.125],[-0.25,0.0,0.25]],
  [[0.0,0.5,0.0],[0.25,0.25,-0.25],[-0.125,-0.125,0.125]],
  [[0.25,-0.25,0.0],[-0.25,0.25,0.0],[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[-0.25,-0.25,0.0],[-0.25,-0.25,0.0],[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[-0.25,-0.25,0.0],[-0.25,-0.25,0.0]],
  [[-0.25,-0.25,0.0],[-0.25,-0.25,0.0]],
  [[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.25,-0.25,0.0],[0.25,0.25,-0.0]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25]],
  [[0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.375,-0.375,0.375],[0.0,-0.25,-0.25],[-0.125,0.125,-0.125],[0.25,0.25,0.0]],
  [[-0.125,-0.125,0.125],[-0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125],[-0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.25,0.0,0.25],[-0.25,0.0,0.25]],
  [[0.5,0.0,0.0],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.0,0.5,0.0],[-0.25,0.25,-0.25],[0.125,-0.125,0.125]],
  [[-0.25,0.25,-0.25],[-0.25,0.25,-0.25],[-0.125,0.125,-0.125],[-0.125,0.125,-0.125]],
  [[-0.25,0.0,-0.25],[0.375,-0.375,-0.375],[0.0,0.25,-0.25],[-0.125,0.125,0.125]],
  [[0.5,0.0,0.0],[-0.25,0.25,-0.25],[0.125,-0.125,0.125]],
  [[-0.25,0.0,0.25],[0.25,0.0,-0.25]],
  [[-0.0,0.0,0.5],[-0.25,0.25,0.25],[-0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.25,0.0,0.25],[0.25,0.0,-0.25]],
  [[-0.25,-0.0,-0.25],[-0.375,0.375,0.375],[-0.25,-0.25,0.0],[-0.125,0.125,0.125]],
  [[0.0,0.0,-0.5],[0.25,0.25,-0.25],[-0.125,-0.125,0.125]],
  [[-0.0,0.0,0.5],[0.0,0.0,0.5]],
  [[0.125,0.125,0.125],[0.125,0.125,0.125],[0.25,0.25,0.25],[0.0,0.0,0.5]],
  [[0.125,0.125,0.125],[0.25,0.25,0.25],[0.0,0.0,0.5]],
  [[-0.25,0.0,0.25],[0.25,0.0,-0.25],[-0.125,0.125,0.125]],
  [[-0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.25,0.0,0.25],[-0.25,0.0,0.25],[-0.25,0.0,0.25],[0.25,0.0,-0.25]],
  [[0.125,-0.125,0.125],[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[0.25,0.0,0.25],[-0.375,-0.375,0.375],[-0.25,0.25,0.0],[-0.125,-0.125,0.125]],
  [[-0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[-0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125],[0.0,-0.25,0.25],[0.0,0.25,-0.25]],
  [[0.0,-0.5,0.0],[0.125,0.125,-0.125],[0.25,0.25,-0.25],[-0.125,-0.125,0.125]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25],[0.125,-0.125,0.125]],
  [[0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25],[0.0,-0.25,0.25],[0.0,0.25,-0.25]],
  [[0.0,0.25,0.25],[0.0,0.25,0.25],[0.125,-0.125,-0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,0.125],[-0.125,-0.125,0.125],[0.125,0.125,0.125]],
  [[-0.0,0.0,0.5],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[-0.0,0.5,0.0],[-0.25,0.25,-0.25],[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.5,0.0,-0.0],[0.25,-0.25,-0.25],[0.125,-0.125,-0.125]],
  [[-0.25,0.25,0.25],[-0.125,0.125,0.125],[-0.25,0.25,0.25],[0.125,-0.125,-0.125]],
  [[0.375,-0.375,0.375],[0.0,0.25,0.25],[-0.125,0.125,-0.125],[-0.25,0.0,0.25]],
  [[0.0,-0.5,0.0],[-0.25,0.25,0.25],[-0.125,0.125,0.125]],
  [[-0.375,-0.375,0.375],[0.25,-0.25,0.0],[0.0,0.25,0.25],[-0.125,-0.125,0.125]],
  [[-0.125,0.125,0.125],[-0.25,0.25,0.25],[0.0,0.0,0.5]],
  [[0.125,0.125,0.125],[0.0,0.25,0.25],[0.0,0.25,0.25]],
  [[0.0,0.25,0.25],[0.0,0.25,0.25]],
  [[0.5,0.0,-0.0],[0.25,0.25,0.25],[0.125,0.125,0.125],[0.125,0.125,0.125]],
  [[0.125,-0.125,0.125],[-0.125,-0.125,0.125],[0.125,0.125,0.125]],
  [[-0.25,-0.0,-0.25],[0.25,0.0,0.25],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.25,-0.25,0.0],[0.25,0.25,-0.0],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125]],
  [[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.25,-0.25,0.0],[0.25,0.25,-0.0],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.25,-0.0,-0.25],[0.25,0.0,0.25],[0.125,0.125,0.125]],
  [[0.125,-0.125,0.125],[-0.125,-0.125,0.125],[0.125,0.125,0.125]],
  [[0.5,0.0,-0.0],[0.25,0.25,0.25],[0.125,0.125,0.125],[0.125,0.125,0.125]],
  [[0.0,0.25,0.25],[0.0,0.25,0.25]],
  [[0.125,0.125,0.125],[0.0,0.25,0.25],[0.0,0.25,0.25]],
  [[-0.125,0.125,0.125],[-0.25,0.25,0.25],[0.0,0.0,0.5]],
  [[-0.375,-0.375,0.375],[0.25,-0.25,0.0],[0.0,0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.0,-0.5,0.0],[-0.25,0.25,0.25],[-0.125,0.125,0.125]],
  [[0.375,-0.375,0.375],[0.0,0.25,0.25],[-0.125,0.125,-0.125],[-0.25,0.0,0.25]],
  [[-0.25,0.25,0.25],[-0.125,0.125,0.125],[-0.25,0.25,0.25],[0.125,-0.125,-0.125]],
  [[0.5,0.0,-0.0],[0.25,-0.25,-0.25],[0.125,-0.125,-0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25],[0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[-0.0,0.5,0.0],[-0.25,0.25,-0.25],[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[-0.0,0.0,0.5],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,0.125],[-0.125,-0.125,0.125],[0.125,0.125,0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[0.0,0.25,0.25],[0.0,0.25,0.25],[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25],[0.0,0.25,0.25],[0.0,0.25,0.25]],
  [[0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25],[0.125,-0.125,0.125]],
  [[0.0,-0.5,0.0],[0.125,0.125,-0.125],[0.25,0.25,-0.25],[-0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125],[0.0,-0.25,0.25],[0.0,0.25,-0.25]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[0.125,0.125,0.125],[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[-0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125]],
  [[0.25,0.0,0.25],[-0.375,-0.375,0.375],[-0.25,0.25,0.0],[-0.125,-0.125,0.125]],
  [[0.125,-0.125,0.125],[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[-0.25,-0.0,-0.25],[0.25,0.0,0.25],[0.25,0.0,0.25],[0.25,0.0,0.25]],
  [[-0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.25,0.0,0.25],[0.25,0.0,-0.25],[-0.125,0.125,0.125]],
  [[0.125,0.125,0.125],[0.25,0.25,0.25],[0.0,0.0,0.5]],
  [[0.125,0.125,0.125],[0.125,0.125,0.125],[0.25,0.25,0.25],[0.0,0.0,0.5]],
  [[-0.0,0.0,0.5],[0.0,0.0,0.5]],
  [[0.0,0.0,-0.5],[0.25,0.25,-0.25],[-0.125,-0.125,0.125]],
  [[-0.25,-0.0,-0.25],[-0.375,0.375,0.375],[-0.25,-0.25,0.0],[-0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.25,0.0,0.25],[0.25,0.0,-0.25]],
  [[-0.0,0.0,0.5],[-0.25,0.25,0.25],[-0.125,0.125,0.125]],
  [[-0.25,0.0,0.25],[0.25,0.0,-0.25]],
  [[0.5,0.0,0.0],[-0.25,0.25,-0.25],[0.125,-0.125,0.125]],
  [[-0.25,0.0,-0.25],[0.375,-0.375,-0.375],[0.0,0.25,-0.25],[-0.125,0.125,0.125]],
  [[-0.25,0.25,-0.25],[-0.25,0.25,-0.25],[-0.125,0.125,-0.125],[-0.125,0.125,-0.125]],
  [[-0.0,0.5,0.0],[-0.25,0.25,-0.25],[0.125,-0.125,0.125]],
  [[0.5,0.0,0.0],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.25,0.0,0.25],[-0.25,0.0,0.25]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125],[-0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125],[-0.125,0.125,0.125]],
  [[0.375,-0.375,0.375],[0.0,-0.25,-0.25],[-0.125,0.125,-0.125],[0.25,0.25,0.0]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.0,0.0,0.5],[0.25,-0.25,0.25],[0.125,-0.125,0.125]],
  [[0.0,-0.25,0.25],[0.0,-0.25,0.25]],
  [[-0.125,-0.125,0.125],[-0.25,-0.25,0.0],[0.25,0.25,-0.0]],
  [[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,-0.125,0.125]],
  [[-0.25,-0.25,0.0],[-0.25,-0.25,0.0]],
  [[0.125,0.125,0.125],[-0.25,-0.25,0.0],[-0.25,-0.25,0.0]],
  [[-0.25,-0.25,0.0],[-0.25,-0.25,0.0],[-0.125,-0.125,0.125]],
  [[-0.25,-0.25,0.0],[-0.25,-0.25,0.0],[-0.25,-0.25,0.0],[0.25,0.25,-0.0]],
  [[0.0,0.5,0.0],[0.25,0.25,-0.25],[-0.125,-0.125,0.125]],
  [[-0.375,0.375,-0.375],[-0.25,-0.25,0.0],[-0.125,0.125,-0.125],[-0.25,0.0,0.25]],
  [[0.0,0.5,0.0],[0.25,0.25,-0.25],[-0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.125,0.125,0.125],[0.25,-0.25,0.0],[-0.25,0.25,0.0]],
  [[0.0,-0.5,0.0],[-0.25,-0.25,-0.25],[-0.125,-0.125,-0.125]],
  [[0.125,0.125,0.125],[0.0,-0.5,0.0],[-0.25,-0.25,-0.25],[-0.125,-0.125,-0.125]],
  [[-0.375,-0.375,-0.375],[-0.25,0.0,0.25],[-0.125,-0.125,-0.125],[-0.25,0.25,0.0]],
  [[0.25,-0.25,0.0],[-0.25,0.25,0.0],[0.125,-0.125,0.125]],
  [[0.0,0.5,0.0],[0.0,-0.5,0.0]],
  [[0.0,0.5,0.0],[0.125,-0.125,0.125],[-0.25,0.25,-0.25]],
  [[0.0,0.5,0.0],[-0.25,0.25,0.25],[0.125,-0.125,-0.125]],
  [[0.25,-0.25,0.0],[-0.25,0.25,0.0]],
  [[-0.5,0.0,0.0],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.0,0.25,-0.25],[0.375,-0.375,-0.375],[-0.125,0.125,0.125],[0.25,0.25,0.0]],
  [[0.5,0.0,0.0],[0.25,-0.25,0.25],[-0.125,0.125,-0.125],[0.125,-0.125,0.125]],
  [[0.125,-0.125,0.125],[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[0.25,0.25,-0.25],[0.25,0.25,-0.25],[0.125,0.125,-0.125],[-0.125,-0.125,0.125]],
  [[-0.0,0.0,0.5],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125],[-0.125,0.125,0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,0.125]],
  [[-0.375,-0.375,0.375],[-0.0,0.25,0.25],[0.125,0.125,-0.125],[-0.25,-0.0,-0.25]],
  [[0.0,-0.25,0.25],[0.0,0.25,-0.25],[0.125,-0.125,0.125]],
  [[0.125,-0.125,0.125],[-0.25,-0.0,-0.25],[0.25,0.0,0.25]],
  [[0.125,-0.125,0.125],[0.125,-0.125,0.125]],
  [[0.0,-0.5,0.0],[0.125,0.125,-0.125],[0.25,0.25,-0.25]],
  [[0.0,-0.25,0.25],[0.0,0.25,-0.25]],
  [[0.125,0.125,0.125],[0.125,-0.125,0.125]],
  [[0.125,-0.125,0.125]],
  [[-0.5,0.0,0.0],[-0.125,-0.125,-0.125],[-0.25,-0.25,-0.25]],
  [[-0.5,0.0,0.0],[-0.125,-0.125,-0.125],[-0.25,-0.25,-0.25],[0.125,0.125,0.125]],
  [[0.375,0.375,0.375],[0.0,0.25,-0.25],[-0.125,-0.125,-0.125],[-0.25,0.25,0.0]],
  [[0.125,-0.125,-0.125],[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[0.125,0.125,0.125],[0.375,0.375,0.375],[0.0,-0.25,0.25],[-0.25,0.0,0.25]],
  [[-0.25,0.0,0.25],[-0.25,0.0,0.25],[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25],[-0.125,0.125,0.125]],
  [[-0.125,0.125,0.125],[0.125,-0.125,-0.125]],
  [[-0.125,-0.125,-0.125],[-0.25,-0.25,-0.25],[0.25,0.25,0.25],[0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125],[0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,0.0,-0.5],[0.25,0.25,0.25],[-0.125,-0.125,-0.125]],
  [[0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,-0.5,0.0],[0.25,0.25,0.25],[0.125,0.125,0.125]],
  [[-0.125,-0.125,0.125],[0.125,-0.125,-0.125]],
  [[0.0,-0.25,-0.25],[0.0,0.25,0.25]],
  [[0.125,-0.125,-0.125]],
  [[0.5,0.0,0.0],[0.5,0.0,0.0]],
  [[-0.5,0.0,0.0],[-0.25,0.25,0.25],[-0.125,0.125,0.125]],
  [[0.5,0.0,0.0],[0.25,-0.25,0.25],[-0.125,0.125,-0.125]],
  [[0.25,-0.25,0.0],[0.25,-0.25,0.0]],
  [[0.5,0.0,0.0],[-0.25,-0.25,0.25],[-0.125,-0.125,0.125]],
  [[-0.25,0.0,0.25],[-0.25,0.0,0.25]],
  [[0.125,0.125,0.125],[-0.125,0.125,0.125]],
  [[-0.125,0.125,0.125]],
  [[0.5,0.0,-0.0],[0.25,0.25,0.25],[0.125,0.125,0.125]],
  [[0.125,-0.125,0.125],[-0.125,-0.125,0.125]],
  [[-0.25,-0.0,-0.25],[0.25,0.0,0.25]],
  [[0.125,-0.125,0.125]],
  [[-0.25,-0.25,0.0],[0.25,0.25,-0.0]],
  [[-0.125,-0.125,0.125]],
  [[0.125,0.125,0.125]],
  [[0,0,0]]]


def compute_surface_distances(mask_gt, mask_pred, spacing_mm):
  """Compute closest distances from all surface points to the other surface.

  Finds all surface elements "surfels" in the ground truth mask `mask_gt` and
  the predicted mask `mask_pred`, computes their area in mm^2 and the distance
  to the closest point on the other surface. It returns two sorted lists of
  distances together with the corresponding surfel areas. If one of the masks
  is empty, the corresponding lists are empty and all distances in the other
  list are `inf` 
  
  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.
    spacing_mm: 3-element list-like structure. Voxel spacing in x0, x1 and x2
        direction 

  Returns:
    A dict with 
    "distances_gt_to_pred": 1-dim numpy array of type float. The distances in mm
        from all ground truth surface elements to the predicted surface, 
        sorted from smallest to largest
    "distances_pred_to_gt": 1-dim numpy array of type float. The distances in mm
        from all predicted surface elements to the ground truth surface, 
        sorted from smallest to largest 
    "surfel_areas_gt": 1-dim numpy array of type float. The area in mm^2 of 
        the ground truth surface elements in the same order as 
        distances_gt_to_pred
    "surfel_areas_pred": 1-dim numpy array of type float. The area in mm^2 of 
        the predicted surface elements in the same order as 
        distances_pred_to_gt
       
  """
  
  # compute the area for all 256 possible surface elements 
  # (given a 2x2x2 neighbourhood) according to the spacing_mm
  neighbour_code_to_surface_area = np.zeros([256])
  for code in range(256):
    normals = np.array(neighbour_code_to_normals[code])
    sum_area = 0
    for normal_idx in range(normals.shape[0]):
      # normal vector
      n = np.zeros([3])
      n[0] = normals[normal_idx,0] * spacing_mm[1] * spacing_mm[2]
      n[1] = normals[normal_idx,1] * spacing_mm[0] * spacing_mm[2]
      n[2] = normals[normal_idx,2] * spacing_mm[0] * spacing_mm[1]
      area = np.linalg.norm(n)
      sum_area += area
    neighbour_code_to_surface_area[code] = sum_area

  # compute the bounding box of the masks to trim
  # the volume to the smallest possible processing subvolume
  mask_all = mask_gt | mask_pred
  bbox_min = np.zeros(3, np.int64)
  bbox_max = np.zeros(3, np.int64)

  # max projection to the x0-axis
  proj_0 = np.max(np.max(mask_all, axis=2), axis=1)
  idx_nonzero_0 = np.nonzero(proj_0)[0]
  if len(idx_nonzero_0) == 0:
    return {"distances_gt_to_pred":  np.array([]), 
            "distances_pred_to_gt":  np.array([]), 
            "surfel_areas_gt":       np.array([]), 
            "surfel_areas_pred":     np.array([])}
    
  bbox_min[0] = np.min(idx_nonzero_0)
  bbox_max[0] = np.max(idx_nonzero_0)

  # max projection to the x1-axis
  proj_1 = np.max(np.max(mask_all, axis=2), axis=0)
  idx_nonzero_1 = np.nonzero(proj_1)[0]
  bbox_min[1] = np.min(idx_nonzero_1)
  bbox_max[1] = np.max(idx_nonzero_1)

  # max projection to the x2-axis
  proj_2 = np.max(np.max(mask_all, axis=1), axis=0)
  idx_nonzero_2 = np.nonzero(proj_2)[0]
  bbox_min[2] = np.min(idx_nonzero_2)
  bbox_max[2] = np.max(idx_nonzero_2)

  # print("bounding box min = {}".format(bbox_min))
  # print("bounding box max = {}".format(bbox_max))

  # crop the processing subvolume.
  # we need to zeropad the cropped region with 1 voxel at the lower, 
  # the right and the back side. This is required to obtain the "full" 
  # convolution result with the 2x2x2 kernel
  cropmask_gt = np.zeros((bbox_max - bbox_min)+2, np.uint8)
  cropmask_pred = np.zeros((bbox_max - bbox_min)+2, np.uint8)

  cropmask_gt[0:-1, 0:-1, 0:-1] = mask_gt[bbox_min[0]:bbox_max[0]+1,
                                          bbox_min[1]:bbox_max[1]+1,
                                          bbox_min[2]:bbox_max[2]+1]

  cropmask_pred[0:-1, 0:-1, 0:-1] = mask_pred[bbox_min[0]:bbox_max[0]+1,
                                              bbox_min[1]:bbox_max[1]+1,
                                              bbox_min[2]:bbox_max[2]+1]

  # compute the neighbour code (local binary pattern) for each voxel
  # the resultsing arrays are spacially shifted by minus half a voxel in each axis.
  # i.e. the points are located at the corners of the original voxels
  kernel = np.array([[[128,64],
                      [32,16]],
                     [[8,4],
                      [2,1]]])
  neighbour_code_map_gt = scipy.ndimage.filters.correlate(cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0) 
  neighbour_code_map_pred = scipy.ndimage.filters.correlate(cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0) 

  # create masks with the surface voxels
  borders_gt   = ((neighbour_code_map_gt != 0) & (neighbour_code_map_gt != 255))
  borders_pred = ((neighbour_code_map_pred != 0) & (neighbour_code_map_pred != 255))

  # compute the distance transform (closest distance of each voxel to the surface voxels)
  if borders_gt.any():
    distmap_gt = scipy.ndimage.morphology.distance_transform_edt(~borders_gt, sampling=spacing_mm)
  else:
    distmap_gt = np.Inf * np.ones(borders_gt.shape)

  if borders_pred.any():  
    distmap_pred = scipy.ndimage.morphology.distance_transform_edt(~borders_pred, sampling=spacing_mm)
  else:
    distmap_pred = np.Inf * np.ones(borders_pred.shape)

  # compute the area of each surface element
  surface_area_map_gt = neighbour_code_to_surface_area[neighbour_code_map_gt]
  surface_area_map_pred = neighbour_code_to_surface_area[neighbour_code_map_pred]

  # create a list of all surface elements with distance and area
  distances_gt_to_pred = distmap_pred[borders_gt]
  distances_pred_to_gt = distmap_gt[borders_pred]
  surfel_areas_gt   = surface_area_map_gt[borders_gt]
  surfel_areas_pred = surface_area_map_pred[borders_pred]

  # sort them by distance
  if distances_gt_to_pred.shape != (0,):
    sorted_surfels_gt = np.array(sorted(zip(distances_gt_to_pred, surfel_areas_gt)))
    distances_gt_to_pred = sorted_surfels_gt[:,0]
    surfel_areas_gt      = sorted_surfels_gt[:,1]

  if distances_pred_to_gt.shape != (0,):
    sorted_surfels_pred = np.array(sorted(zip(distances_pred_to_gt, surfel_areas_pred)))
    distances_pred_to_gt = sorted_surfels_pred[:,0]
    surfel_areas_pred    = sorted_surfels_pred[:,1]

  # return distances_pred_to_gt
  return {"distances_gt_to_pred":  distances_gt_to_pred, 
          "distances_pred_to_gt":  distances_pred_to_gt, 
          "surfel_areas_gt":       surfel_areas_gt, 
          "surfel_areas_pred":     surfel_areas_pred}

def compute_average_surface_distance(surface_distances):
  distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
  distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
  surfel_areas_gt      = surface_distances["surfel_areas_gt"]
  surfel_areas_pred    = surface_distances["surfel_areas_pred"]
  average_distance_gt_to_pred = np.sum( distances_gt_to_pred * surfel_areas_gt) / np.sum(surfel_areas_gt)
  average_distance_pred_to_gt = np.sum( distances_pred_to_gt * surfel_areas_pred) / np.sum(surfel_areas_pred)
  return (average_distance_gt_to_pred, average_distance_pred_to_gt)

def compute_robust_hausdorff(surface_distances, percent):
  distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
  distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
  surfel_areas_gt      = surface_distances["surfel_areas_gt"]
  surfel_areas_pred    = surface_distances["surfel_areas_pred"]
  if len(distances_gt_to_pred) > 0:
    surfel_areas_cum_gt   = np.cumsum(surfel_areas_gt) / np.sum(surfel_areas_gt)
    idx = np.searchsorted(surfel_areas_cum_gt, percent/100.0)
    perc_distance_gt_to_pred = distances_gt_to_pred[min(idx, len(distances_gt_to_pred)-1)]
  else:
    perc_distance_gt_to_pred = np.Inf
    
  if len(distances_pred_to_gt) > 0:
    surfel_areas_cum_pred = np.cumsum(surfel_areas_pred) / np.sum(surfel_areas_pred)
    idx = np.searchsorted(surfel_areas_cum_pred, percent/100.0)
    perc_distance_pred_to_gt = distances_pred_to_gt[min(idx, len(distances_pred_to_gt)-1)]
  else:
    perc_distance_pred_to_gt = np.Inf
    
  return max( perc_distance_gt_to_pred, perc_distance_pred_to_gt)

def compute_surface_overlap_at_tolerance(surface_distances, tolerance_mm):
  distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
  distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
  surfel_areas_gt      = surface_distances["surfel_areas_gt"]
  surfel_areas_pred    = surface_distances["surfel_areas_pred"]
  rel_overlap_gt   = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm]) / np.sum(surfel_areas_gt)
  rel_overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm]) / np.sum(surfel_areas_pred)
  return (rel_overlap_gt, rel_overlap_pred)

def compute_surface_dice_at_tolerance(surface_distances, tolerance_mm):
  distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
  distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
  surfel_areas_gt      = surface_distances["surfel_areas_gt"]
  surfel_areas_pred    = surface_distances["surfel_areas_pred"]
  overlap_gt   = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm])
  overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm])
  surface_dice = (overlap_gt + overlap_pred) / (
      np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred))
  return surface_dice

def compute_dice_coefficient(mask_gt, mask_pred):
  """Compute soerensen-dice coefficient.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`. 
  
  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
  """
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum

def compute_evaluation_metrics_worker_fn(data: dict, num_classes, quite=True):

    pred = data['pred']
    gt = data['gt']
    pd = data ['pd']
    # TODO
    # pred = np.flip(pred, [1,2])
    dsc_organs = []
    nsd_organs = []
    for j in range(num_classes):
      pred_ = (pred == j).astype(np.uint8)
      gt_ = (gt == j).astype(np.uint8)
      # TODO 
    #   pred_ = np.flip(pred_, [1,2])
      # slice_visualize_XY(pred_.transpose(1,2,0), gt_.transpose(1,2,0))
      dice_similarity_coefficient = compute_dice_coefficient(gt_, pred_)
      surface_distance = compute_surface_distances(gt_, pred_, pd)
      surface_dice = compute_surface_dice_at_tolerance(surface_distance, 1)
      dsc_organs.append(dice_similarity_coefficient)
      nsd_organs.append(surface_dice)

      # nsd_organs.append(0.0)
      # print(dice_similarity_coefficient)
    if not quite:
      print("one image done")
    return (dsc_organs, nsd_organs)    

def compute_evaluation_metrics(data, num_workers=8, num_classes=3, quite=True):

#   from debug_utils import slice_visualize_XY
#   organ_list = ["background", "liver", "lesion"]
  # here data must be like return of load_paired_pred_and_gt()

  res_dsc = np.zeros([num_classes, len(data)])
  res_nsd = np.zeros([num_classes, len(data)])
  worker_fn = partial(compute_evaluation_metrics_worker_fn, num_classes=num_classes, quite=quite)
  if num_workers > 0:
    with closing(multiprocessing.Pool(processes=num_workers)) as pool:
      res = pool.map(worker_fn, data)
  else:
    res = []
    for arg in data:
      res.append(worker_fn(arg))
  # print(res)
  for i in range(len(res)):
    res_dsc[:, i] = res[i][0]
    res_nsd[:, i] = res[i][1]

  #  put nan to 0
  res_dsc[np.isnan(res_dsc)] = 0
  res_nsd[np.isnan(res_nsd)] = 0

  print('')
  print('Evluation result:')
  print('\tOrgan\t\tdsc\t\tnsd')
  for i in range(1, num_classes):
    dsc = np.round(np.mean(res_dsc[i]), 2)
    nsd = np.round(np.mean(res_nsd[i]), 2)
    print('\tOrgan no.{}\t\t{}\t\t{}'.format(i, dsc, nsd))
  print('')
  print('Mean dsc: ', np.nanmean(res_dsc))
  print('Mean nsd:' , np.nanmean(res_nsd))
  print('Evaluation score: ', (np.nanmean(res_dsc) + np.nanmean(res_nsd))/2)
  
def load_paired_pred_and_gt(pred_list, gt_list, h5_key='seg'):
    # the image name should be paired, ended with .h5(key=seg) or .nii.gz
    pred_list = [Path(i) for i in pred_list]
    gt_list = [Path(i) for i in gt_list]

    pred_suffix = Path(pred_list[0]).suffix
    gt_suffix = Path(gt_list[0]).suffix
    
    ret = []
    for p, g in zip(sorted(pred_list), sorted(gt_list)):
        assert p.name.split('.')[0] == g.name.split('.')[0], '{} and {} dont match'.format(p.name, g.name)
        pd = [1., 1., 1.]
        if pred_suffix == '.gz':
            _pred, _, pd, _, _  = sitk_load_with_metadata(str(p))
        elif pred_suffix == '.h5':
            _pred = h5py_load(str(p), h5_key)
        
        if gt_suffix == '.gz':
            _gt, _, pd, _, _   = sitk_load_with_metadata(str(g))
        elif gt_suffix == '.h5':
            _gt = h5py_load(str(g), h5_key)

        ret.append({'pred': _pred, 'gt': _gt, 'pd': pd})

    return ret

def debug_paired_pred_and_gt(data: dict):
    from mostoolkit.vis_utils import slice_visualize_XY
    slice_visualize_XY(data['pred'], data['gt'])

if __name__ == '__main__':
    # single pixels, 2mm away
    # mask_gt   = np.zeros((128,128,128), np.uint8)
    # mask_pred = np.zeros((128,128,128), np.uint8)
    # mask_gt[50,60,70] = 1
    # mask_pred[50,60,72] = 1
    # surface_distances = compute_surface_distances(mask_gt, mask_pred, spacing_mm=(3,2,1))
    # print("surface dice at 1mm:      {}".format(compute_surface_dice_at_tolerance(surface_distances, 1)))
    # print("volumetric dice:          {}".format(compute_dice_coefficient(mask_gt, mask_pred)))

    gt = Path('D:\\Data\\AMOS22\\AMOS22\\pro_test')
    gt = [str(i) for i in list(gt.glob('*.h5'))]
    # pred = list(Path(r'D:\Chang\Conferences\BVM-2023\amos2022_base\ensemble').glob('*.nii.gz'))
    pred = list(Path(r'D:\Chang\Conferences\BVM-2023\amos2022_atl\ensemble').glob('*.nii.gz'))
    pred = [str(i) for i in pred]
    eval_data = load_paired_pred_and_gt(pred, gt)
    # debug_paired_pred_and_gt(eval_data[0])

    compute_evaluation_metrics(eval_data, num_workers=0, num_classes=16)
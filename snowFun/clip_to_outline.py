# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 11:18:16 2022

@author: lzell
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.mask import mask
from rasterio.windows import from_bounds
from rasterio.windows import Window
from scipy.ndimage import uniform_filter


def clip_to_outline():
    return 0
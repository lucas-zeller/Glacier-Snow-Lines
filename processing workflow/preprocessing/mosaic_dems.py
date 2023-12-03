# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 18:24:38 2023

@author: lzell
"""

import os
import rasterio as rio
import numpy as np
import shapely
import pyproj
import geopandas as gpd
from rasterio.merge import merge as riomerge
from rasterio.plot import show as rioshow
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject, Resampling
from osgeo import gdal

# set folder paths, etc...
folder_AGVA = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA")
path_regions = os.path.join(folder_AGVA, "RGI", "S2_subregions", "subregions.shp")

# open subregions shapefile
regions_gdf = gpd.read_file(path_regions)
# regions_gdf = regions_gdf[regions_gdf['id']==4]

# define folder path that holds dem segments that need ot be merged
dem_folder_path = os.path.join(folder_AGVA, 'DEMs', '10m_COP_GLO30') 

#%%

### define the region names that you want merged
all_regions = ["region_151_10m"]

for r in all_regions:    
    
    # list out the image names in the folder
    all_image_names = os.listdir(dem_folder_path)
    
    # get only the ones that are this region
    all_image_names = [i for i in all_image_names if i.startswith(r)]
    
    # create image paths from names
    all_paths = [os.path.join(dem_folder_path, i) for i in all_image_names]
    
    # now use gdal.translate to merge all of them
    out_path = os.path.join(os.path.join(dem_folder_path, f"{r}.tif"))
    
    vrt = gdal.BuildVRT("merged.vrt", all_paths)
    translated = gdal.Translate(out_path, vrt, outputType=gdal.GDT_Float32, creationOptions = ['PREDICTOR=2','COMPRESS=LZW'])
    vrt = None
    translated = None
    
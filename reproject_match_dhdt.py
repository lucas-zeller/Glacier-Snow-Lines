# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 20:45:30 2023

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


# function to reproject and align an input image with another 'match' image
def reproj_match(infile, match, outfile):
    """Reproject a file to match the shape and projection of existing raster. 
    
    Parameters
    ----------
    infile : (string) path to input file to reproject
    match : (string) path to raster with desired shape and projection 
    outfile : (string) path to output file tif
    """
    # open input
    with rio.open(infile) as src:
        #src_transform = src.transform
        
        # open input to match
        with rio.open(match) as match:
            dst_crs = match.crs
            
            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,     # input CRS
                dst_crs,     # output CRS
                match.width,   # input width
                match.height,  # input height 
                *match.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
            )

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": 0})
        
        # open output
        with rio.open(outfile, "w", **dst_kwargs) as dst:
            # iterate through bands and write using reproject function
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.cubic)
    return rio.open(outfile).read(1)


# set folder paths, etc...
folder_AGVA = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA")
folder_dhdt = os.path.join(folder_AGVA, 'DEMs', 'Hugonnet_thinning', 'Alaska Albers')
folder_dem = os.path.join(folder_AGVA, "DEMs", "10m_COP_GLO30")
path_regions = os.path.join(folder_AGVA, "RGI", "S2_subregions", "subregions.shp")

# choose which year range you want to do, which subregion
years_to_do = "01_02_rgi60_2000-01-01_2005-01-01"
subregion = 10

# set path to the file of the original dem
path_dem_original = os.path.join(folder_dem, f"Region_{subregion}.tif")

# set path to the file of the dhdt mosaic covering this region
path_dhdt_original = os.path.join(folder_dhdt, years_to_do, 'dhdt', f"Region_{subregion}.tif")

# set destination file path
path_output = os.path.join(folder_AGVA, 'DEMs', "10m_thinning", years_to_do, "dhdt", f"Region_{subregion}.tif")

# send these off to the above function
test = reproj_match(path_dhdt_original, path_dem_original, path_output)


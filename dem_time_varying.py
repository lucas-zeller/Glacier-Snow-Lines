# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:27:24 2023

@author: lzell
"""

### overall goal: create a time-varying DEM for each glacier region.
### open the base COP10/COP30 DEM, and the 5-year dhdt products
### extrapolate the elevation across the region by adding/subtracting the dhdt
### save a single netcdf file for each region, with one layer for each year

import os
import rasterio as rio
import numpy as np
import shapely
import pyproj
import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray as riox
import rasterio as rio
import xarray as xr
from osgeo import gdal
import pandas as pd

# set folder paths, etc...
folder_AGVA = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA")

# open subregions shapefile
path_regions = os.path.join(folder_AGVA, "RGI", "S2_subregions", "subregions.shp")
regions_gdf = gpd.read_file(path_regions)

# select which regions you want to do
regions = [10]

#%%
# iterate through each region
for r in regions:
    print(r)
    
    # open the base DEM
    dem_base_path = os.path.join(folder_AGVA, 'DEMs', "10m_COP_GLO30", f"Region_{r:02d}.tif")
    dem_src = rio.open(dem_base_path)
    dem_base = rio.open(dem_base_path).read(1) # scaling factor is 0.1
    out_meta = dem_src.meta
    
    # dem_base = riox.open_rasterio(dem_base_path)
    # test = test.rename({"band":"time"})
    # test['time'] = pd.to_datetime(['2013-01-01'])
    
    #%%
    
    ### we are going to assume that the base dem represents elevation on 01-01-2013
    
    # define paths to the dhdt products
    dhdt_00_05_path = os.path.join(folder_AGVA, 'DEMs', "10m_thinning", "01_02_rgi60_2000-01-01_2005-01-01", "dhdt", f"Region_{r:02d}.tif")
    dhdt_05_10_path = os.path.join(folder_AGVA, 'DEMs', "10m_thinning", "01_02_rgi60_2005-01-01_2010-01-01", "dhdt", f"Region_{r:02d}.tif")
    dhdt_10_15_path = os.path.join(folder_AGVA, 'DEMs', "10m_thinning", "01_02_rgi60_2010-01-01_2015-01-01", "dhdt", f"Region_{r:02d}.tif")
    dhdt_15_20_path = os.path.join(folder_AGVA, 'DEMs', "10m_thinning", "01_02_rgi60_2015-01-01_2020-01-01", "dhdt", f"Region_{r:02d}.tif")
    
    # open each of the dhdt products
    dhdt_00_05 = rio.open(dhdt_00_05_path).read(1) # no scaling factor
    dhdt_05_10 = rio.open(dhdt_05_10_path).read(1) # no scaling factor
    dhdt_10_15 = rio.open(dhdt_10_15_path).read(1) # no scaling factor
    dhdt_15_20 = rio.open(dhdt_15_20_path).read(1) # no scaling factor
    
    # dhdt_00_05 = riox.open_rasterio(dhdt_00_05_path)
    # dhdt_05_10 = riox.open_rasterio(dhdt_05_10_path)
    # dhdt_10_15 = riox.open_rasterio(dhdt_10_15_path)
    # dhdt_15_20 = riox.open_rasterio(dhdt_15_20_path)
    
    #%%
    ### so the 2013 dem is the original, and we extrapolate out from there
    base = 13
    
    # lets initially just do 2013-2023
    # for i in [10,11,12,13,14,15,16,17,18,19,20,21,22,23]:
    # xr_list = []
    for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]:
    # for i in [13,17,23]:
        
        # calculate numbers years off from 2013
        dy = i-base
        
        print(f"Region {r}, year {i}")
        
        # from this, calculate how much to multiply each of the dem products
        # I can't explain in words how this work, but trust me that I thought through it and it is good
        f10 = min( 2, max(dy,-3)) 
        f15 = max( dy-f10, 0)
        f05 = max( min(dy-f10,0), -5)
        f00 = max( min(dy-f10-f05,0), -5)
        
        #print(2000+i, dy, f"{f00}:{f05}:{f10}:{f15}")
        dem_new = ((dem_base) + (dhdt_00_05*f00*10) + 
                                (dhdt_05_10*f05*10) + 
                                (dhdt_10_15*f10*10) + 
                                (dhdt_15_20*f15*10)  ).astype(int)
        dem_new[dem_base==0] = 0
        
        # dem_new = xr.where(dem_base.equals(0), 0, dem_new)
        # dem_new = dem_new.rename({"band":"time"})
        # dem_new['time'] = pd.to_datetime([f'20{i:02d}-01-01'])
        
        # xr_list.append(dem_new)
        
        # out_path = os.path.join(folder_AGVA, 'DEMs', 'time_varying_DEMs', "10m", f"Region_{r:02d}_test.tif")
        # dem_new.to_netcdf(out_path, encoding={'time':{'zlib': True, 'complevel':5}})
        
        # save to file AGVA\DEMs\time_varying_DEMs\10m
        out_path = os.path.join(folder_AGVA, 'DEMs', 'time_varying_DEMs', "10m", f"Region_{r:02d}_20{i:02d}.tif")
        
        # save dem
        with rio.open(out_path, "w", **out_meta, compress="ZSTD") as dest:
            dest.write(dem_new[None, :, :])
        
        # if i==13: dem_13=dem_new
        # if i==17: dem_17=dem_new
        # if i==23: dem_23=dem_new
        
        #%%
    # full_xr = xr.concat(xr_list, dim='time').astype(int)
    
    # # save to file AGVA\DEMs\time_varying_DEMs\10m
    # out_path = os.path.join(folder_AGVA, 'DEMs', 'time_varying_DEMs', "10m", f"Region_{r:02d}.tif")
    # full_xr.to_netcdf(out_path, encoding={'time':{'zlib': True, 'complevel':5}})
        
#%%
# quick image to see if it worked
# fig,axs = plt.subplots(1,3, sharex=True, sharey=True)
# axs[0].imshow((dem_13-dem_base)*0.1, vmin=-50, vmax=20)
# axs[1].imshow((dem_17-dem_base)*0.1, vmin=-50, vmax=20)
# axs[2].imshow((dem_23-dem_base)*0.1, vmin=-50, vmax=20)
    
















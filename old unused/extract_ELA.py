# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 08:53:17 2023

@author: lzell
"""

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
import snowFun
# from snowFun import extract_ELAs
import datetime
from matplotlib.pyplot import cm

# define folder and file paths
folder_AGVA = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA")
folder_dems = os.path.join(folder_AGVA, "DEMs", "time_varying_DEMs", "10m")
folder_class = os.path.join(folder_AGVA, 'classified images', 'S2_Classified_Merged')
folder_cloud = os.path.join(folder_AGVA, 'classified images', 'S2_Cloud_Merged')
folder_meta = os.path.join(folder_AGVA, "classified images", "meta csv", "S2")

#%%
# open rgi
path_rgi = os.path.join(folder_AGVA, 'RGI', "01_rgi60_Alaska", "01_rgi60_Alaska.shp")
rgi_gdf = gpd.read_file(path_rgi)

#%%
# subset rgi to single outline
rgi_single = rgi_gdf[rgi_gdf['Name']=='Wolverine Glacier'].to_crs("EPSG:3338")
single_geometry = rgi_single.geometry
# single_geometry = single_geometry.buffer(-100) #what if we buffer out the outside 100m

# open class and cloud files with rioxarray
file_name = "S2_043_2021-01-01_2021-12-31_R10_wolv"

# open files, clipping to single outline
xr_class = riox.open_rasterio(os.path.join(folder_class, f"{file_name}.tif")).rio.write_nodata(99)
xr_class = xr_class.rio.clip(single_geometry, from_disk=True, drop=True)#.rio.write_nodata(99)

xr_cloud = riox.open_rasterio(os.path.join(folder_cloud, f"{file_name}.tif")).rio.write_nodata(99)
xr_cloud = xr_cloud.rio.clip(single_geometry, from_disk=True, drop=True)

xr_class = xr_class.rename({"band":"time"})
xr_cloud = xr_cloud.rename({"band":"time"})

# metadata csv, convert date to datetimes
meta_fp = os.path.join(folder_meta, f"{file_name}.csv")
meta_df = pd.read_csv(meta_fp)

# format time axis for cloud and class
datetimes = pd.to_datetime([f"{str(i)[:4]}-{str(i)[4:6]}-{str(i)[6:]}" for i in meta_df['date']])
xr_cloud['time'] = datetimes
xr_class['time'] = datetimes

# create quick binary glacier mask of 0 and 1
glacier_mask = xr_cloud.median(dim='time')
glacier_mask = (glacier_mask<90)#.astype(int)

# calculate total number of pixels
glacier_pixels = glacier_mask.sum().values

first = xr_class.isel(time=-21) # snow=1, not-snow=0   #0snow_1firn_2ice_3rock
first = xr.where(glacier_mask, first, np.nan)

fig,axs=plt.subplots()
first.plot(ax=axs, cbar_kwargs={"label": "mask"})

# open dem of the glacier
path_dem = os.path.join(folder_dems, "Region_10_2021.tif")
xr_dem = riox.open_rasterio(path_dem).rio.write_nodata(0)
xr_dem = xr_dem.rio.clip(single_geometry, from_disk=True, drop=True)/10
xr_dem = xr.where(xr_dem==0, np.nan, xr_dem)

fig,axs=plt.subplots()
xr_dem.plot(ax=axs)

#%%
# create binary mask of useable and unuseable data
usable = xr_cloud.copy().astype(float)
bad_data = [2,3,8,9,10]
good_data = [1,4,5,6,7,11]

usable = xr.where(usable.isin([0,98,99]), np.nan, usable) # replace 0 with nan
usable = xr.where(usable.isin(good_data), 1, usable) # replace good numbers with 1
usable = xr.where(usable.isin(bad_data), 0, usable) # replace bad numbers with 0

# count usable pixels in each time
count_usable_by_time = usable.sum(dim=['x','y']) 
percent_usable_by_time = count_usable_by_time/glacier_pixels

# sum through time to look at frequency of good data
count_usable_by_pixel = usable.sum(dim='time')
percent_usable_by_pixel = count_usable_by_pixel/len(xr_cloud.time) # sum and divide by number of obs
# percent_usable = xr.where(percent_usable.isin([0]), np.nan, percent_usable) # replace 0 with nan

fig,axs=plt.subplots()
percent_usable_by_pixel.plot(ax=axs, cbar_kwargs={"label": "% usable data"})

#%%
### lets interpolate observations into the places where there is cloud cover by using most recent prior obs

# option to keep only somewhat useable imagery
good_times = (percent_usable_by_time>0.15)

xr_class_masked = xr.where(usable==1, xr_class, np.nan).sel(time=good_times)
xr_class_filled = xr_class_masked.ffill(dim='time')
snow = xr.where(xr_class_filled==0, 1, 0)

# now plot percentage of snow cover through time
count_snow_by_time = snow.sum(dim=['x','y']) # total snow obs in each time
percent_snow_by_time = count_snow_by_time/glacier_pixels

# plot it up
fig,axs = plt.subplots(figsize=(8,4))
axs.scatter(percent_snow_by_time.time, percent_snow_by_time,
               c=percent_usable_by_time[good_times], vmin=0, vmax=1)

axs.set_xlabel('Date')
axs.set_ylabel("Percent snow cover")

fig,axs=plt.subplots()
snow.sum('time').plot(ax=axs)
axs.axis('equal')

#%%
# lets grab the classification from a single day and try to get an ela from that
single_day = snow.sel(time="2021-09-11")

to_show = xr.where(glacier_mask==1, single_day, np.nan)
fig, axs=plt.subplots()
to_show.plot(ax=axs)

# lets get the minimum and maximum elevation on the glacier
z_min = np.nanmin(xr_dem)
z_max = np.nanmax(xr_dem)

### lets go through 10 m bands and find how much of each band is snow covered
step = 10 #define the step size between band centers (must be even)
width = 20 #define width of each band (in each direction)

# get the centers of each elevation band
z_bands = np.arange( np.ceil(z_min/step)*step-width, np.ceil(z_max/step)*step-width, step) 

snow_fractions = []
for z in z_bands:
    
    # subset the snow/ice class to this elevation
    band_subset = single_day.where( ( xr_dem>=(z-width) ) & ( xr_dem<(z+width) ) )
    
    # calculate mean. this will give % of this area that is snow
    band_mean = np.nanmean(band_subset)
    
    # append
    snow_fractions.append(band_mean)

# format to numpy array
snow_fractions = np.array(snow_fractions)

### now lets find the elevation(s) where the snow fraction crosses 0.5
sf_centered = snow_fractions-0.5 #center around 0.5
idx_crossing = np.where(np.diff(np.sign(sf_centered))==2)[0] #find indices where it goes from - to +

# interpolate to find elevation partway between z_crossing where the crossing occurs
def get_ELA(idx_c):
    z_c = z_bands[idx_c]
    slope = ( sf_centered[idx_c+1] - sf_centered[idx_c] ) / (step) # calculate slope between this point and the next
    crossing = z_c - sf_centered[idx_c]/slope #calculate where that line crosses 0
    return crossing

z_crossing = [get_ELA(i) for i in idx_crossing]

idx_zero = np.where(sf_centered==0)[0] #also get indices where it is exactly 0.5
z_zero = [z_bands[i] for i in idx_zero] #these elevations are exactly 0.5


# plot the changing amounts of snow with elevation
fig,axs=plt.subplots()
axs.scatter(z_bands, snow_fractions)

axs.plot([z_min, z_max], [0.5, 0.5], c='black', linestyle='dashed') #horizontal line at 0.5
axs.vlines(z_crossing, 0,1, colors="black") #vertical lines where we've identified crossing 0.5

axs.set_xlabel('Elevation (m)')
axs.set_ylabel('Snow Fraction')

### wrap all this up in a function, put it in snowFun
elas = snowFun.extract_ELAs(single_day, xr_dem)

#%%
### now lets see what happens when we extract the ela for every timestep
daily_ELAs = []
    
# Loop through each timestep and call extract_ELAs function
for time_idx in range(len(snow.time)):
    
    # Extract 2D spatial data for the current timestep
    spatial_data = snow.isel(time=time_idx)
    
    # Call the extract_ELAs function with the spatial_data
    elas = snowFun.extract_ELAs(spatial_data, xr_dem)
    
    daily_ELAs.append(elas)
    
    print(f"Timestep {time_idx}: Extracted Value = {elas}")

#%%
# plot the results
fig,axs=plt.subplots()
for i in range(len(datetimes[good_times])):
    elas = daily_ELAs[i]
    dt = datetimes[good_times][i]
    dts = [dt for e in elas]
    ela = np.nanmax(elas)
    
    if ela==0: continue
    
    axs.scatter(dt, ela, c='black')
 
    #%%
# lets try to plot ELA lines on the glacier outline
fig,axs=plt.subplots()

axs.imshow(glacier_mask, alpha=0) #to get the axes to fliip
axs.contour(glacier_mask, colors='black') # add glacier outline

# set colors that will be used
colors_to_use = iter(plt.cm.RdPu(np.linspace(0, 1, len(datetimes[good_times]))))

dem = xr_dem.values[0]
for i in range(len(datetimes[good_times])):
    
    dt = datetimes[good_times][i]
    ela = np.nanmax(daily_ELAs[i])
    c = next(colors_to_use)
    if ela==0: continue
    
    axs.contour(dem, levels=[ela], colors=[c])
    
    
    
    # axs.scatter(dt, ela, c='black')




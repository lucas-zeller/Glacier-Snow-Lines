# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:03:55 2023

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

#%%
# open dem of the glacier
path_dem = os.path.join(folder_dems, "Region_10_2021.tif")
xr_dem = riox.open_rasterio(path_dem).rio.write_nodata(0)
xr_dem = xr_dem.rio.clip(single_geometry, from_disk=True, drop=True)/10

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
# identify spots with snow vs not snow
snow = xr.where(xr_class==0, 1, 0)

fig,axs=plt.subplots(1,2, sharex=True, sharey=True, figsize=(10,4), subplot_kw=dict(box_aspect=1))

# plot the total number of snow observations at each pixel
summed_snow = snow.sum(dim='time') # sum
summed_snow = xr.where(glacier_mask==0, np.nan, summed_snow) # replace off-glacier with nan
summed_snow.plot(ax=axs[0], cbar_kwargs={"label": "count of snow"})

# plot the percentage of snow observations at each pixels
percent_snow = snow.sum(dim='time')/len(xr_cloud.time) # sum
percent_snow = xr.where(glacier_mask==0, np.nan, percent_snow) # replace off-glacier with nan
percent_snow.plot(ax=axs[1], cbar_kwargs={"label": "% of snow"})

plt.suptitle("Not using cloud masking")
plt.tight_layout()


#%%
# mask the classification data to only the useable stuff then do the same as above
xr_class_masked = xr.where(usable==1, xr_class, 99)
snow = xr.where(xr_class_masked==0, 1, 0)


fig,axs=plt.subplots(1,2, sharex=True, sharey=True, figsize=(10,4), subplot_kw=dict(box_aspect=1))

# plot the total number of snow observations at each pixel
summed_snow = snow.sum(dim='time') # sum
summed_snow = xr.where(glacier_mask==1, summed_snow, np.nan) # replace off-glacier with nan
summed_snow.plot(ax=axs[0], cbar_kwargs={"label": "count of snow"})
# xr.where(glacier_mask==1, count_usable, np.nan).plot(ax=axs[0], cbar_kwargs={"label": "count of snow"})

# plot the percentage of snow observations at each pixels
percent_snow = snow.sum(dim='time')/count_usable_by_pixel # sum
percent_snow = xr.where(glacier_mask==0, np.nan, percent_snow) # replace off-glacier with nan
percent_snow.plot(ax=axs[1], cbar_kwargs={"label": "% of snow"})

plt.suptitle("Using cloud masking")
plt.tight_layout()

#%%
# calculate total amount of snow cover in each image, and percentage of snow cover
xr_class_masked = xr.where(usable==1, xr_class, 99)
snow = xr.where(xr_class_masked==0, 1, 0)

count_snow_by_time = snow.sum(dim=['x','y']) # total snow obs in each time
percent_snow_by_time = count_snow_by_time/count_usable_by_time

# keep only obs that are at least __% usable
good_idx = percent_usable_by_time>-1

fig,axs = plt.subplots(figsize=(8,4))
axs.scatter(percent_snow_by_time.time, percent_snow_by_time,
               c=percent_usable_by_time, vmin=0, vmax=1)

axs.set_xlabel('Date')
axs.set_ylabel("Percent snow cover")

#%%
### lets interpolate observations into the places where there is cloud cover by using most recent prior obs
xr_class_masked = xr.where(usable==1, xr_class, np.nan)
xr_class_filled = xr_class_masked.ffill(dim='time')
snow = xr.where(xr_class_filled==0, 1, 0)

# now plot percentage of snow cover through time
count_snow_by_time = snow.sum(dim=['x','y']) # total snow obs in each time
percent_snow_by_time = count_snow_by_time/glacier_pixels

# plot it up
fig,axs = plt.subplots(figsize=(8,4))
axs.scatter(percent_snow_by_time.time, percent_snow_by_time,
               c=percent_usable_by_time, vmin=0, vmax=1)

axs.set_xlabel('Date')
axs.set_ylabel("Percent snow cover")

fig,axs=plt.subplots()
snow.sum('time').plot(ax=axs)
axs.axis('equal')

#%%
# lets do that forward interpolation again, but first remove all images with <10% useable data
good_times = (percent_usable_by_time>0.1)

xr_class_masked = xr.where(usable==1, xr_class, np.nan).sel(time=good_times)
xr_class_filled = xr_class_masked.ffill(dim='time')
snow = xr.where(xr_class_filled==0, 1, 0)

# now plot percentage of snow cover through time
count_snow_by_time = snow.sum(dim=['x','y']) # total snow obs in each time
percent_snow_by_time = count_snow_by_time/glacier_pixels


fig,axs = plt.subplots(figsize=(8,4))
axs.scatter(percent_snow_by_time.time, percent_snow_by_time,
               c=percent_usable_by_time[good_times], vmin=0, vmax=1)

axs.set_xlabel('Date')
axs.set_ylabel("Percent snow cover")

fig,axs=plt.subplots()
snow.sum('time').plot(ax=axs)
axs.axis('equal')

#%%
print(snowFun.test_function(5))



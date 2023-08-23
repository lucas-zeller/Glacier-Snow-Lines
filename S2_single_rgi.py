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
glacier_choice = 'Wolverine'
rgi_single = rgi_gdf[rgi_gdf['Name']==f'{glacier_choice} Glacier'].to_crs("EPSG:3338")
single_geometry = rgi_single.geometry
# single_geometry = single_geometry.buffer(-100) #what if we buffer out the outside 100m

#%%
# open class and cloud files with rioxarray
if glacier_choice == 'Gulkana':
    file_name = "S2_RGI60-01.00570_2015-01-01_2023-01-01_0" #gulkana
elif glacier_choice == 'Wolverine':
    file_name = "S2_RGI60-01.09162_2015-01-01_2023-01-01_0" #wolverine

# open files, clipping to single outline
xr_class = riox.open_rasterio(os.path.join(folder_class, f"{file_name}.tif")).rio.clip(single_geometry, from_disk=True, drop=True)
xr_cloud = riox.open_rasterio(os.path.join(folder_cloud, f"{file_name}.tif")).rio.clip(single_geometry, from_disk=True, drop=True)

# load metadata csv, convert date to datetimes
meta_fp = os.path.join(folder_meta, f"{file_name}.csv")
meta_df = pd.read_csv(meta_fp)

# format time axis for cloud and class
datetimes = pd.to_datetime([f"{str(i)[:4]}-{str(i)[4:6]}-{str(i)[6:]}" for i in meta_df['date']])
xr_class = xr_class.rename({"band":"time"})
xr_cloud = xr_cloud.rename({"band":"time"})
xr_cloud['time'] = datetimes
xr_class['time'] = datetimes

# create quick binary glacier mask of 0 and 1
glacier_mask = xr_cloud.max(dim='time')
glacier_mask = (glacier_mask>0)#.astype(int)

# set cloud images on glacier surface that are 0 to 1, to not confuse with off-glacier area
xr_cloud = xr.where( (xr_cloud==0) & (glacier_mask==1), 1, xr_cloud)
# so now where cloud==0 is considered not useable

# merge images on same day
xr_cloud = xr_cloud.where(xr_cloud<101, 0).groupby('time').max('time')
xr_class = xr_class.where(xr_class<20, 0).groupby('time').max('time')

# get these merged dates
datetimes_merged = xr_cloud.time.values

# get base dem of the glacier
xr_dem = snowFun.get_base_DEM(single_geometry.values[0])
# path_dem = os.path.join(folder_dems, "Region_10_2021.tif") #wolv=10, gulk=8
# xr_dem = riox.open_rasterio(path_dem).rio.write_nodata(0)
# xr_dem = xr_dem.rio.clip(single_geometry, from_disk=True, drop=True)/10

# xr_DEM_timevarying = snowFun.get_time_varying_DEM(single_geometry.values[0])

#%%
# first = xr_class.isel(time=-22) # snow=1, not-snow=0   #0snow_1firn_2ice_3rock
# # first = xr.where(glacier_mask, first, np.nan)

# fig,axs=plt.subplots()
# xr_dem.plot(ax=axs, cbar_kwargs={"label": "mask"})

# figure to check the products
t=95
# t='2016-07-09'
# fig,axs=plt.subplots(1,2, figsize=(10,5))
# xr_cloud.isel(time=t).plot(ax=axs[0], cbar_kwargs={"label": "cloud"})
# xr_class.isel(time=t).plot(ax=axs[1], cbar_kwargs={"label": "class"})
# plt.tight_layout()

# figure to check the products
# t='2018-09-04'
# fig,axs=plt.subplots(1,2, figsize=(10,5))
# xr_cloud.sel(time=t).plot(ax=axs[0], cbar_kwargs={"label": "cloud"})
# xr_class.sel(time=t).plot(ax=axs[1], cmap="Set1", vmin=-0.1, vmax=7.1, cbar_kwargs={"label": "class"})
# plt.tight_layout()

#%%
### create binary mask of useable and unuseable data
# bad_data = [2,3,8,9,10] # see S2 SCL band info for what these numbers mean
# good_data = [1,4,5,6,7,11]
# usable = xr_cloud.copy()#.astype(float)
# usable = xr.where(xr_cloud.isin(good_data), 1, 0)


cloud_thresh = 10 #maximum cloud probability that we will allow to be usable
bad_classes = [5] #class 5 is shadow. we will call these area unusable
good_classes = [1,2,3,4,6] #snow,firn,ice,debris,water are usable areas
usable = xr_cloud.copy()#.astype(float)
usable = xr.where( (xr_cloud<=cloud_thresh) & (xr_cloud>0) & (xr_class.isin(good_classes)), 1, 0)

# count total number of pixels on the glacier surface
glacier_pixels = glacier_mask.sum().values

# count usable pixels in each time step
count_usable_by_time = usable.sum(dim=['x','y']) 
percent_usable_by_time = count_usable_by_time/glacier_pixels

# count total, usable, and percent usable data at each pixel
count_total_by_pixel = xr.where( xr_class>0, 1, 0).sum('time')
count_usable_by_pixel = usable.sum(dim='time') 
percent_usable_by_pixel = count_usable_by_pixel/count_total_by_pixel # sum and divide by number of obs

fig,axs=plt.subplots()
percent_usable_by_pixel.plot(ax=axs, cbar_kwargs={"label": "% usable data"})

#%%
# now we can mask out unusable areas in each time step and then forward fill in
# the gaps using the most recent usable observation
good_times = (percent_usable_by_time>0.1) #remove date that are essentially wholly unusable

xr_class_masked = xr.where(usable==1, xr_class, np.nan).sel(time=good_times)
xr_class_filled = xr_class_masked.ffill(dim='time')
snow = xr.where(xr_class_filled==1, 1, 0)

# now plot percentage of snow cover through time
count_snow_by_time = snow.sum(dim=['x','y']) # total snow obs in each time
percent_snow_by_time = count_snow_by_time/glacier_pixels


fig,axs = plt.subplots(figsize=(8,4))
axs.scatter(percent_snow_by_time.time, percent_snow_by_time,
               c=percent_usable_by_time[good_times], vmin=0, vmax=1)

axs.set_xlabel('Date')
axs.set_ylabel("Percent snow cover")

fig,axs=plt.subplots()
snow.sum('time').plot(ax=axs, cbar_kwargs={"label": "snow frequency"})
axs.axis('equal')

#%%

# figure to check the products
# t='2020-08-17'
# fig,axs=plt.subplots(1,3, figsize=(14,5))
# xr_cloud.sel(time=t).plot(ax=axs[0], cbar_kwargs={"label": "cloud"})
# xr_class.sel(time=t).plot(ax=axs[1], cmap="Set1", vmin=-0.1, vmax=7.1, cbar_kwargs={"label": "class"})
# (snow.sel(time=t)/glacier_mask).plot(ax=axs[2], cmap="Blues", vmin=-0.1, vmax=1, cbar_kwargs={"label": "snow"})
# plt.tight_layout()

# fig,axs = plt.subplots()
# axs.contour(xr_dem.values[0], levels=[1800])

#%%
# extract ELAs from each time step, using the gap-filled product
import snowFun
glacier_ELAs = snowFun.get_the_ELAs(snow, xr_dem, glacier_mask, step=20, width=1, p_snow=0.5)

# lets add aar on to the df as well
glacier_ELAs['aar'] = percent_snow_by_time

#%%
# if ela is above the glacier then we get 9999. below and we get -1
# we can change these to the glacier min or max if we want (buffered by 1)
z_min = np.nanmin(xr_dem.where(xr_dem>0))
z_max = np.nanmax(xr_dem)
glacier_ELAs = glacier_ELAs.replace({'ela': {-1:z_min, 9999:z_max} })

#%%

### figure plotting the timeseries of aar and ela

fig,axs=plt.subplots(2,1, figsize=(8,6))

axs[0].scatter(glacier_ELAs['time'], glacier_ELAs['ela'], c='black', zorder=2, label='ELA')
axs[1].scatter(glacier_ELAs['time'], glacier_ELAs['aar'], c='black', zorder=2, label='AAR')

for ax in axs:
    ax.grid(zorder=1)
    ax.set_xlabel('Date')
    ax.legend()
    
axs[0].set_ylim(z_min,z_max)
axs[1].set_ylim(0,1)
   
fig.autofmt_xdate()
plt.tight_layout()

#%%
### figure plotting ELA against AAR to view relationship
ideal_ELAs = snowFun.idealized_ELA_AAR(xr_dem, glacier_mask)

fig, axs = plt.subplots()
axs.scatter(glacier_ELAs['ela'], glacier_ELAs['aar'])
axs.plot(ideal_ELAs['ela'], ideal_ELAs['aar'], c='black')
axs.set_xlabel('ELA')
axs.set_ylabel('AAR') 
axs.set_ylim(0,1)

#%%
# lets use this aar-ela relationship to root out bad observations
# for each aar we observed, see what the ideal ela would be
glacier_ELAs['aar_round'] = glacier_ELAs['aar'].round(2)
glacier_ELAs['ela_ideal'] = [ ideal_ELAs[ideal_ELAs['aar'].round(2)==i]['ela'].values[0] for i in glacier_ELAs['aar_round'] ]

# how about we incorporate a little error and see the range of elas we could expect
error_allowed = 0.1
glacier_ELAs['ela_ideal_min'] = [ ideal_ELAs[ideal_ELAs['aar'].round(2)==round(min(i+error_allowed,1),2)]['ela'].values[0] for i in glacier_ELAs['aar_round'] ]
glacier_ELAs['ela_ideal_max'] = [ ideal_ELAs[ideal_ELAs['aar'].round(2)==round(max(i-error_allowed,0),2)]['ela'].values[0] for i in glacier_ELAs['aar_round'] ]
glacier_ELAs['quality'] = [1 if (row['ela_ideal_min'] <= row['ela'] <= row['ela_ideal_max']) else 0 for idx,row in glacier_ELAs.iterrows()]

# now lets plot ela-aar relationship again, coloring by whether the observations fall within expected range    
fig, axs = plt.subplots()
axs.scatter(glacier_ELAs['ela'], glacier_ELAs['aar'], c=glacier_ELAs['quality'], cmap='Spectral', vmin=-0.1, vmax=1.1)
axs.plot(ideal_ELAs['ela'], ideal_ELAs['aar'], c='black')
axs.set_xlabel('ELA')
axs.set_ylabel('AAR') 
axs.set_ylim(0,1)

#%%
# plot the timeseries of the "good" elas
fig,axs=plt.subplots(2,1, figsize=(8,6))

axs[0].scatter(glacier_ELAs[glacier_ELAs['quality']==1]['time'], glacier_ELAs[glacier_ELAs['quality']==1]['ela'], c='black', zorder=2, label='ELA')
axs[1].scatter(glacier_ELAs[glacier_ELAs['quality']==1]['time'], glacier_ELAs[glacier_ELAs['quality']==1]['aar'], c='black', zorder=2, label='AAR')

for ax in axs:
    ax.grid(zorder=1)
    ax.set_xlabel('Date')
    ax.legend()

axs[0].set_ylim(z_min,z_max)
axs[1].set_ylim(0,1)

# fig.autofmt_xdate()
plt.tight_layout()

#%%
### get rolling median of quality obs in prior 15 days

# define function to do this
def get_rolling_median(df_obs, col_name, n_days, min_periods=1, center=False):
    temp_df = df_obs[['time',col_name]].set_index('time')
    medians = temp_df.rolling(f'{n_days}D', min_periods=min_periods, center=center).median()
    return medians

# subset df to good obs
glacier_ELAs_good = glacier_ELAs[glacier_ELAs['quality']==1].copy()
test = get_rolling_median(glacier_ELAs_good, 'ela', 30, center=False)
glacier_ELAs_good['ela_rolling'] = test.values
glacier_ELAs_good['ela_diff'] = glacier_ELAs_good['ela']-glacier_ELAs_good['ela_rolling']


# plot and color by difference from median rolling aar
fig,axs=plt.subplots(2,1, figsize=(8,6))

axs[0].scatter(glacier_ELAs_good['time'], glacier_ELAs_good['ela'], c=glacier_ELAs_good['ela_diff'], cmap='gist_heat', zorder=2, label='ELA')
axs[1].scatter(glacier_ELAs_good['time'], glacier_ELAs_good['aar'], c=glacier_ELAs_good['ela_diff'], cmap='gist_heat', zorder=2, label='AAR')

for ax in axs:
    ax.grid(zorder=1)
    ax.set_xlabel('Date')
    ax.legend()

axs[0].set_ylim(z_min,z_max)
axs[1].set_ylim(0,1)

# fig.autofmt_xdate()
plt.tight_layout()

#%%
# lets say that an ela that is 500 or more meter above the rolling median is bad
new_df = glacier_ELAs_good[glacier_ELAs_good['ela_diff']<400]

fig,axs=plt.subplots(2,1, figsize=(8,6))

axs[0].scatter(new_df['time'], new_df['ela'], c='black', zorder=2, label='ELA')
axs[1].scatter(new_df['time'], new_df['aar'], c='black', zorder=2, label='AAR')

for ax in axs:
    ax.grid(zorder=1)
    ax.set_xlabel('Date')
    ax.legend()

axs[0].set_ylim(z_min,z_max)
axs[1].set_ylim(0,1)

# fig.autofmt_xdate()
plt.tight_layout()

#%%
# so now from this, lets extract the snow distribution at the end of each year
df = new_df.copy()
df['time_index'] = df['time']
df = df.set_index('time_index')

ys = [2018,2019,2020,2021,2022]
target = 'ela'
data = []
for y in ys:

    # subset df to this year
    df_subset = df.loc[f'{y}-05-01':f'{y}-11-01']
    
    # get index of maximum ela
    ela_max = df_subset.loc[df_subset[target].idxmax()]
    
    data.append(ela_max)

max_elas = pd.DataFrame(data)
median_ela = np.nanmedian(max_elas[target])

# now go through and grab the snow distributions for these dates, plot them all
all_maps = []
fig, axs = plt.subplots(1,len(ys)+1, figsize=(14,4))
for d in range(len(data)):
    ax = axs[d]
    series = data[d]
    snow_map = snow.sel(time=series['time'].to_pydatetime())
    all_maps.append(snow_map.values)
    
    ax.imshow(snow_map/glacier_mask, cmap='Blues', vmin=-1, vmax=1)
    ax.axis('off')
    ax.set_title(f"ELA:{round(series[target])}\nDate:{series['time']}")
    
    # lets also contour the chosen ela on the glacier too
    ax.contour(xr_dem[0,:,:], levels=[round(series[target])], colors=['yellow'])

average_map = np.nanmedian(all_maps, axis=0)/glacier_mask
axs[-1].imshow(average_map, cmap='Blues', vmin=-1, vmax=1)
axs[-1].axis('off')
axs[-1].set_title(f"ELA:{round(median_ela)}\n5-year Average")

plt.tight_layout()




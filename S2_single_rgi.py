# -*- coding: utf-8 -*-
"""
This has been transfered to a jupyter notebook
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
glacier_choice = 'Gulkana'
rgi_single = rgi_gdf[rgi_gdf['Name']==f'{glacier_choice} Glacier'].to_crs("EPSG:3338")
single_geometry = rgi_single.geometry
# single_geometry = single_geometry.buffer(-100) #what if we buffer out the outside 100m

#%%
# open class and cloud files with rioxarray
if glacier_choice == 'Gulkana':
    file_name = "S2_RGI60-01.00570_2018-01-01_2023-01-01" #gulkana
elif glacier_choice == 'Wolverine':
    file_name = "S2_RGI60-01.09162_2015-01-01_2023-01-01" #wolverine

# open files, clipping to single outline
xr_class = riox.open_rasterio(os.path.join(folder_class, f"{file_name}.tif")).rio.clip(single_geometry, from_disk=True, drop=True)
# xr_cloud = riox.open_rasterio(os.path.join(folder_cloud, f"{file_name}_cloud.tif")).rio.clip(single_geometry, from_disk=True, drop=True)

# load metadata csv, convert date to datetimes
meta_fp = os.path.join(folder_meta, f"{file_name}.csv")
meta_df = pd.read_csv(meta_fp)

# format time axis for cloud and class
datetimes = pd.to_datetime([f"{str(i)[:4]}-{str(i)[4:6]}-{str(i)[6:]}" for i in meta_df['date']])
xr_class = xr_class.rename({"band":"time"})
# xr_cloud = xr_cloud.rename({"band":"time"})
# xr_cloud['time'] = datetimes
xr_class['time'] = datetimes

# create quick binary glacier mask of 0 and 1
glacier_mask = xr_class.max(dim='time')
glacier_mask = (glacier_mask>0)#.astype(int)

# set cloud images on glacier surface that are 0 to 1, to not confuse with off-glacier area
# xr_cloud = xr.where( (xr_cloud==0) & (glacier_mask==1), 1, xr_cloud)
# so now where cloud==0 is considered not useable

# merge images on same day
# xr_cloud = xr_cloud.where(xr_cloud<101, 0).groupby('time').max('time')
xr_class = xr_class.where(xr_class<20, 0).groupby('time').max('time')

# get these merged dates
datetimes_merged = xr_class.time.values

# get base dem of the glacier
xr_dem = snowFun.get_base_DEM(single_geometry.values[0])

print("everything loaded")
# testing not using cloud masking
# xr_cloud = xr.where(xr_cloud>0,1,xr_cloud)

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


cloud_thresh = 20 #maximum cloud probability that we will allow to be usable
bad_classes = [5] #class 5 is shadow. we will call these area unusable
good_classes = [1,2,3,4,6] #snow,firn,ice,debris,water are usable areas
# usable = xr_cloud.copy()#.astype(float)
# usable = xr.where( (xr_cloud<=cloud_thresh) & (xr_cloud>0) & (xr_class.isin(good_classes)), 1, 0)
usable = xr.where( xr_class.isin(good_classes), 1, 0)

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

print("everything masked. testing snow filling")
#%%
# first attempt at a filled snow product is just forward-filling the nans
snow1 = xr_class_masked.ffill(dim='time')
snow1 = xr.where(snow1==1, 1, 0)

#%%
# second attempt: calculate snow using moving-window smoothing
# create raw snow product where snow=1, other class=0, cloud=np.nan
snow_masked = xr.where(xr_class_masked.isin([2,3,4,6]), 0, xr_class_masked)

# Create a new empty dataarray with the same x/y shape, but with 1-day frequency
time_values = pd.to_datetime(snow_masked.time.values)
new_time_values = pd.date_range(start=time_values.min(), end=time_values.max(), freq='D')
snow2 = xr.DataArray(0, coords={'time': new_time_values, 'y': snow_masked.y, 'x': snow_masked.x },
                              dims=('time', 'y', 'x'))

# Update the new DataArray with available data from the original DataArray
snow2 = snow_masked.broadcast_like(snow2)

# now use rolling window mean and then extract the original good obs
snow2 = (snow2.rolling(time=21, min_periods=1, center=False).mean(skipna=False)).loc[dict(time=time_values)]

### make the floats to 0s and 1s, but preserve nans
snow2 = xr.where(snow2>=0.5, 1, snow2)
snow2 = xr.where(snow2<0.5, 0, snow2)

# third attempt: fill in the original classification missing data with the smoothed data
# snow3 = 0
snow3 = snow_masked.fillna(snow2)

#%%
# t=79
# fig,axs = plt.subplots(1,3, figsize=(10,4), sharex=True, sharey=True)
# axs[0].imshow(snow1.isel(time=t), vmin=0, vmax=1)
# axs[1].imshow(snow2.isel(time=t), vmin=0, vmax=1 )#* percent_usable_by_pixel)
# axs[2].imshow(snow3.isel(time=t), vmin=0, vmax=1 )

#%%
# figure showing the frequency of snow in each
# fig,axs = plt.subplots(1,3, figsize=(10,4), sharex=True, sharey=True)
# axs[0].imshow(snow1.mean('time', skipna=True), vmin=0, vmax=1)
# axs[1].imshow(snow2.mean('time', skipna=True), vmin=0, vmax=1)
# axs[2].imshow(snow3.mean('time', skipna=True), vmin=0, vmax=1)

#%%
print("snow filling done. now smoothing")
# now choose which of these products to use from here forward
snow = snow2

### smooth a little bit using some convolution
snow_x = snow.rolling({'x':3}, min_periods=1, center=True).sum()
snow_x = snow_x.rolling({'y':3}, min_periods=1, center=True).sum()
norm_x = (snow>-1).rolling({'x':3}, min_periods=1, center=True).sum()
norm_x = norm_x.rolling({'y':3}, min_periods=1, center=True).sum()

snow_x = snow_x/norm_x # this show what fraction of the 3x3 box around each pixel is snow
snow_x = xr.where(snow_x>=0.5, 1, snow_x)
snow_x = xr.where(snow_x<0.5, 0, snow_x)
snow_x = xr.where(glacier_mask==1, snow_x, np.nan)

snow = snow_x

# we need to recount the pixels in each image, how much is observable, etc...
count_snow_by_time = snow.sum(dim=['x','y'], skipna=True) # total snow obs in each time
count_all_by_time = xr.where(snow>-1,1,0).sum(dim=['x','y'])
percent_all_by_time = count_all_by_time/glacier_pixels
percent_snow_by_time = count_snow_by_time/count_all_by_time


# fig,axs = plt.subplots(1,2, figsize=(10,4), sharex=True, sharey=True)
# t=150
# axs[0].imshow(snow.isel(time=t), vmin=0, vmax=1)
# axs[1].imshow(snow_x.isel(time=t), vmin=0, vmax=1)
#%%
print("final steps")
# fig,axs = plt.subplots(figsize=(8,4))
# axs.scatter(percent_snow_by_time.time, percent_snow_by_time,
#                c=percent_usable_by_time[good_times], vmin=0, vmax=1)

# axs.set_xlabel('Date')
# axs.set_ylabel("Percent snow cover")

# filter out the dates that have less than x% usable data
usable_thresh = 0.85
snow = snow.sel(time=(percent_all_by_time>usable_thresh))

#%%
### make a figure showing the frequency of each data type
titles = ['snow', 'firn', 'ice', 'debris', 'water']
cmap='RdPu_r'

# fig,axs = plt.subplots(1,5, figsize=(14,4))
# axs[0].imshow( xr.where(xr_class_masked==1, 1, 0).sum(dim=['time']) * percent_usable_by_pixel, cmap=cmap)#, vmin=0, vmax=100)
# axs[1].imshow( xr.where(xr_class_masked==2, 1, 0).sum(dim=['time']) * percent_usable_by_pixel, cmap=cmap)#, vmin=0, vmax=100)
# axs[2].imshow( xr.where(xr_class_masked==3, 1, 0).sum(dim=['time']) * percent_usable_by_pixel, cmap=cmap)#, vmin=0, vmax=100)
# axs[3].imshow( xr.where(xr_class_masked==4, 1, 0).sum(dim=['time']) * percent_usable_by_pixel, cmap=cmap)#, vmin=0, vmax=100)
# axs[4].imshow( xr.where(xr_class_masked==6, 1, 0).sum(dim=['time']) * percent_usable_by_pixel, cmap=cmap)#, vmin=0, vmax=100)

# for i in range(len(titles)):
#     ax=axs[i]
#     ax.set_title(titles[i])
#     ax.axis('off')

# plt.tight_layout()


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

#%%
# extract ELAs from each time step, using the gap-filled product
import snowFun
glacier_ELAs = snowFun.get_the_ELAs(snow, xr_dem, glacier_mask, step=20, width=1, p_snow=0.5)

# lets add aar on to the df as well
glacier_ELAs['aar'] = percent_snow_by_time.sel(time=(percent_all_by_time>usable_thresh))

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
   
# fig.autofmt_xdate()
plt.tight_layout()

#%%
### figure plotting ELA against AAR to view relationship
ideal_ELAs = snowFun.idealized_ELA_AAR(xr_dem, glacier_mask)

# fig, axs = plt.subplots()
# axs.scatter(glacier_ELAs['aar'], glacier_ELAs['ela'])
# axs.plot(ideal_ELAs['aar'], ideal_ELAs['ela'], c='black')
# axs.set_xlabel('AAR')
# axs.set_ylabel('ELA') 
# axs.set_xlim(0,1)

#%%
# lets use this aar-ela relationship to root out bad observations
# for each aar we observed, see what the ideal ela would be
glacier_ELAs['aar_round'] = glacier_ELAs['aar'].round(2)
glacier_ELAs['ela_ideal'] = [ ideal_ELAs[ideal_ELAs['aar'].round(2)==i]['ela'].values[0] for i in glacier_ELAs['aar_round'] ]

# how about we incorporate a little error and see the range of elas we could expect
error_allowed = 0.2
glacier_ELAs['ela_ideal_min'] = [ ideal_ELAs[ideal_ELAs['aar'].round(2)==round(min(i+error_allowed,1),2)]['ela'].values[0] for i in glacier_ELAs['aar_round'] ]
glacier_ELAs['ela_ideal_max'] = [ ideal_ELAs[ideal_ELAs['aar'].round(2)==round(max(i-error_allowed,0),2)]['ela'].values[0] for i in glacier_ELAs['aar_round'] ]
glacier_ELAs['quality'] = [1 if (row['ela_ideal_min'] <= row['ela'] <= row['ela_ideal_max']) else 0 for idx,row in glacier_ELAs.iterrows()]

# now lets plot ela-aar relationship again, coloring by whether the observations fall within expected range    
fig, axs = plt.subplots()
axs.scatter(glacier_ELAs['aar'], glacier_ELAs['ela'], c=glacier_ELAs['quality'], cmap='Spectral', vmin=-0.1, vmax=1.1)
axs.plot(ideal_ELAs['aar'], ideal_ELAs['ela'], c='black')
axs.set_xlabel('AAR')
axs.set_ylabel('ELA') 
axs.set_xlim(0,1)

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
### get rolling median of quality ela obs in prior x days

# define function to do this
def get_rolling_median(df_obs, col_name, n_days, min_periods=1, center=False, closed='left'):
    temp_df = df_obs[['time',col_name]].set_index('time')
    medians = temp_df.rolling(f'{n_days}D', min_periods=min_periods, center=center, closed=closed).median()
    return medians

# subset df to good obs
glacier_ELAs_good = glacier_ELAs[glacier_ELAs['quality']==1].copy()
test = get_rolling_median(glacier_ELAs_good, 'ela', 30, center=False)
glacier_ELAs_good['ela_rolling'] = test.values
glacier_ELAs_good['ela_diff'] = glacier_ELAs_good['ela']-glacier_ELAs_good['ela_rolling']


# # plot and color by difference from median rolling aar
# fig,axs=plt.subplots(2,1, figsize=(8,6))

# axs[0].scatter(glacier_ELAs_good['time'], glacier_ELAs_good['ela'], c=glacier_ELAs_good['ela_diff'], cmap='gist_heat', zorder=2, label='ELA')
# axs[1].scatter(glacier_ELAs_good['time'], glacier_ELAs_good['aar'], c=glacier_ELAs_good['ela_diff'], cmap='gist_heat', zorder=2, label='AAR')

# for ax in axs:
#     ax.grid(zorder=1)
#     ax.set_xlabel('Date')
#     ax.legend()

# axs[0].set_ylim(z_min,z_max)
# axs[1].set_ylim(0,1)

# # fig.autofmt_xdate()
# plt.tight_layout()

#%%
# lets say that an ela that is 400 or more meter above the rolling median is bad
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
    df_subset = df.loc[f'{y}-07-01':f'{y}-11-01']
    
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

average_map = np.nanmedian(all_maps, axis=0).astype(int)/glacier_mask
axs[-1].imshow(average_map, cmap='Blues', vmin=-1, vmax=1)
axs[-1].axis('off')
axs[-1].set_title(f"ELA:{round(median_ela)}\n5-year Average")

plt.tight_layout()




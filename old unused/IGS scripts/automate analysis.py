# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:05:51 2022

@author: lzell
"""

import os
import rasterio as rio
import numpy as np
from rasterio.merge import merge as riomerge
from rasterio.plot import show as rioshow
import matplotlib.pyplot as plt
from osgeo import gdal
import glob
import subprocess
import geopandas as gpd
import pandas as pd
from rasterio.mask import mask
from datetime import datetime
import shapely
from shapely import geometry
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing

# set folder
agva_folder = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop','AGVA')
os.chdir(agva_folder)

# load image path
images_folder = os.path.join(agva_folder, 'classified images', 'L8 Classified Images PC')
images_folder = os.path.join(agva_folder, 'classified images', 'L8 Classified Images no PC')

img_name = "P68_R18_2015-01-01_2021-12-30_90"
image_path = os.path.join(images_folder, img_name+".tif")

# load images metadata
meta_path = os.path.join(agva_folder, 'classified images', 'meta csv', img_name+'.csv')
img_data = pd.read_csv(meta_path)
img_data['date'] = img_data.apply(lambda row: datetime.strptime(str(row.id[-8:]), '%Y%m%d'), axis=1)
img_data['year'] = img_data.apply(lambda row: row.date.year, axis=1)

# load rgi outlines
rgi_path = os.path.join(agva_folder, 'RGI', '01_rgi60_Alaska', '01_rgi60_Alaska.shp')
rgi_gdf = gpd.read_file(rgi_path)

wolv_ba = np.array([-0.85, -0.14, -1.07, -1.87, -1.46, -1.81, -0.87])
wolv_ela = np.array([1237, 1193, 1270, 1369, 1259, 1336, 1237])
years = [2015,2016,2017,2018,2019,2020,2021]

#%%

# open rio image object, get crs
image_rio = rio.open(image_path)
image_crs = image_rio.crs
image_bounds = image_rio.bounds

# make image bounds into a shapely object (to filter the outlines shapefile with)
image_bbox = geometry.box(*image_bounds)
xmin,ymin,xmax,ymax = (image_bounds[0],image_bounds[1],image_bounds[2],image_bounds[3])

# use only glaciers greater than 1 km2
rgi_gdf_large = rgi_gdf[rgi_gdf.Area >= 2]

# reproject outlines to the image crs
rgi_gdf_large = rgi_gdf_large.to_crs(image_crs)
wolv_gdf = rgi_gdf_large[rgi_gdf_large.Name == 'Wolverine Glacier']

# filter outlines to those that overlap our image
rgi_image = rgi_gdf_large.cx[xmin:xmax, ymin:ymax].copy()
print(len(rgi_image), " glaciers >2 km2 are in this image")
#rgi_image = rgi_gdf_large[rgi_gdf_large.Name == 'Wolverine Glacier']

# quickly subselect only a few if you want
rgi_image = rgi_image[100:200]
print(len(rgi_image), " glaciers will be analyzed here")


# list of the subselected geoemtries
#geometries_list = [i for i in rgi_image['geometry']]

#%%
# show a quick figure to make sure the right geometries have been selected
fig,ax = plt.subplots(figsize=(10,10))
rgi_image.plot(ax=ax, color='black')
ax.plot(*image_bbox.exterior.xy)

#%%
# add empty columns to df
for y in range(2015,2022):
    rgi_image["min "+str(y)] = np.nan
    rgi_image["min aar "+str(y)] = np.nan
    rgi_image["min date "+str(y)] = np.nan

def analyze_images(input_list): # input = [index, row]
    index = input_list[0]
    rgi = input_list[1]
    
    with rio.open(image_path) as src:
        image_clip, out_transform = mask(dataset=src, shapes=[rgi.geometry], crop=True, nodata=99)
        out_meta = src.meta
        
    image = image_clip.astype(float)
    image[image==99] = np.nan
    
    # find area of the image, in km2
    area_found = np.count_nonzero(~np.isnan(image[0,:,:]))*30*30 / (1000*1000)
    area_rgi = rgi.Area
    
    ### if the area found is <80% of the rgi area, skip it
    if area_found < area_rgi*0.8: 
        for y in range(2015,2022):
            rgi_image.loc[index,"min "+str(y)] = np.nan
            rgi_image.loc[index,"min date "+str(y)] = np.nan
            rgi_image.loc[index,"min aar "+str(y)] = np.nan
        return [index, np.nan, np.nan, np.nan]
    
    ### otherwise, calculate the minimum accumulation area in each year
    
    # make copy df to store per-image info
    rgi_data = img_data.copy()
    
    # calculate area of snow, firn and ice in each image
    rgi_data['area_snow'] = np.sum(image==0, axis=(1,2)) * 30 * 30 / 1000000 
    rgi_data['area_firn'] = np.sum(image==1, axis=(1,2)) * 30 * 30 / 1000000 
    rgi_data['area_ice'] = np.sum(image==2, axis=(1,2)) * 30 * 30 / 1000000 
    
    rgi_data['area_snow_rel'] = np.sum(image==0, axis=(1,2)) * 30 * 30 / (1000000*area_rgi) 
    rgi_data['area_firn_rel'] = np.sum(image==1, axis=(1,2)) * 30 * 30 / (1000000*area_rgi)
    rgi_data['area_ice_rel'] = np.sum(image==2, axis=(1,2)) * 30 * 30 / (1000000*area_rgi)

    # option to filter by cloudy percentage if you want
    df_to_use = rgi_data[rgi_data['cloud_cover_land']<100]
    
    # calculate minimum each year
    years = [2015,2016,2017,2018,2019,2020,2021]
    dfs_by_year = []
    max_n = 0
    
    for y in years:
        df_sub = df_to_use[df_to_use.year == y]
        dfs_by_year.append(df_sub)
        if df_sub.shape[0]>max_n: max_n=df_sub.shape[0]
    
    # find minimum accumulation area in each year
    min_aa = []
    min_aa_date = []
    
    y=2015
    for df in dfs_by_year:
        areas = df['area_snow']+df['area_firn']
        areas_rel = df['area_snow_rel']+df['area_firn_rel']
        
        rgi_image.loc[index, "min "+str(y)] = np.nanmin(areas)
        rgi_image.loc[index, "min aar "+str(y)] = np.nanmin(areas_rel)
        rgi_image.loc[index, "min date "+str(y)] = max(df[df['area_snow']+df['area_firn']==np.nanmin(areas)].date)
        y+=1
        #return([index, np.nanmin(areas), np.nanmin(areas_rel), max(df[df['area_snow']+df['area_firn']==np.nanmin(areas)].date)])
        # min_aa.append(np.nanmin(areas))
        # min_aa_date.append(max(df[df['area_snow']+df['area_firn']==np.nanmin(areas)].date))

# test out multithreading
multithread = 0

if multithread == 0:
      
    # loop through each geometry
    c=0
    total = len(rgi_image)
    
    for index, rgi in rgi_image.iterrows():
        if c%10==0:
            print(c,'out of',total,'analyzed')
        c+=1
        
        analyze_images([index,rgi])
        
        # with rio.open(image_path) as src:
        #     image_clip, out_transform = mask(dataset=src, shapes=[rgi.geometry], crop=True, nodata=99)
        #     out_meta = src.meta
            
        # image = image_clip.astype(float)
        # image[image==99] = np.nan
        
        # # find area of the image, in km2
        # area_found = np.count_nonzero(~np.isnan(image[0,:,:]))*30*30 / (1000*1000)
        # area_rgi = rgi.Area
        
        # ### if the area found is <80% of the rgi area, skip it
        # if area_found < area_rgi*0.8: 
        #     for y in range(2015,2022):
        #         rgi_image.loc[index,"min "+str(y)] = np.nan
        #         rgi_image.loc[index,"min date "+str(y)] = np.nan
        #         rgi_image.loc[index,"min aar "+str(y)] = np.nan
        #     continue
        
        # ### otherwise, calculate the minimum accumulation area in each year
        
        # # make copy df to store per-image info
        # rgi_data = img_data.copy()
        
        # # calculate area of snow, firn and ice in each image
        # rgi_data['area_snow'] = np.sum(image==0, axis=(1,2)) * 30 * 30 / 1000000 
        # rgi_data['area_firn'] = np.sum(image==1, axis=(1,2)) * 30 * 30 / 1000000 
        # rgi_data['area_ice'] = np.sum(image==2, axis=(1,2)) * 30 * 30 / 1000000 
        
        # rgi_data['area_snow_rel'] = np.sum(image==0, axis=(1,2)) * 30 * 30 / (1000000*area_rgi) 
        # rgi_data['area_firn_rel'] = np.sum(image==1, axis=(1,2)) * 30 * 30 / (1000000*area_rgi)
        # rgi_data['area_ice_rel'] = np.sum(image==2, axis=(1,2)) * 30 * 30 / (1000000*area_rgi)
    
        # # option to filter by cloudy percentage if you want
        # df_to_use = rgi_data[rgi_data['cloud_cover_land']<100]
        
        # # calculate minimum each year
        # years = [2015,2016,2017,2018,2019,2020,2021]
        # dfs_by_year = []
        # max_n = 0
        
        # for y in years:
        #     df_sub = df_to_use[df_to_use.year == y]
        #     dfs_by_year.append(df_sub)
        #     if df_sub.shape[0]>max_n: max_n=df_sub.shape[0]
        
        # # find minimum accumulation area in each year
        # min_aa = []
        # min_aa_date = []
        
        # y=2015
        # for df in dfs_by_year:
        #     areas = df['area_snow']+df['area_firn']
        #     areas_rel = df['area_snow_rel']+df['area_firn_rel']
            
        #     rgi_image.loc[index, "min "+str(y)] = np.nanmin(areas)
        #     rgi_image.loc[index, "min aar "+str(y)] = np.nanmin(areas_rel)
        #     rgi_image.loc[index, "min date "+str(y)] = max(df[df['area_snow']+df['area_firn']==np.nanmin(areas)].date)
            
        #     # min_aa.append(np.nanmin(areas))
        #     # min_aa_date.append(max(df[df['area_snow']+df['area_firn']==np.nanmin(areas)].date))
            
        #     y+=1



else:
    # make a list so that you can use mulitprocessing
    list_of_rows = []
    # for index, rgi in rgi_image.iterrows():
    #         list_of_rows.append([index,rgi])  

    # if __name__ == '__main__':
    #     pool = multiprocessing.Pool(4)
    #     result = pool.map(analyze_images, list_of_rows)
        
    #     for r in result:
    #         rgi_image.loc[r[0], "min "+str(y)] = np.nanmin(areas)
    #         rgi_image.loc[r[0], "min aar "+str(y)] = np.nanmin(areas_rel)
    #         rgi_image.loc[r[0], "min date "+str(y)] = 0
    #rgi_image.apply(lambda row: analyze_images_lambda(row), axis=1)

#%%
# per-glacier mean and median aar of all years
rgi_image['mean aar'] = rgi_image.apply(lambda row: np.nanmean([row['min aar 2015'],
                                                                row['min aar 2016'],
                                                                row['min aar 2017'],
                                                                row['min aar 2018'],
                                                                row['min aar 2019'],
                                                                row['min aar 2020'],
                                                                row['min aar 2021'],
                                                                ]), axis=1)

rgi_image['median aar'] = rgi_image.apply(lambda row: np.nanmedian([row['min aar 2015'],
                                                                    row['min aar 2016'],
                                                                    row['min aar 2017'],
                                                                    row['min aar 2018'],
                                                                    row['min aar 2019'],
                                                                    row['min aar 2020'],
                                                                    row['min aar 2021'],
                                                                    ]), axis=1)

# per year variation from median
for y in years:
    rgi_image[str(y)+' aar variation'] = rgi_image.apply(lambda row: row['min aar '+str(y)] - row['median aar'], axis=1)

#%%
# calculate mean of each column each year
means = []
medians = []
aar_vars = []

for y in years:
    mea = np.nanmean(rgi_image["min aar "+str(y)].astype(float))
    med = np.nanmedian(rgi_image["min aar "+str(y)].astype(float))
    var = np.nanmean(rgi_image[str(y)+' aar variation'].astype(float))
    means.append(mea)
    medians.append(med)
    aar_vars.append(var)

#%%
# scatterplot showing the average AAR of all glaciers plotted against the wolverine ELAs
fig,ax = plt.subplots(1,2, figsize=(8,3))
ax[0].scatter(means,wolv_ela*-1,marker='X', c='tab:red', s=50)
ax[1].scatter(medians,wolv_ela*-1,marker='X', c='tab:red', s=50)

ax[0].set_xlabel('Mean AAR')
ax[1].set_xlabel('Median AAR')
ax[0].set_ylabel('Wolv ELA')
ax[1].set_ylabel('Wolv ELA')

plt.tight_layout() 

#%%
# scatterplot showing the average AAR variation plotted against wolverine ELAs
fig,ax = plt.subplots( figsize=(4,3))
ax.scatter(aar_vars,wolv_ela*-1,marker='X', c='tab:red', s=50)

ax.set_xlabel('AAR variation')
ax.set_ylabel('Wolv ELA')

plt.tight_layout() 

#%%
# map showing the outlines colored by mean aar of all years
fig,ax = plt.subplots(figsize=(10,8))
rgi_image.plot(ax=ax, column="median aar", legend=True, cmap='viridis', vmin=0, vmax=1, legend_kwds={'label': "Average AAR 2015-2021"})

#%%
# maps showing the annual variation of each glacier each year
fig,axs = plt.subplots(3,3, figsize=(12,12), sharex=True, sharey=True)
c=0
for y in years:
    a = axs[c//3,c%3]
    
    if y==2021: #add with colorbar
        divider = make_axes_locatable(axs[2,1])
        cax = divider.append_axes("left", size="5%", pad=0.1)
        rgi_image.plot(ax=a, column=str(y)+' aar variation', cmap='coolwarm_r', vmin=-0.3, vmax=0.3, legend=True, cax=cax)
    else:
        rgi_image.plot(ax=a, column=str(y)+' aar variation', cmap='coolwarm_r', vmin=-0.3, vmax=0.3)
    
    a.set_title(y)
    a.axis('off')
    c+=1
axs[2,1].axis('off')
axs[2,2].axis('off')
plt.tight_layout()
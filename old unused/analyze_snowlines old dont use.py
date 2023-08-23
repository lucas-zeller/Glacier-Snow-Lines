# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:44:00 2022

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

# set folder
agva_folder = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop','AGVA')
os.chdir(agva_folder)

# load image path
images_folder = os.path.join(agva_folder, 'classified')
img_name = "P68_R18_2015-01-01_2021-12-30_90"
image_path = os.path.join(images_folder, img_name, 'merged_'+img_name+'.tif')

# load images metadata
meta_path = os.path.join(images_folder, 'meta csv', img_name+'.csv')
img_data = pd.read_csv(meta_path)
img_data['date'] = img_data.apply(lambda row: datetime.strptime(str(row.id[-8:]), '%Y%m%d'), axis=1)

# load rgi outlines
rgi_path = os.path.join(agva_folder, 'RGI', '01_rgi60_Alaska', '01_rgi60_Alaska.shp')
rgi_gdf = gpd.read_file(rgi_path)

wolv_gdf = rgi_gdf[rgi_gdf.Name == 'Wolverine Glacier']

#%%

# open rio image object, get crs
image_rio = rio.open(image_path)
image_crs = image_rio.crs

wolv_gdf = wolv_gdf.to_crs(image_crs)

#%%

with rio.open(image_path) as src:
        image_clip, out_transform = mask(dataset=src, shapes=wolv_gdf.geometry, crop=True, nodata=99)
        out_meta = src.meta
image = image_clip.astype(float)
image[image==99] = np.nan

#%%

# show first 50 figures
fig, axs = plt.subplots(10,5)
c=0
for x in range(10):
    for y in range(5):
        axs[x,y].imshow(image[c,:,:], cmap='Blues',vmin=-0.5,vmax=2)
        axs[x,y].axis('off')
        c+=1


#%%
# load total snow, firn, ice area in each image
# 0snow 1firn 2ice 3rock
img_data['area_snow'] = np.sum(image==0, axis=(1,2)) * 30 * 30 / 1000000 
img_data['area_firn'] = np.sum(image==1, axis=(1,2)) * 30 * 30 / 1000000 
img_data['area_ice'] = np.sum(image==2, axis=(1,2)) * 30 * 30 / 1000000 


fig,axs = plt.subplots(2,figsize=(9,9), sharex=True,sharey=True)

dots_snow = axs[0].scatter( img_data['date'], img_data['area_snow'], c=img_data['cloud_cover_land'], cmap='Spectral', label='snow')
dots_sf = axs[1].scatter( img_data['date'], img_data['area_snow']+img_data['area_firn'], c=img_data['cloud_cover_land'], cmap='Spectral', label='snow and firn')

for ax in axs:
    ax.axvline(datetime.strptime('20150901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed', label="September 1")
    ax.axvline(datetime.strptime('20160901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed')
    ax.axvline(datetime.strptime('20170901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed')
    ax.axvline(datetime.strptime('20180901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed')
    ax.axvline(datetime.strptime('20190901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed')
    ax.axvline(datetime.strptime('20200901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed')
    ax.axvline(datetime.strptime('20210901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed')

plt.xlabel('Date')
plt.ylabel('Area')
plt.colorbar(dots_snow, label='cloud cover', ax=axs[0])
plt.colorbar(dots_sf, label='cloud cover', ax=axs[1])
axs[0].legend()
axs[1].legend()

#plt.setp( axs[-1].xaxis.get_majorticklabels(), rotation=-45, ha="left" )
fig.autofmt_xdate()
fig.tight_layout() 


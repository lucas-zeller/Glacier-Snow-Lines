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
images_folder = os.path.join(agva_folder, 'classified', 'no PC')
img_name = "P68_R18_2015-01-01_2021-12-30_90"
image_path = os.path.join(images_folder, img_name, 'merged_'+img_name+'.tif')

# load images metadata
meta_path = os.path.join(agva_folder, 'classified', 'meta csv', img_name+'.csv')
img_data = pd.read_csv(meta_path)
img_data['date'] = img_data.apply(lambda row: datetime.strptime(str(row.id[-8:]), '%Y%m%d'), axis=1)
img_data['year'] = img_data.apply(lambda row: row.date.year, axis=1)

# load rgi outlines
rgi_path = os.path.join(agva_folder, 'RGI', '01_rgi60_Alaska', '01_rgi60_Alaska.shp')
rgi_gdf = gpd.read_file(rgi_path)

# select a single rgi outline
wolv_gdf = rgi_gdf[rgi_gdf.Name == 'Wolverine Glacier']

# load in situ measurements
wolv_ba = [-0.85, -0.14, -1.07, -1.87, -1.46, -1.81, -0.87]
wolv_ela = [1237, 1193, 1270, 1369, 1259, 1336, 1237]
wolv_insitu_years = ['20150901','20160901','20170901','20180901','20190901','20200901','20210901']
for y in range(7): wolv_insitu_years[y] = datetime.strptime(wolv_insitu_years[y], '%Y%m%d')


#%%

# open rio image object, get crs
image_rio = rio.open(image_path)
image_crs = image_rio.crs

# reproject outline to proper crs
wolv_gdf = wolv_gdf.to_crs(image_crs)

# open classification image, clip to glacier area
with rio.open(image_path) as src:
        image_clip, out_transform = mask(dataset=src, shapes=wolv_gdf.geometry, crop=True, nodata=99)
        out_meta = src.meta
image = image_clip.astype(float)
image[image==99] = np.nan

# find total snow, firn, ice area in each image
# 0snow 1firn 2ice 3rock
img_data['area_snow'] = np.sum(image==0, axis=(1,2)) * 30 * 30 / 1000000 
img_data['area_firn'] = np.sum(image==1, axis=(1,2)) * 30 * 30 / 1000000 
img_data['area_ice'] = np.sum(image==2, axis=(1,2)) * 30 * 30 / 1000000 

#%%
# filter by cloud cover here (if you want)
df_to_use = img_data[img_data['cloud_cover_land']<100]

#%%
# sort into separate df by year
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

for df in dfs_by_year:
    areas = df['area_snow']+df['area_firn']
    min_aa.append(np.nanmin(areas))
    min_aa_date.append(max(df[df['area_snow']+df['area_firn']==np.nanmin(areas)].date))

#%%
### figure showing all classification images
fig, axs = plt.subplots(len(years),max_n)
c=0
for y in range(len(years)):
    
    for i in range(max_n):
        # check if this image exists, add image to the figure if it does
        if i<dfs_by_year[y].shape[0]:
            axs[y,i].imshow(image[c,:,:], cmap='Blues',vmin=-0.5,vmax=2)
            c+=1
        axs[y,i].axis('off')
    axs[y,0].set_ylabel(years[y])
        # axs[x,y].imshow(image[c,:,:], cmap='Blues',vmin=-0.5,vmax=2)
        # axs[x,y].axis('off')
        # c+=1


#%%
### figure plotting the total snow area in each image over time, also total snow+firn area through time, colored by whole-image cloudy percentage
fig,axs = plt.subplots(3,figsize=(9,9), sharex=True, sharey=False)


dots_snow = axs[0].scatter( df_to_use['date'], df_to_use['area_snow'], c=df_to_use['cloud_cover_land'], cmap='Spectral', vmin=0, vmax=100, label='snow')
dots_sf = axs[1].scatter( df_to_use['date'], df_to_use['area_snow']+df_to_use['area_firn'], c=df_to_use['cloud_cover_land'], cmap='Spectral', vmin=0, vmax=100, label='ice')
axs[2].scatter(wolv_insitu_years, wolv_ela, marker='X', c='black', s=50)
# axs3 = axs[2].twinx()
# axs3.scatter(wolv_insitu_years, wolv_ba, marker='o', c='black', s=50)

for ax in axs[0:2]:
    ax.axvline(datetime.strptime('20150901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed', label="September 1")
    ax.axvline(datetime.strptime('20160901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed')
    ax.axvline(datetime.strptime('20170901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed')
    ax.axvline(datetime.strptime('20180901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed')
    ax.axvline(datetime.strptime('20190901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed')
    ax.axvline(datetime.strptime('20200901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed')
    ax.axvline(datetime.strptime('20210901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed')

axs[2].set_xlabel('Date')
axs[0].set_ylabel('Snow Area')
axs[1].set_ylabel('Snow & Firn Area')
axs[2].set_ylabel('ELA')
#axs3.set_ylabel('Annual Balance')

axs[2].invert_yaxis()

#plt.colorbar(dots_snow, label='cloud cover', ax=axs[0])
# plt.colorbar(dots_sf, label='cloud cover', ax=axs[1])
# axs[0].legend()
# axs[1].legend()

#plt.setp( axs[-1].xaxis.get_majorticklabels(), rotation=-45, ha="left" )
fig.autofmt_xdate()
fig.tight_layout() 

#%%
### figure plotting the total total snow+firn area through time, with sinple colorscheme
fig,axs = plt.subplots(2,figsize=(9,5), sharex=True, sharey=False)

#dots_snow = axs[0].scatter( img_data['date'], img_data['area_snow'], c=img_data['cloud_cover_land'], cmap='Spectral', label='snow')
dots_sf = axs[0].scatter( df_to_use['date'], df_to_use['area_snow']+df_to_use['area_firn'], c='black', cmap='Spectral', label='ice')
axs[1].scatter(wolv_insitu_years, wolv_ela, marker='X', c='black', s=50)
# axs3 = axs[1].twinx()
# axs3.scatter(wolv_insitu_years, wolv_ba, marker='o', c='black', s=50)

for ax in axs[0:2]:
    ax.axvline(datetime.strptime('20150901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed', label="September 1")
    ax.axvline(datetime.strptime('20160901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed')
    ax.axvline(datetime.strptime('20170901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed')
    ax.axvline(datetime.strptime('20180901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed')
    ax.axvline(datetime.strptime('20190901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed')
    ax.axvline(datetime.strptime('20200901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed')
    ax.axvline(datetime.strptime('20210901', '%Y%m%d'), color="black", alpha=0.5, linestyle='dashed')

axs[1].set_xlabel('Date')
axs[0].set_ylabel('Snow Area')
axs[1].set_ylabel('ELA')
#axs3.set_ylabel('Annual Balance')

axs[1].invert_yaxis()

#plt.colorbar(dots_snow, label='cloud cover', ax=axs[0])
# plt.colorbar(dots_sf, label='cloud cover', ax=axs[1])
# axs[0].legend()
# axs[1].legend()

#plt.setp( axs[-1].xaxis.get_majorticklabels(), rotation=-45, ha="left" )
fig.autofmt_xdate()
fig.tight_layout() 

#%%
### scatterplot showing the minimum accumulation area in each year plotted against the in situ ela and in situ annual balance
fig,axs = plt.subplots(1,3,figsize=(8,2.5))
axs[0].scatter(min_aa,wolv_ela,marker='X', c='black', s=50)
axs[1].scatter(min_aa,wolv_ba,marker='X', c='black', s=50)
axs[2].scatter(wolv_ela,wolv_ba,marker='X', c='black', s=50)

axs[0].set_xlabel('Automated Accumulation Area')
axs[1].set_xlabel('Automated Accumulation Area')
axs[2].set_xlabel('In situ ela')

axs[0].set_ylabel('In situ ela')
axs[1].set_ylabel('In situ Ba')
axs[2].set_ylabel('In situ Ba')

plt.tight_layout() 












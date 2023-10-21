# -*- coding: utf-8 -*-
"""
Import and visualize training data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from datetime import datetime
import rasterio as rio
from rasterio.mask import mask
import pandas as pd
import geopandas as gpd
import copy
import time
import multiprocessing
import json

from scipy.ndimage import uniform_filter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error


#%%
save = 0 

# define training folder path
training_folder_path = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA","training_data")

df_1 = pd.read_csv(os.path.join(training_folder_path,'LC08_055020_20160827_training_dataset_0snow_1firn_2ice_3rock_4water_5shadowedsnow.csv')).drop('.geo', axis=1)
df_2 = pd.read_csv(os.path.join(training_folder_path,'LC08_067016_20190808_training_dataset_0snow_1firn_2ice_3rock_4water_5shadowedsnow.csv')).drop('.geo', axis=1)
df_3 = pd.read_csv(os.path.join(training_folder_path,'LC08_067017_20160831_training_dataset_0snow_1firn_2ice_3rock_4water_5shadowedsnow.csv')).drop('.geo', axis=1)
df_4 = pd.read_csv(os.path.join(training_folder_path,'LC08_068018_20180929_training_dataset_0snow_1firn_2ice_3rock_4water_5shadowedsnow.csv')).drop('.geo', axis=1)
df_5 = pd.read_csv(os.path.join(training_folder_path,'LC08_070017_20200815_training_dataset_0snow_1firn_2ice_3rock_4water_5shadowedsnow.csv')).drop('.geo', axis=1)

dfs = [df_1, df_2, df_3, df_4, df_5]

print(df_1.columns)

#%%
#rename bands
renames = {'SR_B1':'coastal','SR_B2':'blue','SR_B3':'green','SR_B4':'red','SR_B5':'nir','SR_B6':'swir1','SR_B7':'swir2'}
for d in dfs:
    d.rename(columns=renames,inplace=True)
    
print(df_1.columns)

#%%
# add columns

def add_ndwi(df):
    green = df['green']
    nir = df['nir']
    ndwi = (green-nir)/(green+nir)
    df['ndwi'] = ndwi

def add_ndsi(df):
    green = df['green']
    swir1 = df['swir1']
    ndsi = (green-swir1)/(green+swir1)
    df['ndsi'] = ndsi

def ndwi_times_ndsi(df):
    ndwi = df['ndwi']+1
    ndsi = df['ndsi']+1
    mult = ndwi*ndsi
    df['ndwi_ndsi'] = mult
    
def convert_temp(df):
    b10 = df['ST_B10']
    temp = b10*0.00341802+149-273.15
    df['temp_c'] = temp

for d in dfs:
    add_ndwi(d)
    add_ndsi(d)
    convert_temp(d)
    ndwi_times_ndsi(d)

df_master = pd.concat(dfs)
print('Observations')
print('Snow:', len(df_master[df_master['type']==0]))
print('Firn:', len(df_master[df_master['type']==1]))
print('Ice: ', len(df_master[df_master['type']==2]))
#%%
# summary figures
show_ice = 1
c_ice = 'tab:blue'
c_firn = 'tab:brown'
c_snow = 'tab:olive'


fig, axs = plt.subplots(3,4, figsize=(9,9))

axs[0,0].hist(df_master[df_master['type']==0]['coastal'], bins=range(0,50000,1000), alpha=0.5, color=c_snow, label='snow')
axs[0,0].hist(df_master[df_master['type']==1]['coastal'], bins=range(0,50000,1000), alpha=0.5, color=c_firn, label='firn')
if show_ice: axs[0,0].hist(df_master[df_master['type']==2]['coastal'], bins=range(0,50000,1000), alpha=0.5, color=c_ice, label='ice')

axs[0,1].hist(df_master[df_master['type']==0]['blue'], bins=range(0,50000,1000), alpha=0.5, color=c_snow)
axs[0,1].hist(df_master[df_master['type']==1]['blue'], bins=range(0,50000,1000), alpha=0.5, color=c_firn)
if show_ice: axs[0,1].hist(df_master[df_master['type']==2]['blue'], bins=range(0,50000,1000), alpha=0.5, color=c_ice)

axs[0,2].hist(df_master[df_master['type']==0]['green'], bins=range(0,50000,1000), alpha=0.5, color=c_snow)
axs[0,2].hist(df_master[df_master['type']==1]['green'], bins=range(0,50000,1000), alpha=0.5, color=c_firn)
if show_ice: axs[0,2].hist(df_master[df_master['type']==2]['green'], bins=range(0,50000,1000), alpha=0.5, color=c_ice)

axs[0,3].hist(df_master[df_master['type']==0]['red'], bins=range(0,50000,1000), alpha=0.5, color=c_snow)
axs[0,3].hist(df_master[df_master['type']==1]['red'], bins=range(0,50000,1000), alpha=0.5, color=c_firn)
if show_ice: axs[0,3].hist(df_master[df_master['type']==2]['red'], bins=range(0,50000,1000), alpha=0.5, color=c_ice)

axs[1,0].hist(df_master[df_master['type']==0]['nir'], bins=range(0,50000,1000), alpha=0.5, color=c_snow)
axs[1,0].hist(df_master[df_master['type']==1]['nir'], bins=range(0,50000,1000), alpha=0.5, color=c_firn)
if show_ice: axs[1,0].hist(df_master[df_master['type']==2]['nir'], bins=range(0,50000,1000), alpha=0.5, color=c_ice)

axs[1,1].hist(df_master[df_master['type']==0]['swir1'], bins=range(5000,13000,100), alpha=0.5, color=c_snow)
axs[1,1].hist(df_master[df_master['type']==1]['swir1'], bins=range(5000,13000,100), alpha=0.5, color=c_firn)
if show_ice: axs[1,1].hist(df_master[df_master['type']==2]['swir1'], bins=range(5000,13000,100), alpha=0.5, color=c_ice)

axs[1,2].hist(df_master[df_master['type']==0]['swir2'], bins=range(5000,13000,100), alpha=0.5, color=c_snow)
axs[1,2].hist(df_master[df_master['type']==1]['swir2'], bins=range(5000,13000,100), alpha=0.5, color=c_firn)
if show_ice: axs[1,2].hist(df_master[df_master['type']==2]['swir2'], bins=range(5000,13000,100), alpha=0.5, color=c_ice)
    
axs[1,3].hist(df_master[df_master['type']==0]['temp_c'], bins=np.arange(-10,10,0.1), alpha=0.5, color=c_snow)
axs[1,3].hist(df_master[df_master['type']==1]['temp_c'], bins=np.arange(-10,10,0.1), alpha=0.5, color=c_firn)
if show_ice: axs[1,3].hist(df_master[df_master['type']==2]['temp_c'], bins=np.arange(-10,10,0.1), alpha=0.5, color=c_ice)

axs[2,0].hist(df_master[df_master['type']==0]['ndwi'], bins=np.arange(-0.5,1,0.01), alpha=0.5, color=c_snow)
axs[2,0].hist(df_master[df_master['type']==1]['ndwi'], bins=np.arange(-0.5,1,0.01), alpha=0.5, color=c_firn)
if show_ice: axs[2,0].hist(df_master[df_master['type']==2]['ndwi'], bins=np.arange(-0.5,1,0.01), alpha=0.5, color=c_ice)

axs[2,1].hist(df_master[df_master['type']==0]['ndsi'], bins=np.arange(-0.5,1,0.01), alpha=0.5, color=c_snow)
axs[2,1].hist(df_master[df_master['type']==1]['ndsi'], bins=np.arange(-0.5,1,0.01), alpha=0.5, color=c_firn)
if show_ice: axs[2,1].hist(df_master[df_master['type']==2]['ndsi'], bins=np.arange(-0.5,1,0.01), alpha=0.5, color=c_ice)

axs[2,2].hist(df_master[df_master['type']==0]['ndwi_ndsi'], bins=np.arange(1,3,0.01), alpha=0.5, color=c_snow)
axs[2,2].hist(df_master[df_master['type']==1]['ndwi_ndsi'], bins=np.arange(1,3,0.01), alpha=0.5, color=c_firn)
if show_ice: axs[2,2].hist(df_master[df_master['type']==2]['ndwi_ndsi'], bins=np.arange(1,3,0.01), alpha=0.5, color=c_ice)

# axs[2,3].hist2d(df_master[df_master['type']==0]['ndwi_ndsi'], bins=np.arange(1,3,0.01), alpha=0.5, color=c_snow)
# axs[2,3].hist2d(df_master[df_master['type']==1]['ndwi_ndsi'], bins=np.arange(1,3,0.01), alpha=0.5, color=c_firn)
# if show_ice: axs[2,2].hist2d(df_master[df_master['type']==2]['ndwi_ndsi'], bins=np.arange(1,3,0.01), alpha=0.5, color=c_ice)
axs[2,3].hist2d(df_master['nir'],df_master['ndsi'], bins=100,cmap='plasma')

#axs[0][1].hist2d(tracks_2016[~np.isnan(bal_2016_masked)].flatten(), bal_2016_masked[~np.isnan(bal_2016_masked)].flatten(), bins=nbins, cmap=scatter_cmap, zorder=1, cmin=0, vmin=1, cmax=30, range=[[-0.25,6.25],[-0.25,6.25]])

titles = ['Coastal', 'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'Temp', 'NDWI', 'NDSI', 'NDWI*NDSI']
for c in range(len(titles)):
    axs[c//4,c%4].set_title(titles[c])

axs[0,0].legend()

fig.tight_layout()

#%%
def scat(x,y,ax):
    ax.scatter(df_master[df_master['type']==2][x],df_master[df_master['type']==2][y], s=0.01, alpha=1, c=c_ice, label='ice')
    ax.scatter(df_master[df_master['type']==1][x],df_master[df_master['type']==1][y], s=0.01, alpha=0.5, c=c_firn, label='firn')
    ax.scatter(df_master[df_master['type']==0][x],df_master[df_master['type']==0][y], s=0.01, alpha=0.3, c=c_snow, label='snow')
    ax.set_xlabel(x)
    ax.set_ylabel(y)

fig, axs = plt.subplots(3,3, figsize=(9,9))
scat('nir','ndsi',axs[0,0])
scat('nir','red',axs[0,1])
scat('ndwi','ndsi',axs[0,2])
scat('green','nir',axs[1,0])
scat('swir1','swir2',axs[1,1])
scat('nir','coastal',axs[1,2])
axs[0,0].legend()
plt.tight_layout()


#%%

# main bands
fig, axs = plt.subplots(5,7, figsize=(9,9), sharex='col', sharey='row')

for c in range(len(dfs)):
    axs[c,0].hist(dfs[c][dfs[c]['type']==0]['coastal'], bins=range(0,50000,1000), alpha=0.5, color=c_snow, label='snow')
    axs[c,0].hist(dfs[c][dfs[c]['type']==1]['coastal'], bins=range(0,50000,1000), alpha=0.5, color=c_firn, label='firn')
    if show_ice: axs[c,0].hist(dfs[c][dfs[c]['type']==2]['coastal'], bins=range(0,50000,1000), alpha=0.5, color=c_ice, label='ice')
    
    axs[c,1].hist(dfs[c][dfs[c]['type']==0]['blue'], bins=range(0,50000,1000), alpha=0.5, color=c_snow)
    axs[c,1].hist(dfs[c][dfs[c]['type']==1]['blue'], bins=range(0,50000,1000), alpha=0.5, color=c_firn)
    if show_ice: axs[c,1].hist(dfs[c][dfs[c]['type']==2]['blue'], bins=range(0,50000,1000), alpha=0.5, color=c_ice)
    
    axs[c,2].hist(dfs[c][dfs[c]['type']==0]['green'], bins=range(0,50000,1000), alpha=0.5, color=c_snow)
    axs[c,2].hist(dfs[c][dfs[c]['type']==1]['green'], bins=range(0,50000,1000), alpha=0.5, color=c_firn)
    if show_ice: axs[c,2].hist(dfs[c][dfs[c]['type']==2]['green'], bins=range(0,50000,1000), alpha=0.5, color=c_ice)
    
    axs[c,3].hist(dfs[c][dfs[c]['type']==0]['red'], bins=range(0,50000,1000), alpha=0.5, color=c_snow)
    axs[c,3].hist(dfs[c][dfs[c]['type']==1]['red'], bins=range(0,50000,1000), alpha=0.5, color=c_firn)
    if show_ice: axs[c,3].hist(dfs[c][dfs[c]['type']==2]['red'], bins=range(0,50000,1000), alpha=0.5, color=c_ice)
    
    axs[c,4].hist(dfs[c][dfs[c]['type']==0]['nir'], bins=range(0,50000,1000), alpha=0.5, color=c_snow)
    axs[c,4].hist(dfs[c][dfs[c]['type']==1]['nir'], bins=range(0,50000,1000), alpha=0.5, color=c_firn)
    if show_ice: axs[c,4].hist(dfs[c][dfs[c]['type']==2]['nir'], bins=range(0,50000,1000), alpha=0.5, color=c_ice)
    
    axs[c,5].hist(dfs[c][dfs[c]['type']==0]['swir1'], bins=range(5000,13000,100), alpha=0.5, color=c_snow)
    axs[c,5].hist(dfs[c][dfs[c]['type']==1]['swir1'], bins=range(5000,13000,100), alpha=0.5, color=c_firn)
    if show_ice: axs[c,5].hist(dfs[c][dfs[c]['type']==2]['swir1'], bins=range(5000,13000,100), alpha=0.5, color=c_ice)
    
    axs[c,6].hist(dfs[c][dfs[c]['type']==0]['swir2'], bins=range(5000,13000,100), alpha=0.5, color=c_snow)
    axs[c,6].hist(dfs[c][dfs[c]['type']==1]['swir2'], bins=range(5000,13000,100), alpha=0.5, color=c_firn)
    if show_ice: axs[c,6].hist(dfs[c][dfs[c]['type']==2]['swir2'], bins=range(5000,13000,100), alpha=0.5, color=c_ice)

cols = ['Coastal', 'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2',]
for ax, col in zip(axs[0], cols):
    ax.set_title(col)

fig.tight_layout()



#%%
# other bands
fig, axs = plt.subplots(5,5, figsize=(9,9), sharex='col', sharey='row')

c_ice = 'tab:blue'
c_firn = 'tab:brown'
c_snow = 'tab:olive'

for c in range(len(dfs)):
    
    axs[c,0].hist(dfs[c][dfs[c]['type']==0]['temp_c'], bins=np.arange(-10,10,0.1), alpha=0.5, color=c_snow, label='snow')
    axs[c,0].hist(dfs[c][dfs[c]['type']==1]['temp_c'], bins=np.arange(-10,10,0.1), alpha=0.5, color=c_firn, label='firn')
    if show_ice: axs[c,0].hist(dfs[c][dfs[c]['type']==2]['temp_c'], bins=np.arange(-10,10,0.1), alpha=0.5, color=c_ice, label='ice')
    
    axs[c,1].hist(dfs[c][dfs[c]['type']==0]['ndwi'], bins=np.arange(-0.5,1,0.01), alpha=0.5, color=c_snow)
    axs[c,1].hist(dfs[c][dfs[c]['type']==1]['ndwi'], bins=np.arange(-0.5,1,0.01), alpha=0.5, color=c_firn)
    if show_ice: axs[c,1].hist(dfs[c][dfs[c]['type']==2]['ndwi'], bins=np.arange(-0.5,1,0.01), alpha=0.5, color=c_ice)
    
    axs[c,2].hist(dfs[c][dfs[c]['type']==0]['ndsi'], bins=np.arange(-0.5,1,0.01), alpha=0.5, color=c_snow)
    axs[c,2].hist(dfs[c][dfs[c]['type']==1]['ndsi'], bins=np.arange(-0.5,1,0.01), alpha=0.5, color=c_firn)
    if show_ice: axs[c,2].hist(dfs[c][dfs[c]['type']==2]['ndsi'], bins=np.arange(-0.5,1,0.01), alpha=0.5, color=c_ice)
    
    axs[c,3].hist(dfs[c][dfs[c]['type']==0]['ndwi_ndsi'], bins=np.arange(1,3,0.01), alpha=0.5, color=c_snow)
    axs[c,3].hist(dfs[c][dfs[c]['type']==1]['ndwi_ndsi'], bins=np.arange(1,3,0.01), alpha=0.5, color=c_firn)
    if show_ice: axs[c,3].hist(dfs[c][dfs[c]['type']==2]['ndwi_ndsi'], bins=np.arange(1,3,0.01), alpha=0.5, color=c_ice)
    
    axs[c,4].hist(dfs[c][dfs[c]['type']==0]['ndwi_ndsi'], alpha=0.5, color=c_snow)
    axs[c,4].hist(dfs[c][dfs[c]['type']==1]['ndwi_ndsi'], alpha=0.5, color=c_firn)
    if show_ice: axs[c,4].hist(dfs[c][dfs[c]['type']==2]['ndwi_ndsi'], alpha=0.5, color=c_ice)
    
cols = ['Temp', 'NDWI', 'NDSI', 'NDWI*NDSI', 'open']
for ax, col in zip(axs[0], cols):
    ax.set_title(col)

axs[0,0].legend()

fig.tight_layout()

#%%

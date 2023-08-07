# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 15:55:10 2022

@author: lzell
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from datetime import datetime

# set folder/file path
agva_folder = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop','AGVA')
data_path = os.path.join(agva_folder, 'AA observations csv', 'PC', 'annual_minimums_temp.csv')
rgi_path = os.path.join(agva_folder, 'RGI', '01_rgi60_Alaska', '01_rgi60_Alaska.shp')
rgi_o3_path = os.path.join(agva_folder, 'RGI', '01_rgi_o3regions', 'RGI01_O3Regions.shp')
climate_folder = os.path.join(agva_folder, 'Climate')
os.chdir(agva_folder)

# open rgi shapefile and observed AA datafile
AAs_df = pd.read_csv(data_path)
rgi_gdf = gpd.read_file(rgi_path)#.set_index('RGIId')

# select only rows in rgi_gdf that are in AAs_df
rgi_gdf = rgi_gdf[rgi_gdf['RGIId'].isin(AAs_df['RGIId'])]

# merge the two dataframes
rgi_gdf = rgi_gdf.merge(AAs_df, on='RGIId')

# import rgio3
rgio3 = gpd.read_file(rgi_o3_path)

# add O3Region as column to rgi_gdf
rgi_gdf = gpd.sjoin(rgi_gdf, rgio3[['O3Region','geometry']]) 

# get list of o3regions
O3Regions = rgi_gdf['O3Region'].unique().tolist()
O3Regions.sort()

# background image
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
outlines_bad = world[world['name'].isin(['United States of America', 'Canada'])]
#states_gdf = gpd.read_file('http://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_5m.json')
states_gdf = gpd.read_file(os.path.join(agva_folder,'alaska_outline.json'))
outlines = states_gdf[states_gdf['NAME'].isin(['Alaska'])]

# reproject to better crs
rgi_gdf = rgi_gdf.to_crs(epsg=3338)
outlines = outlines.to_crs(epsg=3338)

# load climate df
climate_df = pd.read_csv(os.path.join(climate_folder,'per_glacier_climate_2013_2021.csv')).drop(['.geo',"system:index"],axis=1)

# load el nino data
ONI_index = pd.read_csv(os.path.join(climate_folder,'ONI_index.txt'), delimiter='\s+')
ONI_monthly_index = pd.read_csv(os.path.join(climate_folder,'ONI_monthly_index.txt'), delimiter='\s+')

# subset ONI to most recent years
ONI_index = ONI_index[ONI_index['YR']>2011]
ONI_monthly_index = ONI_monthly_index[ONI_monthly_index['YR']>2011]

#%%
# lets check to make sure the o3region worked
fig,ax = plt.subplots()
c=1
for n in O3Regions:
    subset = rgi_gdf[rgi_gdf['O3Region']==n]
    subset['c']=c
    subset.plot(ax=ax, column='c', cmap='prism', vmin=0, vmax=16)
    c+=1

#%%
# filter out obs where we had 0 observations
rgi_gdf = rgi_gdf.replace(-1,np.nan)
rgi_gdf = rgi_gdf.replace('-1',np.nan)

# take only rows that match climate and aar data
rgi_gdf = rgi_gdf[rgi_gdf['RGIId'].isin(climate_df['RGIId'])]

# merge climate data with rgi_gdf
rgi_gdf = rgi_gdf.merge(climate_df, on='RGIId')

# replace -99 with np.nan (-99 was used for masked climate data)
rgi_gdf = rgi_gdf.replace(-99,np.nan)
rgi_gdf = rgi_gdf.replace('-99',np.nan)

# do some further calculations
aar_cols = ['min_AAR_2013', 'min_AAR_2014','min_AAR_2015', 'min_AAR_2016', 'min_AAR_2017', 'min_AAR_2018', 'min_AAR_2019', 'min_AAR_2020', 'min_AAR_2021']
aa_cols = ['min_AA_2013', 'min_AA_2014','min_AA_2015', 'min_AA_2016', 'min_AA_2017', 'min_AA_2018', 'min_AA_2019', 'min_AA_2020', 'min_AA_2021']
    
# optionally, filter out glacier years where we had only 1 observation (n_obs)
def remove_1s(row, year):
    if row["n_obs_"+str(year)]==1:
        row['min_AAR_'+str(year)] = np.nan
        row['min_AA_'+str(year)] = np.nan
    return row

# for y in range(2013,2022):
#     rgi_gdf = rgi_gdf.apply(lambda row: remove_1s(row, y), axis=1)

rgi_gdf['mean_AAR'] = rgi_gdf[aar_cols].mean(axis=1)
rgi_gdf['mean_AA'] = rgi_gdf[aa_cols].mean(axis=1)

#%%

# now for each glacier, calculate the relationship between min_aar and each climate variable
def r_var(row, var_name):
    dat = []
    aars = []
    
    for y in range(2013,2022):
        a = row[['min_AAR_'+str(y)]].values[0]
        d = row[[str(var_name)+str(y)]].values[0]
        if ~np.isnan(a):
            aars.append(a)
            dat.append(d)
    
    # print(dat)
    # print(aars)
    
    if len(dat)>=4:
        corr = np.corrcoef([dat, aars])[0,1]
    else:
        corr = np.nan
    row['corr_'+str(var_name)] = corr
    
    return row

climate_names = ['wprecip', 'sprecip', 'aprecip', 'wtemp', 'stemp', 'atemp']

load = 1
if load:
    rgi_gdf = gpd.read_file("rgi_corr.shp")
else:
    for n in climate_names:
        print(n)
        rgi_gdf = rgi_gdf.apply(lambda row: r_var(row, n), axis=1)

#%%
save = 0
if save:
    rgi_gdf.to_file("rgi_corr.shp")

#%%
# make a figure showing the spatial distribution of r values
fig,axs = plt.subplots(2,3, figsize=(12,6), sharex=True, sharey=True)

n=0
names = climate_names
for i in range(6):
    
    r=n//3
    c=i%3
    ax = axs[r,c]
    name = names[i]
    print(r,c)
    n+=1
    
    outlines.boundary.plot(ax=ax, color='0.7', alpha=1, zorder=-1)
    d = rgi_gdf.plot(ax=ax, column='corr_'+name, alpha=1, legend=False, cmap='coolwarm', vmin=-1, vmax=1)# color=colors[o2-2])
    ax.set_title(name)

    #west chugach
    # ax.set_xlim(1.50e+5, 5.50e+5)
    # ax.set_ylim(1.05e+6, 1.35e+6)
    #main glaciers
    ax.set_xlim(-1.39e+5, 1.59e+6)
    ax.set_ylim(7.25e+5, 1.627e+6)

plt.tight_layout()

#%%
# average r value within each o3 region
fig,axs = plt.subplots(2,3, figsize=(12,6), sharex=True, sharey=True)
#corrs = np.zeros([5,6])
o2_names = ['Alaska Range', 'Alaska Peninsula', 'W. Chugach', 'St. Elias', 'North Coast']

n=0
names = climate_names
for i in range(6):
    
    r=n//3
    c=i%3
    ax = axs[r,c]
    name = names[i]
    print(r,c)
    n+=1
    
    outlines.boundary.plot(ax=ax, color='0.7', alpha=1, zorder=-1)
    ax.set_title(name)
    
    j=0
    for reg in O3Regions: #[2,3,4,5,6]
        subset = rgi_gdf[rgi_gdf['O3Region']==str(reg)].copy()
        col = subset['corr_'+name]
        avg = np.nanmedian(col)
        subset['color'] = avg
        #corrs[j,i]=avg
        j+=1
        subset.plot(ax=ax, column='color', alpha=1, legend=False, cmap='coolwarm', vmin=-1, vmax=1)
        
        #d = rgi_gdf.plot(ax=ax, column='corr_'+name, alpha=1, legend=False, cmap='coolwarm_r', vmin=-1, vmax=1)# color=colors[o2-2])
    

    #west chugach
    # ax.set_xlim(1.50e+5, 5.50e+5)
    # ax.set_ylim(1.05e+6, 1.35e+6)
    #main glaciers
    ax.set_xlim(-1.39e+5, 1.59e+6)
    ax.set_ylim(7.25e+5, 1.627e+6)

plt.tight_layout()

#%%
# average r value within each o2 region
fig,axs = plt.subplots(2,3, figsize=(12,6), sharex=True, sharey=True)
corrs = np.zeros([5,6])
o2_names = ['Alaska Range', 'Alaska Peninsula', 'W. Chugach', 'St. Elias', 'North Coast']

n=0
names = climate_names
for i in range(6):
    
    r=n//3
    c=i%3
    ax = axs[r,c]
    name = names[i]
    print(r,c)
    n+=1
    
    outlines.boundary.plot(ax=ax, color='0.7', alpha=1, zorder=-1)
    ax.set_title(name)
    
    j=0
    for reg in [2,3,4,5,6]:
        subset = rgi_gdf[rgi_gdf['O2Region']==str(reg)].copy()
        col = subset['corr_'+name]
        avg = np.nanmedian(col)
        subset['color'] = avg
        corrs[j,i]=avg
        j+=1
        subset.plot(ax=ax, column='color', alpha=1, legend=False, cmap='coolwarm', vmin=-1, vmax=1)
        
        #d = rgi_gdf.plot(ax=ax, column='corr_'+name, alpha=1, legend=False, cmap='coolwarm_r', vmin=-1, vmax=1)# color=colors[o2-2])
    

    #west chugach
    # ax.set_xlim(1.50e+5, 5.50e+5)
    # ax.set_ylim(1.05e+6, 1.35e+6)
    #main glaciers
    ax.set_xlim(-1.39e+5, 1.59e+6)
    ax.set_ylim(7.25e+5, 1.627e+6)

plt.tight_layout()

#%%
fig,ax = plt.subplots(figsize=(8,4))
im = ax.imshow(corrs, cmap='bwr', vmin=-1, vmax=1)
ax.set_xticks([0,1,2,3,4,5])
ax.set_xticklabels(climate_names, rotation = -45, ha="left")
ax.set_yticks([0,1,2,3,4])
ax.set_yticklabels(o2_names)

cbar = fig.colorbar(im, ax=ax, label='Correlation',  ticks=[-1, -0.5, 0, 0.5, 1])
#cbar.ax.set_yticklabels(['-5%', '0', '+5%']) 

for (j,i),label in np.ndenumerate(corrs):
    if abs(label)>0.3: weight='bold'
    else: weight='normal'
    ax.text(i, j, round(label,2), ha='center', va='center', weight=weight)
    
plt.tight_layout()

#%%

# make a figure showing the spatial distribution of r values
# fig,axs = plt.subplots(2,3, figsize=(12,6), sharex=True, sharey=True)

# n=0
# names = climate_names
# for i in range(6):
    
#     r=n//3
#     c=i%3
#     ax = axs[r,c]
#     name = names[i]
#     print(r,c)
#     n+=1
    
#     ax.scatter(rgi_gdf['corr_'+name], np.log(rgi_gdf['Area']), s=1, c='black')
#     ax.set_xlabel("Correlation")
#     ax.set_ylabel("Area")
#     ax.set_title(name)

# plt.tight_layout()







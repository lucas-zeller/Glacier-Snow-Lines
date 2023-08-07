# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:26:20 2022

@author: lzell
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from datetime import datetime
import matplotlib as mpl
import scipy.stats
#from mpl_toolkits.axes_grid1 import make_axes_locatable

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
new_outline = gpd.read_file(os.path.join(agva_folder,'us_outline_20m.json'))

# reproject to better crs
rgi_gdf = rgi_gdf.to_crs(epsg=3338)
outlines = outlines.to_crs(epsg=3338)

# load climate dfs
climate_2 = pd.read_csv(os.path.join(climate_folder,'02_climate_2013_2021.csv')).drop(['.geo',"system:index"],axis=1)
climate_3 = pd.read_csv(os.path.join(climate_folder,'03_climate_2013_2021.csv')).drop(['.geo',"system:index"],axis=1)
climate_4 = pd.read_csv(os.path.join(climate_folder,'04_climate_2013_2021.csv')).drop(['.geo',"system:index"],axis=1)
climate_5 = pd.read_csv(os.path.join(climate_folder,'05_climate_2013_2021.csv')).drop(['.geo',"system:index"],axis=1)
climate_6 = pd.read_csv(os.path.join(climate_folder,'06_climate_2013_2021.csv')).drop(['.geo',"system:index"],axis=1)
climate_list = [climate_2, climate_3, climate_4, climate_5, climate_6]

#O3 climate
climate_o3 = pd.read_csv(os.path.join(climate_folder,'O3_regions_climate_2013_2021.csv')).drop(['.geo',"system:index"],axis=1)

# load el nino data
ONI_index = pd.read_csv(os.path.join(climate_folder,'ONI_index.txt'), delimiter='\s+')
ONI_monthly_index = pd.read_csv(os.path.join(climate_folder,'ONI_monthly_index.txt'), delimiter='\s+')

# subset ONI to most recent years
ONI_index = ONI_index[ONI_index['YR']>2011]
ONI_monthly_index = ONI_monthly_index[ONI_monthly_index['YR']>2011]

#%%
# filter out obs where we had 0 observations
rgi_gdf = rgi_gdf.replace(-1,np.nan)
rgi_gdf = rgi_gdf.replace('-1',np.nan)

# do some further calculations
aar_cols = ['min_AAR_2013', 'min_AAR_2014','min_AAR_2015', 'min_AAR_2016', 'min_AAR_2017', 'min_AAR_2018', 'min_AAR_2019', 'min_AAR_2020', 'min_AAR_2021']
aa_cols = ['min_AA_2013', 'min_AA_2014','min_AA_2015', 'min_AA_2016', 'min_AA_2017', 'min_AA_2018', 'min_AA_2019', 'min_AA_2020', 'min_AA_2021']

# add doy as column
def get_doy(row, year):
    date = str(row['min_date_'+str(y)])
    if date == '-1': 
        doy = np.nan
    elif date == 'nan':
        doy = np.nan
    else:
        date = date[:10]
        date_format = datetime.strptime(date, '%Y-%m-%d')
        doy = date_format.timetuple().tm_yday
    return doy
    
for y in range(2013,2022):
    rgi_gdf['doy'+str(y)] = rgi_gdf.apply(lambda row: get_doy(row, y), axis=1)
    
# optionally, filter out glaicer years where we had only 1 observation (n_obs)
def remove_1s(row, year):
    if row["n_obs_"+str(year)]==1:
        row['min_AAR_'+str(year)] = np.nan
        row['min_AA_'+str(year)] = np.nan
    return row

# for y in range(2013,2022):
#     rgi_gdf = rgi_gdf.apply(lambda row: remove_1s(row, y), axis=1)

rgi_gdf['mean_AAR'] = rgi_gdf[aar_cols].mean(axis=1)
rgi_gdf['mean_AA'] = rgi_gdf[aa_cols].mean(axis=1)

# add doy as column
for y in range(2013,2022):
    rgi_gdf['doy'+str(y)] = rgi_gdf.apply(lambda row: get_doy(row, y), axis=1)
    

#%%
# make a dataframe where we can hold data for each o2 region

#%%
##### now experiment with making maps/figures
#%%
# annual variation from mean AAR for each glacier, each year

fig,axs = plt.subplots(3,3, figsize=(15,9), sharex=True, sharey=True)

# rgi_gdf['centroid_column'] = rgi_gdf.centroid
# rgi_gdf = rgi_gdf.set_geometry('centroid_column')

n=0
for y in range(2013,2022):

    r=n//3
    c=n%3
    print(y,r,c)
    ax = axs[r,c]
    
    color_col = rgi_gdf['min_AAR_'+str(y)]-rgi_gdf['mean_AAR']
    subset = rgi_gdf#[rgi_gdf['O2Region']=='2']
    color_col = subset['min_AAR_'+str(y)]-subset['mean_AAR']
    
    outlines.boundary.plot(ax=ax, color='0.7', alpha=1, zorder=-1)    
    subset.plot(ax=ax, column=color_col, alpha=0.9, legend=False, cmap='coolwarm_r', vmin=-20, vmax=20, legend_kwds={'label': "2015 AAR Variation"})
    
    ax.set_title(str(y)+"  ", loc='right', y=1.0, pad=-20, fontsize=18)
    ax.set_xlim(-1.39e+5, 1.59e+6)
    ax.set_ylim(7.25e+5, 1.627e+6)
    ax.set_xticks([])
    ax.set_yticks([])
    
    n+=1

plt.tight_layout()

#%%
### make a fiure to grab the legend form it
plt.figure(figsize=(5,4), dpi=300)
plt.scatter(color_col.values, color_col.values, c=color_col.values, cmap='coolwarm_r', vmin=-5, vmax=5, label='AAR Variation')
cbar = plt.colorbar(label='AAR Variation', extend='both', ticks=[-5, 0, 5])
cbar.ax.set_yticklabels(['-5%', '0', '+5%']) 

#%%
# Show o2 regions, o3 regions
fig,axs = plt.subplots(2, figsize=(12,12))

colors = ['red', 'orange', 'green', 'blue', 'purple']
n=0
for o2 in [2,3,4,5,6]:
    subset = rgi_gdf[rgi_gdf['O2Region']==str(o2)]
    subset.plot(ax=axs[0], color=colors[o2-2])
    
c=1
for n in O3Regions:
    subset = rgi_gdf[rgi_gdf['O3Region']==n].copy()
    subset['c']=c
    subset.plot(ax=axs[1], column='c', cmap='prism', vmin=0, vmax=16)
    c+=1

for ax in axs:
    outlines.boundary.plot(ax=ax, color='0.7', alpha=1, zorder=-1)
    ax.set_xlim(-1.39e+5, 1.59e+6)
    ax.set_ylim(7.25e+5, 1.627e+6)

plt.tight_layout()

#%%
#study area figure
fig,axs = plt.subplots(2, figsize=(12,12))
colors = ['red', 'orange', 'green', 'blue', 'purple']

for o2 in [2,3,4,5,6]:
    subset = rgi_gdf[rgi_gdf['O2Region']==str(o2)]
    subset.plot(ax=axs[1], color=colors[o2-2])

outlines.boundary.plot(ax=axs[1], color='0.7', alpha=1, zorder=-1)
axs[1].set_xlim(-1.39e+5, 1.59e+6)
axs[1].set_ylim(7.25e+5, 1.627e+6)

world.boundary.plot(ax=axs[0], color='0.7', alpha=1, zorder=-1)
axs[0].set_xlim(-175, -130)
axs[0].set_ylim(51, 70)


#%%
# calculate mean (or median) AAR variation per O2 region, in each year
fig,axs = plt.subplots(3,3, figsize=(15,9))

n=0
for y in range(2013,2022):
    #if n>0: continue
    r=n//3
    c=n%3
    print(y,r,c)
    ax = axs[r,c]

    outlines.boundary.plot(ax=ax, color='0.7', alpha=1, zorder=-1)
    
    for o2 in rgi_gdf['O2Region'].unique():
        subset = rgi_gdf[rgi_gdf['O2Region']==o2].copy()
        col = subset['min_AAR_'+str(y)]-subset['mean_AAR']
        avg = np.nanmean(col)
        subset['color'] = avg
        subset.plot(ax=ax, column='color', alpha=1, legend=False, cmap='coolwarm_r', vmin=-5, vmax=5)
        
    ax.set_title(str(y)+"  ", loc='right', y=1.0, pad=-20, fontsize=18)
    ax.set_xlim(-1.39e+5, 1.59e+6)
    ax.set_ylim(7.25e+5, 1.627e+6)
    ax.set_xticks([])
    ax.set_yticks([])
    n+=1
    
plt.tight_layout()

#%%
# calculate mean (or median) AAR variation per O3 region, in each year
fig,axs = plt.subplots(3,3, figsize=(15,9))

n=0
for y in range(2013,2022):
    #if n>0: continue
    r=n//3
    c=n%3
    print(y,r,c)
    ax = axs[r,c]

    outlines.boundary.plot(ax=ax, color='0.7', alpha=1, zorder=-1)
    
    for o in O3Regions:
        subset = rgi_gdf[rgi_gdf['O3Region']==o].copy()
        col = subset['min_AAR_'+str(y)]-subset['mean_AAR']
        avg = np.nanmean(col)
        subset['color'] = avg
        subset.plot(ax=ax, column='color', alpha=1, legend=False, cmap='coolwarm_r', vmin=-5, vmax=5)
        
    ax.set_title(str(y)+"  ", loc='right', y=1.0, pad=-20, fontsize=18)
    ax.set_xlim(-1.39e+5, 1.59e+6)
    ax.set_ylim(7.25e+5, 1.627e+6)
    ax.set_xticks([])
    ax.set_yticks([])
    n+=1
    
plt.tight_layout()

#%%
# Day of year of min AAR for each glacier
fig,axs = plt.subplots(3,3, figsize=(15,10))

n=0
for y in range(2013,2022):

    r=n//3
    c=n%3
    print(y,r,c)
    ax = axs[r,c]
    
    color_col = rgi_gdf['doy'+str(y)]
    
    outlines.boundary.plot(ax=ax, color='0.7', alpha=1, zorder=-1)
    rgi_gdf.plot(ax=ax, column=color_col, alpha=0.9, legend=False, cmap='coolwarm_r', vmin=200, vmax=330)
    
    ax.set_title(y, loc='right', y=1.0, pad=-14)
    ax.set_xlim(-1.39e+5, 1.59e+6)
    ax.set_ylim(7.25e+5, 1.627e+6)
    
    n+=1

plt.tight_layout()

#%%
# DOY per O2 region
fig,axs = plt.subplots(3,3, figsize=(15,10))

n=0
for y in range(2013,2022):

    r=n//3
    c=n%3
    print(y,r,c)
    ax = axs[r,c]

    outlines.boundary.plot(ax=ax, color='0.7', alpha=1, zorder=-1)
    
    for o2 in rgi_gdf['O2Region'].unique():
        subset = rgi_gdf[rgi_gdf['O2Region']==o2].copy()
        col = subset['doy'+str(y)]
        avg = np.nanmean(col)
        #print(avg)
        subset['color'] = avg
        subset.plot(ax=ax, column='color', alpha=1, legend=False, cmap='coolwarm_r', vmin=230, vmax=300)
        
    ax.set_title(y, loc='right', y=1.0, pad=-14)
    ax.set_xlim(-1.39e+5, 1.59e+6)
    ax.set_ylim(7.25e+5, 1.627e+6)
    n+=1
    
plt.tight_layout()


#%%
# scatterplots of AAR variation by latitude, longitude, etc...
y = 2019
fig,ax = plt.subplots(1)

x = np.log(rgi_gdf['Area'])
y = np.log(rgi_gdf['mean_AA'])

ax.scatter(x,y, c='black', s=0.1)
ax.set_xlabel('Glacier Area (log)')
ax.set_ylabel('AA (log)')

#%%
# temporal trends of AAR, broken up into O2 regions
fig,axs = plt.subplots(2,3, figsize=(10,5))

# calculate alaska-wide trend first
xs = [2013,2014,2015,2016,2017,2018,2019,2020,2021]
ys_all = []
for x in xs:
    y = np.mean(rgi_gdf['min_AAR_'+str(x)]-rgi_gdf['mean_AAR'])
    ys_all.append(y)

# then iterate through O2 regions
n=0
names = ['Alaska Range', 'Alaska Peninsula', 'W. Chugach', 'St. Elias', 'North Coast']
colors = ['red', 'orange', 'green', 'blue', 'purple']
for o2 in [2,3,4,5,6]:
    
    r=n//3
    c=n%3
    print(o2,r,c)
    ax = axs[r,c]
    o2_name = names[n]
    color = colors[n]
    n+=1
    
    # subset ot only that o2 region
    subset = rgi_gdf[rgi_gdf['O2Region']==str(o2)].copy()
    count = len(subset)
    
    # iterate through years
    ys_o2 = []
    for x in xs:
        y = np.mean(subset['min_AAR_'+str(x)]-subset['mean_AAR'])
        ys_o2.append(y)
    
    # add entire region and o2 region to correct subplot
    ax.hlines(y=0, xmin=2012, xmax=2022, linewidth=1, color='grey')
    ax.plot(xs, ys_all, c='black', linestyle='dashed')
    ax.plot(xs, ys_o2, c=color, linewidth=2)
    
    # format axis
    ax.set_ylim(-14,14)
    ax.set_xlim(2012.5, 2021.5)
    if r>0: ax.set_xlabel('Year', fontsize=12)
    if c==0: ax.set_ylabel('AAR variation', fontsize=12)
    ax.set_title(o2_name)
    ax.text(2019.25,-12.5, "n="+str(count))
    ax.set_yticks([-10,-5,0,5,10])
    ax.set_yticklabels(['-10%','-5%','0','+5%','+10%'])
    
    # plot everything together on the extra axis
    axs[1,2].plot(xs, ys_o2, c=color, linewidth=2)

axs[1,2].hlines(y=0, xmin=2014, xmax=2022, linewidth=1, color='grey')
axs[1,2].set_ylim(-14,14)
axs[1,2].set_xlim(2012.5, 2021.5)
axs[1,2].set_xlabel('Year', fontsize=12)
#axs[1,2].set_ylabel('AAR variation')
axs[1,2].set_title('All together')
axs[1,2].text(2019.25,-12, "n="+str(len(rgi_gdf)))
axs[1,2].set_yticks([-10,-5,0,5,10])
axs[1,2].set_yticklabels(['-10%','-5%','0','+5%','+10%'])

plt.tight_layout()

#%%
# temporal trends of AAR, broken up into O3 regions
fig,axs = plt.subplots(5,3, figsize=(10,13))

# calculate alaska-wide trend first
xs = [2013,2014,2015,2016,2017,2018,2019,2020,2021]
ys_all = []
for x in xs:
    y = np.mean(rgi_gdf['min_AAR_'+str(x)]-rgi_gdf['mean_AAR'])
    ys_all.append(y)

# plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.prism(np.linspace(0,1,16)))
# #cmap='prism', vmin=0, vmax=16
# norm = mpl.colors.Normalize(vmin=0,vmax=16)
# c_m = mpl.cm.prism
# s_m = mpl.cm.ScalarMappable(cmap=c_m, norm=norm)
# s_m.set_array([])

colors = iter(plt.cm.prism(np.linspace(0, 1, 17)))
col = next(colors) 
# then iterate through O3 regions
n=0
names = ['Alaska Range', 'Alaska Peninsula', 'W. Chugach', 'St. Elias', 'North Coast']
#colors = ['red', 'orange', 'green', 'blue', 'purple', 'red', 'orange', 'green', 'blue', 'purple', 'red', 'orange', 'green', 'blue', 'purple']
for o in O3Regions:
    
    r=n//3
    c=n%3
    print(o,r,c)
    ax = axs[r,c]
    o3_name = o
    #color = colors[n]
    n+=1
    
    # subset ot only that o region
    subset = rgi_gdf[rgi_gdf['O3Region']==str(o)].copy()
    count = len(subset)
    
    # iterate through years
    ys_o3 = []
    for x in xs:
        y = np.mean(subset['min_AAR_'+str(x)]-subset['mean_AAR'])
        ys_o3.append(y)
    
    col = next(colors)
    
    # add entire region and o2 region to correct subplot
    ax.hlines(y=0, xmin=2012, xmax=2022, linewidth=1, color='grey')
    ax.plot(xs, ys_all, c='black', linestyle='dashed')
    ax.plot(xs, ys_o3, linewidth=2, c=col)
    
    # format axis
    ax.set_ylim(-20,20)
    ax.set_xlim(2012.5, 2021.5)
    if r>0: ax.set_xlabel('Year', fontsize=12)
    if c==0: ax.set_ylabel('AAR variation', fontsize=12)
    ax.set_title(o3_name)
    ax.text(2019.25,-18.5, "n="+str(count))
    ax.set_yticks([-16,-8,0,8,16])
    ax.set_yticklabels(['-16%','-8%','0','+8%','+16%'])
    
    # plot everything together on the extra axis
    #axs[1,2].plot(xs, ys_o2, c=color, linewidth=2)

# axs[1,2].hlines(y=0, xmin=2014, xmax=2022, linewidth=1, color='grey')
# axs[1,2].set_ylim(-14,14)
# axs[1,2].set_xlim(2012.5, 2021.5)
# axs[1,2].set_xlabel('Year', fontsize=12)
# #axs[1,2].set_ylabel('AAR variation')
# axs[1,2].set_title('All together')
# axs[1,2].text(2019.25,-12, "n="+str(len(rgi_gdf)))
# axs[1,2].set_yticks([-10,-5,0,5,10])
# axs[1,2].set_yticklabels(['-10%','-5%','0','+5%','+10%'])

plt.tight_layout()


#%%
### comparison of regional remotely-sensed AAR vs in situ ela (O2 regions)
wolv_ela = np.array([1327, 1313, 1237, 1193, 1270, 1369, 1259, 1336, 1237])
gulk_ela = np.array([1923, 1730, 1970, 1958, 2188, 1811, 1977, 1792, 1805])
taku_ela = np.array([1148, 1187, 1217, 1168, 1174, 1308, 1528, 1086, 1005])
lecr_ela = np.array([1365, 1268, 1229, 1487, 1881, 1759, 2023, 1180, 1090])

# temporal trends of AAR, broken up into O2 regions
fig,axs = plt.subplots(4,2, figsize=(10,13))

# calculate alaska-wide trend first
xs = [2013,2014,2015,2016,2017,2018,2019,2020,2021]

ys_all = []
for x in xs:
    y = np.mean(rgi_gdf['min_AAR_'+str(x)]-rgi_gdf['mean_AAR'])
    ys_all.append(y)
    
# calculate variations for each glacier and their respective region
glaciers = [wolv_ela, gulk_ela, lecr_ela, taku_ela]
regions = ['4','2','6','6']
names = ['West Chugach - Wolverine','Alaska Range - Gulkana','North Coast - Lemon Creek', 'North Coast - Taku']
colors = ['green', 'red', 'purple', 'purple']
ylims = [(100,-100),(400,-400),(600,-600),(400,-400)]
for c in [0,1,2,3]:
    ax = axs[c,0]
    axb = axs[c,1]
    
    # subset to the region
    subset = rgi_gdf[rgi_gdf['O2Region']==regions[c]].copy()
    
    # iterate through years
    ys_o2 = []
    for x in xs:
        y = np.mean(subset['min_AAR_'+str(x)]-subset['mean_AAR'])
        ys_o2.append(y)
        
    # add entire region and o2 region to correct subplot
    ax.hlines(y=0, xmin=2012, xmax=2022, linewidth=1, color='grey')
    ax.plot(xs, ys_o2, c='black', linestyle='dashed', label='Region')
    
    # format axis
    ax.set_ylim(-12,12)
    ax.set_xlim(2012.5, 2021.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Remote - AAR variation')
    ax.set_title(names[c])
    
    # scatterplot of in-situ vs remote
    axb.scatter(ys_o2,glaciers[c]-np.mean(glaciers[c]), c='black')
    
    # annotate with the years of observation
    # for i, txt in enumerate(xs):
        
    #     # check to make sure we aren't trying to annotate outside of the axis limits
    #     if (float(ys_o2[i])>=ylims[c][1]) and (float(ys_o2[i])<=ylims[c][0]):
    #         axb.annotate(str(txt)[2:], (ys_o2[i], glaciers[c][i]-np.mean(glaciers[c])))
    
    # format axis
    axb.set_xlim(-12,12)
    axb.set_ylim(ylims[c])
    axb.set_xlabel('Remote - AAR Variation')
    axb.set_ylabel('In situ - ELA Variation')
    axb.set_title(names[c])
    
    # twinx for in-situ comparison
    ax2 = ax.twinx()
    ax2.plot(xs, glaciers[c]-np.mean(glaciers[c]), c=colors[c], label='In situ')
    
    ax2.set_ylim(ylims[c])
    ax2.set_ylabel('In situ - ELA variation')
    
    ax.legend(loc=3)
    ax2.legend(loc=4)

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.4)

#%%
### comparison of regional remotely-sensed AAR vs in situ ela (O3 regions)
wolv_ela = np.array([1327, 1313, 1237, 1193, 1270, 1369, 1259, 1336, 1237])
gulk_ela = np.array([1923, 1730, 1970, 1958, 2188, 1811, 1977, 1792, 1805])
taku_ela = np.array([1148, 1187, 1217, 1168, 1174, 1308, 1528, 1086, 1005])
lecr_ela = np.array([1365, 1268, 1229, 1487, 1881, 1759, 2023, 1180, 1090])

# temporal trends of AAR, broken up into O2 regions
fig,axs = plt.subplots(4,2, figsize=(10,13))

# calculate alaska-wide trend first
xs = [2013,2014,2015,2016,2017,2018,2019,2020,2021]

ys_all = []
for x in xs:
    y = np.mean(rgi_gdf['min_AAR_'+str(x)]-rgi_gdf['mean_AAR'])
    ys_all.append(y)
    
# calculate variations for each glacier and their respective region
glaciers = [wolv_ela, gulk_ela, lecr_ela, taku_ela]
regions = ['04.01','02.04','06.02','06.02']
names = ['West Chugach - Wolverine','Alaska Range - Gulkana','North Coast - Lemon Creek', 'North Coast - Taku']
colors = ['green', 'red', 'purple', 'purple']
ylims = [(100,-100),(400,-400),(600,-600),(400,-400)]
for c in [0,1,2,3]:
    ax = axs[c,0]
    axb = axs[c,1]
    
    # subset to the region
    subset = rgi_gdf[rgi_gdf['O3Region']==regions[c]].copy()
    
    # iterate through years
    ys_o2 = []
    for x in xs:
        y = np.mean(subset['min_AAR_'+str(x)]-subset['mean_AAR'])
        ys_o2.append(y)
        
    # add entire region and o2 region to correct subplot
    ax.hlines(y=0, xmin=2012, xmax=2022, linewidth=1, color='grey')
    ax.plot(xs, ys_o2, c='black', linestyle='dashed', label='Region')
    
    # format axis
    ax.set_ylim(-12,12)
    ax.set_xlim(2012.5, 2021.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Remote - AAR variation')
    ax.set_title(names[c])
    
    # scatterplot of in-situ vs remote
    axb.scatter(ys_o2,glaciers[c]-np.mean(glaciers[c]), c='black')
    
    # annotate with the years of observation
    for i, txt in enumerate(xs):
        
        # check to make sure we aren't trying to annotate outside of the axis limits
        if (float(ys_o2[i])>=ylims[c][1]) and (float(ys_o2[i])<=ylims[c][0]):
            axb.annotate(str(txt)[2:], (ys_o2[i], glaciers[c][i]-np.mean(glaciers[c])))
    
    # format axis
    axb.set_xlim(-12,12)
    axb.set_ylim(ylims[c])
    axb.set_xlabel('Remote - AAR Variation')
    axb.set_ylabel('In situ - ELA Variation')
    axb.set_title(names[c])
    
    # twinx for in-situ comparison
    ax2 = ax.twinx()
    ax2.plot(xs, glaciers[c]-np.mean(glaciers[c]), c=colors[c], label='In situ')
    
    ax2.set_ylim(ylims[c])
    ax2.set_ylabel('In situ - ELA variation')
    
    ax.legend(loc=3)
    ax2.legend(loc=4)

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.4)


#%%
### comparison of remotely-sensed AAR of single glacier vs in situ ela
fig,axs = plt.subplots(4,2, figsize=(8,10))

# calculate alaska-wide trend first
xs = [2013,2014,2015,2016,2017,2018,2019,2020,2021]

ys_all = []
for x in xs:
    y = np.mean(rgi_gdf['min_AAR_'+str(x)]-rgi_gdf['mean_AAR'])
    ys_all.append(y)
    
# calculate variations for each glacier and their respective region
glaciers = [wolv_ela, gulk_ela, lecr_ela, taku_ela]
regions = ['4','2','6','6']
names = ['Wolverine Glacier','Gulkana Glacier','Lemon Creek Glacier', 'Taku Glacier']
colors = ['green', 'red', 'purple','purple']
ylimsa = [(-20,20),(-12,12),(-50,50),(-30,30)]
ylimsb = [(100,-100),(400,-400),(600,-600),(400,-400)]
for c in [0,1,2,3]:
    axa = axs[c,0]
    axb = axs[c,1]
    
    # subset to the glacier
    subset = rgi_gdf[rgi_gdf['Name']==names[c]].copy()
    
    # iterate through years
    ys_glac = []
    for x in xs:
        y = (subset['min_AAR_'+str(x)]-subset['mean_AAR']).values[0]
        ys_glac.append(y)
    #print(names[c],ys_glac)
    ys_glac = np.array(ys_glac)
        
    # add remotely sensed AAR plot
    axa.hlines(y=0, xmin=2012, xmax=2022, linewidth=1, color='grey')
    axa.scatter(xs, ys_glac, c='black', label='Remote')
    #axa.plot(xs, ys_glac, c='black', linestyle='dashed', label='Remote')
    
    # print correlation for each glacier
    corr = np.corrcoef([ys_glac[~np.isnan(ys_glac)], glaciers[c][~np.isnan(ys_glac)]-np.mean(glaciers[c])])[0,1]
    print(names[c], corr)
    
    # format axis
    axa.set_ylim(ylimsa[c]) #ax.set_ylim(-12,12)
    axa.set_xlim(2012.5, 2021.5)
    axa.set_xlabel('Year')
    axa.set_ylabel('Remote: AAR variation (%)')
    axa.set_title(names[c])
    
    # scatterplot of in-situ vs remote
    axb.scatter(ys_glac, glaciers[c]-np.mean(glaciers[c]), c='black')
    
    # annotate with the years of observation
    # for i, txt in enumerate(xs):
        
    #     # check to make sure we aren't trying to annotate outside of the axis limits
    #     if (float(ys_glac[i])>=ylimsa[c][0]) and (float(ys_glac[i])<=ylimsa[c][1]):
    #         axb.annotate(str(txt)[2:], (ys_glac[i], glaciers[c][i]-np.mean(glaciers[c])))
    
    # format axis
    axb.set_xlim(ylimsa[c])  #axb.set_xlim(-12,12)
    axb.set_ylim(ylimsb[c])
    axb.set_xlabel('Remote: AAR Variation (%)')
    axb.set_ylabel('In situ: ELA Variation (m)')
    axb.set_title(names[c])
    
    # twinx for in situ comparison
    ax2 = axa.twinx()
    ax2.plot(xs, glaciers[c]-np.mean(glaciers[c]), c=colors[c], label='In Situ')
    
    ax2.set_ylim(ylimsb[c])
    ax2.set_ylabel('In situ: ELA variation (m)')
    
    axa.legend(loc=3)
    ax2.legend(loc=2)

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)


#%%
### comparison of remotely-sensed AAR of single glacier vs remotely-sensed regional AAR
fig,axs = plt.subplots(4,2, figsize=(10,13))

# years
xs = [2013,2014,2015,2016,2017,2018,2019,2020,2021]
    
# calculate variations for each glacier and their respective region
glaciers = [wolv_ela, gulk_ela, lecr_ela, taku_ela]
regions = ['4','2','6','6']
names = ['Wolverine Glacier','Gulkana Glacier','Lemon Creek Glacier', 'Taku Glacier']
colors = ['green', 'red', 'purple','purple']
ylimsa = [(-20,20),(-12,12),(-50,50),(-30,30)]
ylimsb = [(-20,20),(-12,12),(-20,20),(-20,20)]
for c in [0,1,2,3]:
    axa = axs[c,0]
    axb = axs[c,1]
    
    # subset to the glacier
    subset_g = rgi_gdf[rgi_gdf['Name']==names[c]].copy()
    
    # iterate through years
    ys_glac = []
    for x in xs:
        y = subset_g['min_AAR_'+str(x)]-subset_g['mean_AAR']
        ys_glac.append(y)
    
    # subset to the region
    subset_r = rgi_gdf[rgi_gdf['O2Region']==regions[c]].copy()
    
    # iterate through years
    ys_region = []
    for x in xs:
        y = np.nanmean(subset_r['min_AAR_'+str(x)]-subset_r['mean_AAR'])
        ys_region.append(y)
        
    # add glacier-specific AAR plot
    axa.hlines(y=0, xmin=2012, xmax=2022, linewidth=1, color='grey')
    axa.scatter(xs, ys_glac, c='black', label='Single Glacier')
    #axa.plot(xs, ys_glac, c='black', linestyle='dashed', label='Single Glacier')
    
    # add regional AAR
    ax2 = axa.twinx()
    ax2.plot(xs, ys_region, c=colors[c], label='Region')
    
    # scatterplot of in-situ vs remote
    axb.scatter(ys_glac, ys_region, c='black')
    
    # annotate with the years of observation
    for i, txt in enumerate(xs):
        
        # check to make sure we aren't trying to annotate outside of the axis limits
        if (float(ys_glac[i])>=ylimsa[c][0]) and (float(ys_glac[i])<=ylimsa[c][1]):
            axb.annotate(str(txt)[2:], (ys_glac[i], ys_region[i]))
            
    # format axis
    axa.set_ylim(ylimsa[c])
    axa.set_xlim(2012.5, 2021.5)
    axa.set_xlabel('Year')
    axa.set_ylabel('Single Glacier - AAR variation')
    axa.set_title(names[c])
    
    # format axis
    ax2.set_ylim(ylimsb[c])
    ax2.set_ylabel('Region - AAR variation')
    axa.legend(loc=3)
    ax2.legend(loc=4)
    
    # format axis
    axb.set_xlim(ylimsa[c])
    axb.set_ylim(ylimsb[c])
    axb.set_xlabel('Single Glacier - AAR variation')
    axb.set_ylabel('Region - AAR variation')
    axb.set_title(names[c])
    
plt.tight_layout()

#%%
# make a figure showing relationship between climate variables and AAR variation in each o2 region
yrs = [2013,2014,2015,2016,2017,2018,2019,2020,2021]
climate_names = ['wprecip', 'sprecip', 'aprecip', 'wtemp', 'stemp', 'atemp', 'wSST', 'sSST', 'aSST']
label_names = ['W. precip (m)', 'S. precip (m)', 'A. precip (m)', 'W. temp (C)', 'S. temp (C)', 'A. temp (C)', 'W. SST (C)', 'S. SST (C)', 'A. SST (C)']
o2_names = ['Alaska Range', 'Alaska Peninsula', 'W. Chugach', 'St. Elias', 'North Coast']
colors = ['red', 'orange', 'green', 'blue', 'purple']

fig,axs = plt.subplots(9,5, figsize=(7.5,10))

corrs = np.zeros([9,6])
ps = corrs.copy()

c = 0
for o2 in [2,3,4,5,6]:
    r=0
    subset = rgi_gdf[rgi_gdf['O2Region']==str(o2)].copy()
    climate_df = climate_list[int(o2)-2]
    
    # get annual aar variation data
    aar_data = []
    percent_obs = []
    for y in yrs:
        aar_data.append(np.nanmean(subset['min_AAR_'+str(y)]-subset['mean_AAR']))
        obs = len(subset[~np.isnan(subset['min_AAR_'+str(y)])])
        total = len(subset)
        percent_obs.append(obs/total)
    
    # iterate through climate variables, plotting each in their own subplot
    for cn in climate_names:
        ax = axs[r,c]
        climate_data = climate_df[cn]
        n = len(aar_data)
         
        # calculate correlation coefficients, p-value        
        corr = np.corrcoef([climate_data, aar_data])[0,1]
        corr_a = abs(corr)
        t = corr_a*np.sqrt((n-2))/np.sqrt((1-corr_a*corr_a))
        p = (1-scipy.stats.t.cdf(t, n-2))*2
        
        corrs[r,c] = corr
        ps[r,c] = p
        
        # scatter plot
        #ax.scatter(climate_data, aar_data, c=np.sqrt(np.array(percent_obs)), s=50, vmin=0.25, vmax=1, cmap='coolwarm')
        #ax.scatter(climate_data, aar_data, c=np.full(len(aar_data),corr), s=50, vmin=-1, vmax=1, cmap='coolwarm')
        ax.scatter(climate_data, aar_data, c=colors[c], s=25, vmin=-1, vmax=1, cmap='coolwarm') 
        
        # set number of axis ticks
        #plt.locator_params(axis='y', nbins=6)
        ax.xaxis.set_major_locator(plt.MaxNLocator(2))
        
        # set top row titles names
        if r==0: ax.set_title(o2_names[c])
        
        # set lefthand y-axis labels
        if c==4: 
            axs[r,4].set_ylabel(label_names[r], rotation=270, labelpad=15, fontsize=9)
            axs[r,4].yaxis.set_label_position("right")
        
        # annotate points
        # for i, txt in enumerate(yrs):
        #     ax.annotate(str(txt)[2:], (climate_data[i], aar_data[i]))
        
        
        
        r+=1
    
    
    #axs[r,0].set_ylabel('AAR Variation')
    
    c+=1
axs[4,0].set_ylabel('AAR Variation. (%)', fontsize=12)
axs[-1,2].set_xlabel('Climate Parameter', fontsize=12)
    
corrs[:,-1] = np.mean(corrs, axis=1)
ps[:,-1] = 1

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)

#%%
ynames = o2_names.copy()
ynames.append('Average')

fig,ax = plt.subplots(figsize=(5,5))
im = ax.imshow(corrs, cmap='PiYG', vmin=-1, vmax=1)
ax.set_yticks([0,1,2,3,4,5,6,7,8])
ax.set_yticklabels(label_names)
ax.set_xticks([0,1,2,3,4,5])
ax.set_xticklabels(ynames, rotation = -45, ha="left")

cbar = fig.colorbar(im, ax=ax, label='Correlation',  ticks=[-1, -0.5, 0, 0.5, 1])
#cbar.ax.set_yticklabels(['-5%', '0', '+5%']) 

for (j,i),label in np.ndenumerate(corrs):
    #if abs(label)>0.3: weight='bold'
    if abs(ps[j,i])<0.10: weight='bold'
    else: weight='normal'
    ax.text(i, j, round(label,2), ha='center', va='center', weight=weight, size=9)
    
plt.tight_layout()


fig,ax = plt.subplots(figsize=(5,5))
im = ax.imshow(ps, cmap='Reds_r', vmin=-0.5, vmax=1)
ax.set_yticks([0,1,2,3,4,5,6,7,8])
ax.set_yticklabels(label_names)
ax.set_xticks([0,1,2,3,4,5])
ax.set_xticklabels(ynames, rotation = -45, ha="left")

cbar = fig.colorbar(im, ax=ax, label='P-Value')#,  ticks=[-1, -0.5, 0, 0.5, 1])
#cbar.ax.set_yticklabels(['-5%', '0', '+5%']) 

for (j,i),label in np.ndenumerate(ps):
    if abs(label)<0.1: weight='bold'
    else: weight='normal'
    ax.text(i, j, round(label,2), ha='center', va='center', weight=weight, size=9)
    
plt.tight_layout()

#%%
# make a figure showing relationship between climate variables and AAR variation in each o2 region
yrs = [2013,2014,2015,2016,2017,2018,2019,2020,2021]
climate_names = ['wprecip', 'sprecip', 'aprecip', 'wtemp', 'stemp', 'atemp', 'wSST', 'sSST', 'aSST']
label_names = ['W. precip (m)', 'S. precip (m)', 'A. precip (m)', 'W. temp (C)', 'S. temp (C)', 'A. temp (C)', 'W. SST (C)', 'S. SST (C)', 'A. SST (C)']
# o2_names = ['Alaska Range', 'Alaska Peninsula', 'W. Chugach', 'St. Elias', 'North Coast']
# colors = ['red', 'orange', 'green', 'blue', 'purple']

fig,axs = plt.subplots(15,9, figsize=(13,20))

corrs = np.zeros([16,9])

r = 0
for o in O3Regions:
    c=0
    subset = rgi_gdf[rgi_gdf['O3Region']==str(o)].copy()
    climate_df = climate_o3[climate_o3['O3Region']==float(o)].copy()
    
    # get annual aar variation data
    aar_data = []
    percent_obs = []
    for y in yrs:
        aar_data.append(np.nanmean(subset['min_AAR_'+str(y)]-subset['mean_AAR']))
        obs = len(subset[~np.isnan(subset['min_AAR_'+str(y)])])
        total = len(subset)
        percent_obs.append(obs/total)
    
    # iterate through climate variables, plotting each in their own subplot
    for cn in climate_names:
        ax = axs[r,c]
        climate_data = []
        for y in yrs:
            climate_data.append(climate_df[cn+str(y)].values[0])
        #climate_data = climate_df[cn]
        
        # remove nan indexes
        climate_data = np.array(climate_data)[~np.isnan(np.array(aar_data))]
        aar_data_c = np.array(aar_data)[~np.isnan(np.array(aar_data))]
        
        # calculate correlation coefficients
        corr = np.corrcoef([climate_data, aar_data_c])[0,1]
        corrs[r,c] = corr
        
        # scatter plot
        #ax.scatter(climate_data, aar_data, c=np.sqrt(np.array(percent_obs)), s=50, vmin=0.25, vmax=1, cmap='coolwarm')
        #ax.scatter(climate_data, aar_data, c=np.full(len(aar_data),corr), s=50, vmin=-1, vmax=1, cmap='coolwarm')
        ax.scatter(climate_data, aar_data_c, c='black', s=25, vmin=-1, vmax=1, cmap='coolwarm') 
        
        # set number of axis ticks
        #plt.locator_params(axis='y', nbins=6)
        ax.xaxis.set_major_locator(plt.MaxNLocator(2))
        
        # set bottom row x-axis names
        if r==14: ax.set_xlabel(label_names[c])
        
        # annotate points
        # for i, txt in enumerate(yrs):
        #     ax.annotate(str(txt)[2:], (climate_data[i], aar_data[i]))
        
        
        
        c+=1
    
    # set lefthand y-axis labels
    axs[r,0].set_ylabel(O3Regions[r])
    #axs[r,0].set_ylabel('AAR Variation')
    
    r+=1
corrs[-1,:] = np.nanmean(corrs, axis=0)

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)

#%%
# make a figure showing relationship between climate variables and AAR variation in each o2 region
yrs = [2013,2014,2015,2016,2017,2018,2019,2020,2021]
climate_names = ['wprecip', 'sprecip', 'aprecip', 'wtemp', 'stemp', 'atemp', 'wSST', 'sSST', 'aSST']
label_names = ['W. precip (m)', 'S. precip (m)', 'A. precip (m)', 'W. temp (C)', 'S. temp (C)', 'A. temp (C)', 'W. SST (C)', 'S. SST (C)', 'A. SST (C)']
# o2_names = ['Alaska Range', 'Alaska Peninsula', 'W. Chugach', 'St. Elias', 'North Coast']
# colors = ['red', 'orange', 'green', 'blue', 'purple']

fig,axs = plt.subplots(9,15, figsize=(22,13))

corrs = np.zeros([9,16])

r = 0
for o in O3Regions:
    c=0
    subset = rgi_gdf[rgi_gdf['O3Region']==str(o)].copy()
    climate_df = climate_o3[climate_o3['O3Region']==float(o)].copy()
    
    # get annual aar variation data
    aar_data = []
    percent_obs = []
    for y in yrs:
        aar_data.append(np.nanmean(subset['min_AAR_'+str(y)]-subset['mean_AAR']))
        obs = len(subset[~np.isnan(subset['min_AAR_'+str(y)])])
        total = len(subset)
        percent_obs.append(obs/total)
    
    # iterate through climate variables, plotting each in their own subplot
    for cn in climate_names:
        ax = axs[c,r]
        climate_data = []
        for y in yrs:
            climate_data.append(climate_df[cn+str(y)].values[0])
        #climate_data = climate_df[cn]
        
        # remove nan indexes
        climate_data = np.array(climate_data)[~np.isnan(np.array(aar_data))]
        aar_data_c = np.array(aar_data)[~np.isnan(np.array(aar_data))]
        
        # calculate correlation coefficients
        corr = np.corrcoef([climate_data, aar_data_c])[0,1]
        corrs[c,r] = corr
        
        # scatter plot
        #ax.scatter(climate_data, aar_data, c=np.sqrt(np.array(percent_obs)), s=50, vmin=0.25, vmax=1, cmap='coolwarm')
        #ax.scatter(climate_data, aar_data, c=np.full(len(aar_data),corr), s=50, vmin=-1, vmax=1, cmap='coolwarm')
        ax.scatter(climate_data, aar_data_c, c='black', s=25, vmin=-1, vmax=1, cmap='coolwarm') 
        
        # set number of axis ticks
        #plt.locator_params(axis='y', nbins=6)
        ax.xaxis.set_major_locator(plt.MaxNLocator(2))
        
        # set bottom row x-axis names
        if r==0: ax.set_ylabel(label_names[c])
        
        # annotate points
        # for i, txt in enumerate(yrs):
        #     ax.annotate(str(txt)[2:], (climate_data[i], aar_data[i]))
        
        c+=1
    
    # set bottom x-axis labels
    axs[8,r].set_xlabel(o)
    
    r+=1
corrs[:,-1] = np.nanmean(corrs, axis=1)

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)

#%%
ynames = O3Regions.copy()
ynames.append('Average')

fig,ax = plt.subplots(figsize=(13,6))
im = ax.imshow(corrs, cmap='bwr', vmin=-1, vmax=1)
ax.set_yticks([0,1,2,3,4,5,6,7,8])
ax.set_yticklabels(label_names)#, rotation = 45, va="top")
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
ax.set_xticklabels(ynames)

cbar = fig.colorbar(im, ax=ax, label='Correlation',  ticks=[-1, -0.5, 0, 0.5, 1])
#cbar.ax.set_yticklabels(['-5%', '0', '+5%']) 

for (j,i),label in np.ndenumerate(corrs):
    if abs(label)>0.3: weight='bold'
    else: weight='normal'
    ax.text(i, j, round(label,2), ha='center', va='center', weight=weight)
    
plt.tight_layout()

#%%
# comparison on ENSO data to remotely sensed AARs

# calculate alaska-wide trend first
xs = [2013,2014,2015,2016,2017,2018,2019,2020,2021]
ys_all = []
for x in xs:
    y = np.mean(rgi_gdf['min_AAR_'+str(x)]-rgi_gdf['mean_AAR'])
    ys_all.append(y)
    
# calculate annual ONI anomaly
onis_all = []
for x in xs:
    oni_subset1 = ONI_monthly_index[(ONI_monthly_index['YR']==x) & (ONI_monthly_index['MON']<=9)]
    oni_subset2 = ONI_monthly_index[(ONI_monthly_index['YR']==x-1) & (ONI_monthly_index['MON']>=10)]
    o = np.mean( np.append(oni_subset1['ANOM'].values, oni_subset2['ANOM'].values) )
    onis_all.append(o)

# then iterate through O2 regions
n=0
names = ['Alaska Range', 'Alaska Peninsula', 'W. Chugach', 'St. Elias', 'North Coast', 'All Regions']
colors = ['red', 'orange', 'green', 'blue', 'purple', 'grey']
seasons = ['SON', 'OND', 'NDJ', 'DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'annual']
#seasons = ['SON', 'OND', 'NDJ', 'DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ', 'DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'annual']
seasons = ['MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ', 'DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'annual']

corrs = np.zeros([6,13])

fig,axs = plt.subplots(6,13, figsize=(18,10), sharey='row')

# first iterate through each subregion
for o2 in [2,3,4,5,6,7]:
    r = o2-2
    

    o2_name = names[r]
    color = colors[r]
    n+=1
    
    # subset ot only that o2 region
    if r==5:
        subset = rgi_gdf.copy()
    else:
        subset = rgi_gdf[rgi_gdf['O2Region']==str(o2)].copy()
    count = len(subset)
    
    # iterate through years
    ys_o2 = []
    for x in xs:
        y = np.mean(subset['min_AAR_'+str(x)]-subset['mean_AAR'])
        ys_o2.append(y)
    
    # then iterate through each of the 12 seasons
    for c in range(len(seasons)):
        ax = axs[r,c]
        
        # extract the ONI anomaly for that season
        onis_s = []
        
        if c<12:
            for x in xs:
                if c<6: x-=1 # for the first few seasons (Oct, Nov, Dec), we want to look at previous year
                o = ONI_index[(ONI_index['YR']==x) & (ONI_index['SEAS']==seasons[c])]
                o = np.mean(o['ANOM'])
                onis_s.append(o)
                
        else:
            onis_s = onis_all
            color = 'grey'

        # add points to plot
        ax.scatter(ys_o2, onis_s, c=color, s=50)
        
        # annotate points
        for i, txt in enumerate(xs):
            ax.annotate(str(txt)[2:], (ys_o2[i], onis_s[i]))
        
        # calculate correlation coefficients
        corr = np.corrcoef([ys_o2, onis_s])[0,1]
        corrs[r,c] = corr
        
        axs[5,c].set_xlabel('AAR Variation')
        axs[0,c].set_title(seasons[c])
        
        
    
        
        # ax.set_title(o2_name+' n='+str(count))
    
    # format axis
        
    axs[r,0].set_ylabel(names[r])


# axs[1,2].scatter(ys_all, onis_all, c='black', s=50)
# axs[1,2].set_xlabel('AAR Variation')
# axs[1,2].set_ylabel('ONI Anomaly')
# axs[1,2].set_title('All together n='+str(len(rgi_gdf)))

plt.tight_layout()

#%%
xnames = seasons
ynames = names

fig,ax = plt.subplots(figsize=(10,3.5))
im = ax.imshow(corrs, cmap='bwr', vmin=-0.5, vmax=0.5)
ax.set_xticks(np.arange(0,len(seasons)))
ax.set_xticklabels(xnames)
ax.set_yticks([0,1,2,3,4,5])
ax.set_yticklabels(ynames)
fig.colorbar(im, ax=ax, label='Correlation')

for (j,i),label in np.ndenumerate(corrs):
    if abs(label)>0.3: weight='bold'
    else: weight='normal'
    ax.text(i, j, round(label,2), ha='center', va='center', weight=weight)
    
plt.tight_layout()
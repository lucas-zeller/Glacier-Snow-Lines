# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:22:44 2022

@author: lzell
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
import geopandas as gpd
import geemap
from geemap import ml
import ee



#%%
save = 0 

# define folder paths, filepaths, etc...
fp_main = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA")
fp_rgi = os.path.join(fp_main, "RGI", "01_rgi60_Alaska", "01_rgi60_Alaska.shp")

# open rgi with geopandas
rgi_gdf = gpd.read_file(fp_rgi) #EPSG:4326
print("Initial crs:",rgi_gdf.crs)
rgi_gdf = rgi_gdf.to_crs("EPSG:3338") # reproject to Alaska Albers
print("Reprojected to Alaska Albers (epsg:3338)")


#%%
# subset to exclude the Brooks range (O2Region==1)
rgi_gdf_subset = rgi_gdf[~(rgi_gdf['O2Region']=='1')]

# remove glaciers smaller than 0.5km2
rgi_gdf_subset = rgi_gdf_subset[~(rgi_gdf_subset['Area']<0.5)]

fig,axs = plt.subplots()
rgi_gdf_subset.plot(ax=axs)

#%%
# now buffer by 5 km, and remerge the geometries
rgi_gdf_buff_geom = rgi_gdf_subset.buffer(5000).unary_union
rgi_gdf_buff = gpd.GeoDataFrame(crs=rgi_gdf.crs, geometry=[rgi_gdf_buff_geom])

#%%
# figure to visualize this buffer
fig,axs = plt.subplots()
rgi_gdf_buff.plot(ax=axs)
rgi_gdf_subset.plot(ax=axs, color='black')

#%%
# now save this buffered gdf to a shapefile and go upload it to be an asset on GEE
save = 0
if save:
    out_path = os.path.join(fp_main, "RGI", "01_rgi60_Alaska", "01_rgi60_Alaska_buffered_5km.shp")
    rgi_gdf_buff.to_file(out_path)  




  
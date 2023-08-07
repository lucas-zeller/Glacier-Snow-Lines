# -*- coding: utf-8 -*-
"""
Load eEntinel-2 Imagery of a given area, classify it using a random forest stored
as a GEE asset.
"""

import ee
#import geetools
import numpy as np
import json
import pandas as pd
import geopandas as gpd
import os
import numpy as np


#%%
# # Trigger the authentication flow.
# ee.Authenticate()

# # Initialize the library.
# ee.Initialize()

#%%
'''USER DEFINED VARIABLES'''

# rgi outlines
asset_rgi01_Alaska = ee.FeatureCollection('projects/lzeller/assets/01_rgi60_Alaska')

# simple outline
asset_simpleoutline = ee.FeatureCollection('projects/lzeller/assets/AGVAsimplearea')  # eventually redo this to be areas within 5km of rgi outlines >0.5km

# subregion outlines
asset_subregions = ee.FeatureCollection('projects/lzeller/assets/Alaska_RGI_Subregions')



# function to return the geometry of a given image collection
def get_ic_geometry(ic):
    return ic.geometry().dissolve()

# function to redraw subregion boundaries to bbox of glacier within them
def redraw_boundary(region):
    subset_rgi = asset_rgi01_Alaska.filterBounds(region.geometry())
    subset_bounds = subset_rgi.map( lambda f : f.setGeometry(f.geometry().bounds()) )
    subset_bounds = subset_bounds.geometry().bounds()
    
    return region.setGeometry(subset_bounds)

# remake subregion geometries
asset_subregions = asset_subregions.map( lambda f : redraw_boundary(f))

# load dem mosaic
dem = (ee.ImageCollection('COPERNICUS/DEM/GLO30')
               .select('DEM')
               .filterBounds(asset_subregions)
               .mosaic()
               .multiply(10)
               .toInt()
        )

#%%
# iterate through each subregion, exporting dem for each
# get the names/numbers of those regions, aggregate to list (to iterate), count them
names = asset_subregions.aggregate_array('id').getInfo()
# print(names)

n_features = len(names)
subregions_list = asset_subregions.toList(n_features+1) 


# now for each subregion or interest, clip single_image_clipped to that geometry and then export
for i in range(0, len(names)):
    
    # define folders, etc...
    # description = f'S2_{sensing_orbit_number:03d}_{date_start}_{date_end}'
    description10 = f'region_{names[i]:02d}_10m' 
    description30 = f'region_{names[i]:02d}_30m'
    folder10 = '10m_COP_GLO30'
    folder30 = '30m_COP_GLO30'
    
    # if you want to just do one or two regions
    # if names[i] in [9,10]: 
    #     print(names[i])
    # else: continue

    # grab this feature
    feature_i = ee.Feature(subregions_list.get(i)) 
    
    # grab geometry of the region
    region = feature_i.geometry()
    
    # clip dem to this region
    clipped_dem = dem.clip(region).unmask(0) 

    # export the image to drive
    task = ee.batch.Export.image.toDrive(
        image = clipped_dem, #regional_clipped_image,
        region = region.bounds(), # region.bounds()
        folder = folder10,
        scale = 10,
        maxPixels = int(1e13),
        crs = 'EPSG:3338',
        crsTransform = [10,0,0,0,-10,0],
        description = description10,
        skipEmptyTiles = True
        )
    
    task.start()
    print('DEM export started', f"{description10}")
    
    # export the image to drive
    task = ee.batch.Export.image.toDrive(
        image = clipped_dem, #regional_clipped_image,
        region = region.bounds(), # region.bounds()
        folder = folder30,
        scale = 30,
        maxPixels = int(1e13),
        crs = 'EPSG:3338',
        crsTransform = [30,0,0,0,-30,0],
        description = description30,
        skipEmptyTiles = True
        )
    
    task.start()
    print('DEM export started', f"{description30}")

        
        
        
        
        
        
        
        
        
        
        
        
        
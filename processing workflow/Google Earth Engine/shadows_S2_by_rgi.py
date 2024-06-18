# -*- coding: utf-8 -*-
"""
Classify all sentinel-2 imagery of a given glacier polygon and export it
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
# # # Trigger the authentication flow.
# ee.Authenticate()

# # # Initialize the library.
ee.Initialize()

#%%
'''USER DEFINED VARIABLES'''
# run_id_export = 0
# run_cloudprob_export = 0
save_metadata = 0
run_shadow_export = 1
# run_scl_export = 0 


date_start = '2018-01-01'
date_end = '2023-01-01'

merge_dtis = 0

# list of all the sensing orbit numbers you want to use
# [1,101,15,115,72,29,129,86,43,143,100,57,14,114,71,28,128,85,42] # from west to east
# [1,14,15,28,29,42,43,57,71,72,85,86,100,101,114,115,128,129,143] # in numeric order
# wolv: 43 and 143

# open list of the validation glaciers, so we can run those first
folder_AGVA = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA")
all_validation_df = pd.read_csv(os.path.join(folder_AGVA, 'Validation', 'Validation Glaciers.csv'))
all_validation_rgi = list(all_validation_df['RGIId'].values)

# rgi outlines
asset_rgi01_Alaska = ee.FeatureCollection('projects/lzeller/assets/01_rgi60_Alaska')

### subset to the rgi outlines you want to use 
# rgi_to_use = asset_rgi01_Alaska.filter(ee.Filter.inList('Name',['Wolverine Glacier', 'Gulkana Glacier']))

### check which ones we've already classified

# rgi_to_use = asset_rgi01_Alaska.filter(ee.Filter.inList('O2Region',["2"])) #wolv=region4, gulk=region2, brooks=1, peninsula=3
rgi_to_use = asset_rgi01_Alaska.filter(ee.Filter.inList('O2Region',["2","3","4","5","6"])) # need meta for 2,5,6
rgi_to_use = rgi_to_use.filter(ee.Filter.gte('Area',2))

rgi_to_use = rgi_to_use.filter(ee.Filter.inList('RGIId',all_validation_rgi).Not())
# rgi_to_use = rgi_to_use.filter(ee.Filter.inList('RGIId',all_validation_rgi))

# rgi_to_use = asset_rgi01_Alaska.filter(ee.Filter.gte('Area',5))
# rgi_to_use = asset_rgi01_Alaska.filter(ee.Filter.inList('RGIId',['RGI60-01.15588']))

print(len(rgi_to_use.aggregate_array('RGIId').getInfo()))
rgi_to_use = rgi_to_use.sort('RGIId')

# simple outline
asset_simpleoutline = ee.FeatureCollection('projects/lzeller/assets/AGVAsimplearea')  # eventually redo this to be areas within 5km of rgi outlines >0.5km

# subregion outlines
# asset_subregions = ee.FeatureCollection('projects/lzeller/assets/Alaska_RGI_Subregions')

# load dem image 
dem = (ee.ImageCollection('COPERNICUS/DEM/GLO30')
               .select('DEM')
               .filterBounds(rgi_to_use)
               .mosaic())


# iterate through each rgi outline, sending them off to be analyzed
rgi_names = rgi_to_use.aggregate_array('RGIId').getInfo()
# print(rgi_names)
n_features = len(rgi_names)

rgi_list = rgi_to_use.toList(n_features) 

### get list of the ones which have already been classified
folder = os.path.join('G:',os.sep,'My Drive','AGVA Snow Lines',"S2_Shadow_Raw")
all_images = os.listdir(folder)
all_rgis_done = list(set( [i[3:17] for i in all_images] ))
print("We have already IDed:",len(all_rgis_done))
print("This order is asking for:", n_features)
print()
print("This order should end up running:",n_features-len(all_rgis_done))
print("There should be",len(all_rgis_done),'duplicates')
print()

# print(all_rgis_done[0])
# print(rgi_names[0])
# print("We are identifying this many duplicates:",len(done))

#%%

# now for each subregion or interest, clip single_image_clipped to that geometry and then export
# skip=1
done = []
tt = 0
for i in range(0, len(rgi_names)):
    
    # get rgi name
    rgi_name = rgi_names[i]
    
    # if it's already been run, then skip it
    if rgi_name in all_rgis_done:
        done.append(rgi_name)
        continue
    else:
        tt+=1
    # grab this feature
    rgi_i = ee.Feature(rgi_list.get(i)) 
    # print(rgi_i.getInfo())
    
    # various ways to skip to certain glaciers
    # if rgi_name == "RGI60-01.18971": skip=0
    # if skip: continue

    # if i<474: continue  
    # if i!=270: continue

    # print(rgi_name)
    # if rgi_name != "RGI60-01.18951": continue
    # print(rgi_name)
    
    ###############################
    # metadata for o2region=2, i=270 was never exported DONE
    # also i=474 and above never exported DONE
    
    # ran out of drive space. 01.15769 not saved (this had 3 exports for some reason?) DONE
    # 18951 not saved, 18971 and above not saved DONE
    ###############################
    
    # create folder and file names
    description = f'S2_{rgi_name}_{date_start}_{date_end}'
    description_cloud = f'S2_{rgi_name}_{date_start}_{date_end}_cloud'
    description_scl = f'S2_{rgi_name}_{date_start}_{date_end}_scl'
    folder_img = 'S2_Classified_Raw'
    folder_cloud = 'S2_Cloud_Raw'
    folder_scl = 'S2_SCL_Raw'
    folder_masked = 'S2_Classified_Cloudmasked_Raw'
    
    if run_shadow_export:
        # if i<2977: continue
        print(f"{i} of {len(rgi_names)} : {rgi_name}")
        
    
    # Load image collection which we will want to classify
    # S2_images = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    S2_images = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                      .filterDate(date_start, date_end) # filter by date range
                      .filter(ee.Filter.calendarRange(5, 11, 'month')) # filter by month
                      .filterBounds(rgi_i.geometry())
                      .sort('system:time_start'))
    
    # get a list of all the datatake_identifiers (individual pass overs) that there are
    # these will be used for metadata later on
    if merge_dtis:
        dtis = S2_images.aggregate_array("DATATAKE_IDENTIFIER").distinct()
    else:
        dtis = S2_images.aggregate_array("DATATAKE_IDENTIFIER")
    # print("Number of DTIs:", len(dtis.getInfo()))
    
    def get_shadows(image):
        
        # get solar azimuth and zenith   
        azimuth = image.get('MEAN_SOLAR_AZIMUTH_ANGLE')
        zenith = image.get('MEAN_SOLAR_ZENITH_ANGLE')
    
        # calculate shadows
        shadow_map = ee.Terrain.hillShadow(**{
          'image': dem,
          'azimuth': azimuth,
          'zenith': zenith,
          'neighborhoodSize': 200,
          'hysteresis': True
        })
        
        # a little convolution to smooth it out
        boxcar = ee.Kernel.square(30, 'meters', True)
        shadow_map = shadow_map.convolve(boxcar).round()
        
        return shadow_map
    
    
    # apply function to classify shadows in full image collection
    S2_shadows = S2_images.map( lambda i : get_shadows(i)) # maybe should clip to rgi before classifying?
    
    # collapse the classified image collection into a single, multi-band image
    single_image = S2_shadows.toBands()
    
    # clip, sort out the mask number, etc...
    single_image = single_image.set({'dtis':dtis}) # set the list of dtis as a property
    single_image = single_image.unmask(99) # set masked pixels to 99
    single_image = single_image.clip(rgi_i.geometry()).unmask(99) # clip to rgi outline
    
    # export the image
    if run_shadow_export:
            task = ee.batch.Export.image.toDrive(
                image = single_image.toUint8(),
                region = rgi_i.geometry(),
                folder = 'S2_Shadow_Raw',
                scale = 10,
                maxPixels = int(1e13),
                crs = 'EPSG:3338',
                crsTransform = [10,0,0,0,-10,0],
                description = description,
                skipEmptyTiles = True
                )
        
            task.start()
            print('Shadow export started')
        
print("This order will actually end up running:",tt)
print("We have identified this many duplicates:",len(done))
        #%%
    
        
        
        
    
    
    
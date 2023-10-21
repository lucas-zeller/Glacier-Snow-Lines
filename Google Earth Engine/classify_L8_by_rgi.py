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
# # Trigger the authentication flow.
# ee.Authenticate()

# # Initialize the library.
# ee.Initialize()

#%%
'''USER DEFINED VARIABLES'''
run_id_export = 0
run_cloudprob_export = 0
save_metadata = 1
run_cloudmasked_id_export = 1
run_scl_export = 0 


date_start = '2018-01-01'
date_end = '2019-01-01'

merge_dtis = 0

# list of all the sensing orbit numbers you want to use
# [1,101,15,115,72,29,129,86,43,143,100,57,14,114,71,28,128,85,42] # from west to east
# [1,14,15,28,29,42,43,57,71,72,85,86,100,101,114,115,128,129,143] # in numeric order
# wolv: 43 and 143

# rgi outlines
asset_rgi01_Alaska = ee.FeatureCollection('projects/lzeller/assets/01_rgi60_Alaska')

### subset to the rgi outlines you want to use 
rgi_to_use = asset_rgi01_Alaska.filter(ee.Filter.inList('Name',['Wolverine Glacier']))

# rgi_to_use = asset_rgi01_Alaska.filter(ee.Filter.inList('O2Region',["4"])) #wolv=region4, gulk=region2
rgi_to_use = rgi_to_use.filter(ee.Filter.gte('Area',2))

# rgi_to_use = asset_rgi01_Alaska.filter(ee.Filter.gte('Area',5))

print(len(rgi_to_use.aggregate_array('RGIId').getInfo()))
rgi_to_use = rgi_to_use.sort('RGIId')

# simple outline
asset_simpleoutline = ee.FeatureCollection('projects/lzeller/assets/AGVAsimplearea')  # eventually redo this to be areas within 5km of rgi outlines >0.5km

# subregion outlines
asset_subregions = ee.FeatureCollection('projects/lzeller/assets/Alaska_RGI_Subregions')



### load in rnadom forest classifier, do initial conversion from string to ee.classifier
asset_RF_L8 = ee.FeatureCollection('projects/lzeller/assets/random_forest_L8_TOA')

# first replace "#" with "\n" in each tree
rf_strings = asset_RF_L8.aggregate_array("tree").map(
  lambda x : ee.String(x).replace( "#", "\n", "g") )

# then convert the strings to a RF calssifier
RF_classifier = ee.Classifier.decisionTreeEnsemble(rf_strings)


# function to rename Sentinel-2 bands
def renameBandsL8(image):
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    renamed_bands = ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    return image.select(bands).rename(renamed_bands);

# function to add ndwi, ndsi to an image after bands are renamed
def add_ndwi_ndsi(image):
    ndwi = image.normalizedDifference(['green', 'nir']).rename('ndwi')
    ndsi = image.normalizedDifference(['green', 'swir1']).rename('ndsi')
    return image.addBands([ndwi, ndsi])

# lists of names of each band that is going to be used
bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
renamed_bands = ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2']
raw_predictors = ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndwi', 'ndsi']
so_bands = ['SO_coastal', 'SO_blue', 'SO_green', 'SO_red', 'SO_nir', 'SO_swir1', 'SO_swir2', 'SO_ndsi', 'SO_ndwi']
dn_bands = ['DN_coastal', 'DN_blue', 'DN_green', 'DN_red', 'DN_nir', 'DN_swir1', 'DN_swir2', "DN_ndsi", "DN_ndwi"]

# load snow-on 'baseline' mosaic
snow_on = (ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
                  .filterDate('2013-01-01', '2023-01-01')
                  .filter(ee.Filter.dayOfYear(50,121))
                  .filterBounds(asset_simpleoutline)
                  .filter(ee.Filter.lte('CLOUD_COVER_LAND',10))
                  .median())

# rename the snow-on bands, add ndwi and ndsi
snow_on = renameBandsL8(snow_on)
snow_on = add_ndwi_ndsi(snow_on)

# function to calculate the diff-norm image, add bands to image you want to classify
def add_DN_bands(image):
    # select bands
    original_image = image.select(raw_predictors)
    SO_image = snow_on.select(raw_predictors)
    
    # divide
    DN_image = original_image.divide(SO_image)
    
    # rename and add bands
    DN_image = DN_image.select(raw_predictors).rename(dn_bands)
    SO_image = SO_image.select(raw_predictors).rename(so_bands)
    return_image = image.addBands(SO_image)
    return_image = return_image.addBands(DN_image)
    
    return return_image

# function to return the geometry of a given image collection
def get_ic_geometry(ic):
    return ic.geometry().dissolve()

# function to redraw subregion boundaries to bbox of glacier within them (to limit size)
def redraw_boundary(region):
    subset_rgi = asset_rgi01_Alaska.filterBounds(region.geometry())
    subset_bounds = subset_rgi.map( lambda f : f.setGeometry(f.geometry().bounds()) )
    subset_bounds = subset_bounds.geometry().bounds()
    
    return region.setGeometry(subset_bounds)

# remake subregion geometries
asset_subregions = asset_subregions.map( lambda f : redraw_boundary(f))

#%%

# iterate through each rgi outline, sending them off to be analyzed
rgi_names = rgi_to_use.aggregate_array('RGIId').getInfo()
# print(rgi_names)
n_features = len(rgi_names)

rgi_list = rgi_to_use.toList(n_features) 

# now for each subregion or interest, clip single_image_clipped to that geometry and then export
# skip=1
for i in range(0, len(rgi_names)):
    
    # grab this feature
    rgi_i = ee.Feature(rgi_list.get(i)) 
    # print(rgi_i.getInfo())
    
    # get rgi name
    rgi_name = rgi_names[i]
    
    # various ways to skip to certain glaciers
    # if rgi_name == "RGI60-01.18971": skip=0
    # if skip: continue

    # if i<474: continue  
    # if i!=270: continue

    # print(rgi_name)
    # if rgi_name != "RGI60-01.18951": continue
    # print(rgi_name)
    
    # create folder and file names
    description = f'L8_{rgi_name}_{date_start}_{date_end}'
    description_cloud = f'L8_{rgi_name}_{date_start}_{date_end}_cloud'
    description_scl = f'L8_{rgi_name}_{date_start}_{date_end}_scl'
    folder_img = 'L8_Classified_Raw'
    folder_cloud = 'L8_Cloud_Raw'
    folder_scl = 'L8_SCL_Raw'
    folder_masked = 'L8_Classified_Cloudmasked_Raw'
    
    if run_cloudmasked_id_export or run_id_export or run_cloudprob_export or save_metadata or run_scl_export:
        print(f"\n{i} of {len(rgi_names)} : {rgi_name}")
    
    # Load image collection which we will want to classify
    # S2_images = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    L8_images = (ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
                      .filterDate(date_start, date_end) # filter by date range
                      .filterBounds(rgi_i.geometry())
                      .sort('system:time_start')) 
    
    # get list of image product ids to save
    image_ids = L8_images.aggregate_array('LANDSAT_PRODUCT_ID')
    
    ### function to classify snow/firn/ice using random forest
    def classify_image_rf(image):
        
        # rename bands, add ndwi, ndsi
        image = renameBandsL8(image) 
        image = add_ndwi_ndsi(image)
        image = add_DN_bands(image)
        
        # classify with rf
        ided = image.classify(RF_classifier).unmask(99)
        
        # copy properties
        ided = ided.copyProperties(image)
        
        return ided 
    
    ### function to get binary mask for clouds and cloud shadows
    def get_clouds_and_shadows_L8(image):
        
        # grab the QA_pixel band
        qa_pixel_band = image.select('QA_PIXEL')
        
        # create bitmasks for cloud and shadow (bits 3 and 4)
        cloud_mask = 1 << 3
        shadow_mask = 1 << 4
        
        # test each pixel for cloud and shadow
        clouds = qa_pixel_band.bitwiseAnd(cloud_mask).neq(0)
        shadows = qa_pixel_band.bitwiseAnd(shadow_mask).neq(0)
        
        # take the property image and format it into its own ee.image
        cloud_prob = ee.Image(image.get('cloud_mask')).select('probability').rename('cloud_prob')
        
        # get all the metadata and properties to be the same as the original
        cloud_prob = ee.Image(image).addBands(ee.Image(cloud_prob)).set('cloud_mask', None).select('cloud_prob')
        
        return clouds.add(shadows)
    
    
    ##### run id export for this glacier if you want 
    # apply function to classify snow/firn/ice in full image collection
    # S2_identified = S2_images.map( lambda i : classify_image(i)) 
    L8_identified = L8_images.map( lambda i : classify_image_rf(i)) # maybe should clip to rgi before classifying?
        
    # collapse the classified image collection into a single, multi-band image
    single_image = L8_identified.toBands()
    
    # single_image = single_image.set({'dtis':dtis}) # set the list of dtis as a property
    single_image = single_image.unmask(99) # set masked pixels to 99
    single_image = single_image.clip(rgi_i.geometry()).unmask(99) # clip to rgi outline
    
    if run_id_export:
        # export the image to drive
        task = ee.batch.Export.image.toDrive(
            image = single_image, #regional_clipped_image,
            region = rgi_i.geometry(), # region.bounds()
            folder = folder_img,
            scale = 10,
            maxPixels = int(1e13),
            crs = 'EPSG:3338',
            crsTransform = [10,0,0,0,-10,0],
            description = description,
            skipEmptyTiles = True
            )
        
        task.start()
        print('Classified image stack export started')#, f"{description}")
        
        
        
    # run cloud probability export for this glacier if you want
       
    # apply function to grab SCL from each image
    L8_clouds = L8_images.map( lambda i : get_clouds_and_shadows_L8(i))
    
    # collapse the classified image collection into a single, multi-band image
    single_image_cloud = L8_clouds.toBands()
    
    single_image_cloud = single_image_cloud.unmask(199) # set masked pixels to 99
    single_image_cloud = single_image_cloud.clip(rgi_i.geometry()).unmask(199) # clip to rgi outline
    
    # round down to nearest 10 and export
    # single_image_cloud = single_image_cloud.add(5).divide(10).round().toInt8()
    
    if run_cloudprob_export:
        # export the image to drive
        task = ee.batch.Export.image.toDrive(
            image = single_image_cloud, #regional_clipped_image,
            region = rgi_i.geometry(), # region.bounds()
            folder = folder_cloud,
            scale = 10,
            maxPixels = int(1e13),
            crs = 'EPSG:3338',
            crsTransform = [10,0,0,0,-10,0],
            description = description_cloud,
            skipEmptyTiles = True
            )
        
        task.start()
        print('Cloud Probability image stack export started')#, f"{description_cloud}")
     
 
    if run_cloudmasked_id_export:
        cloud_mask = single_image_cloud.lte(0.5) # this will make pixels that arent clouds or shadows "true"
        cloud_masked_class = single_image.mask(cloud_mask).unmask(99)
        
        # export the image to drive
        task = ee.batch.Export.image.toDrive(
            image = cloud_masked_class, #regional_clipped_image,
            region = rgi_i.geometry(), # region.bounds()
            folder = folder_masked,
            scale = 10,
            maxPixels = int(1e13),
            crs = 'EPSG:3338',
            crsTransform = [10,0,0,0,-10,0],
            description = description,
            skipEmptyTiles = True
            )
        
        task.start()
        print('Classified image stack export started')
        
    #%%
        
    if save_metadata:
        
        # create initial df with datatake_identifier, sensing_orbit_number as columns
        out_df = pd.DataFrame( {'LANDSAT_PRODUCT_ID':image_ids.getInfo(), 'RGIId':rgi_name} )
        
        # add columns for satellite (a/b), date    GS[SS]_[YYYYMMDDTHHMMSS]_[RRRRRR]_N[xx.yy]
        # out_df['satellite'] = [ i[:4] for i in out_df['datatake_identifier'] ]
        # out_df['date'] = [ i[5:13] for i in out_df['datatake_identifier'] ]
        
        out_path = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop','AGVA','classified images','meta csv','L8', description+'.csv')
        out_df.to_csv(out_path)
        print('Metadata exported to local machine')
        
        #%%
    
        
        
        
    
    
    
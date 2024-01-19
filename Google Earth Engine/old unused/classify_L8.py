# -*- coding: utf-8 -*-
"""
Load Landsat Imagery of a given area, classify it using a random forest stored
as a GEE asset.
"""

import ee
#import geetools
import numpy as np
import json
import pandas as pd
import geopandas as gpd
import os


#%%
# # Trigger the authentication flow.
# ee.Authenticate()

# # Initialize the library.
# ee.Initialize()

#%%
'''USER DEFINED VARIABLES'''
run_id_export = 0
run_cloudmask_export = 1 # this isn't working with landsat collection 2 currently
save_metadata = 0

wrs_row = 18
wrs_path = 68
cloud_max = 90
date_start = '2015-01-01'
date_end = '2021-12-30'
description = 'P{}_R{}_{}_{}_{}'.format(wrs_path,wrs_row,date_start,date_end,cloud_max)
description_cloud = 'P{}_R{}_{}_{}_{}_cloud'.format(wrs_path,wrs_row,date_start,date_end,cloud_max)


#%%
# load all the GEE cloud assets that we will be using
#asset_rgi01_Alaska = ee.data.getAsset('projects/lzeller/assets/01_rgi60_Alaska')
asset_rgi01_Alaska = ee.FeatureCollection('projects/lzeller/assets/01_rgi60_Alaska')
asset_RF_L8 = ee.FeatureCollection('projects/lzeller/assets/random_forest_strings_with_SO')

#%%
### functions which will be called later

# for renaming bands ('SO_' indicates snow-on)
var bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10', 'ndwi', 'ndsi'];
var renamed_bands = ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ST_B10', 'ndwi', 'ndsi']
var so_bands = ['SO_coastal', 'SO_blue', 'SO_green', 'SO_red', 'SO_nir', 'SO_swir1', 'SO_swir2', 'SO_ST_B10', 'SO_ndwi', 'SO_ndsi'];
var dn_bands = ['DN_coastal', 'DN_blue', 'DN_green', 'DN_red', 'DN_nir', 'DN_swir1', 'DN_swir2', 'DN_ST_B10', 'DN_ndwi', 'DN_ndsi']


# rename bands to match naming convention of the RF
# 'SO' indicates the same bands but in the 'snow-on' mosaic
def renameBandsL8(image):
    bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10', 'SO_SR_B1', 'SO_SR_B2', 'SO_SR_B3', 'SO_SR_B4', 'SO_SR_B5', 'SO_SR_B6', 'SO_SR_B7', 'SO_ST_B10']
    new_bands = ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ST_B10', 'SO_coastal', 'SO_blue', 'SO_green', 'SO_red', 'SO_nir', 'SO_swir1', 'SO_swir2', 'SO_ST_B10']
    image = image.select(bands).rename(new_bands)
    return image

# add NDSI to an image
def addNDSIl8(image):
    ndsi = image.normalizedDifference(['green', 'swir1']).rename('ndsi')
    ndsi_SO = image.normalizedDifference(['SO_green', 'SO_swir1']).rename('SO_ndsi')
    return image.addBands([ndsi, ndsi_SO])

# add NDWI to an image
def addNDWIl8(image):
    ndwi = image.normalizedDifference(['green', 'nir']).rename('ndwi')
    ndwi_SO = image.normalizedDifference(['SO_green', 'SO_nir']).rename('SO_ndwi')
    return image.addBands([ndwi, ndwi_SO])

# calculate surface temperature (in celcius) from L8 band 10
def calcTemp(image):
    temp_c = image.select('ST_B10').multiply(0.00341802).add(149).subtract(273.15).rename('temp_c')
    temp_c_so = image.select('SO_ST_B10').multiply(0.00341802).add(149).subtract(273.15).rename('SO_temp_c')
    return image.addBands([temp_c, temp_c_so])

# calculate the difference between a given image and the snow-on mosaic,
# normalized to 0-1 by dividing by the snow-on values of each pixel
def diffNorm(image):
    bands = ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'temp_c', 'ndwi', 'ndsi']
    so_bands = ['SO_coastal', 'SO_blue', 'SO_green', 'SO_red', 'SO_nir', 'SO_swir1', 'SO_swir2', 'SO_temp_c', 'SO_ndwi', 'SO_ndsi']
    dn_bands = ['dn_coastal', 'dn_blue', 'dn_green', 'dn_red', 'dn_nir', 'dn_swir1', 'dn_swir2', 'dn_temp_c', 'dn_ndwi', 'dn_ndsi']
    diff_image = image.select(so_bands).subtract(image.select(bands))
    diff_image_norm = diff_image.divide(image.select(so_bands))#.multiply(-1).add(1)
    diff_image_norm = diff_image_norm.select(so_bands).rename(dn_bands)
    return image.addBands(diff_image_norm)

def maskL8T2(image):
  # Bits 3 and 4 are cloud shadow and cloud, respectively.
  cloudShadowBitMask = (1 << 3);
  cloudsBitMask = (1 << 4);
  
  # Get the pixel QA band.
  qa = image.select('QA_PIXEL');
  # Both flags should be set to zero, indicating clear conditions.
  mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0));
  #return image.updateMask(mask);
  return(mask)


#%%

# Load image collection which we will want to classify, filtering by cloud cover
L8_glaciers = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                  .filterDate(date_start, date_end)
                  .filter(ee.Filter.eq('WRS_PATH', wrs_path))
                  .filter(ee.Filter.eq('WRS_ROW', wrs_row))
                  .filter(ee.Filter.lte('CLOUD_COVER_LAND',cloud_max))
                  .sort('system:time_start')) 

print("Number of images to start: ", len(L8_glaciers.getInfo().get('features'))) 
#for i in L8_glaciers.getInfo().get('features'):
    #print (i.get('id')) 
print()

#%%
#make a cloud-score collection

cloud_collection = L8_glaciers.map( lambda image : maskL8T2(image) )
single_cloud_image = cloud_collection.toBands().clipToCollection(asset_rgi01_Alaska)



#%%
# load snow-on 'baseline' mosaic
snow_on = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                  .filter(ee.Filter.dayOfYear(50,121))
                  .filterBounds(ee.Geometry.Point(-149.9041,60.4134))
                  .filter(ee.Filter.lte('CLOUD_COVER_LAND',10))
                  .median())
                  
# rename bands ('SO_' indicates snow-on)
bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10']
so_bands = ['SO_SR_B1', 'SO_SR_B2', 'SO_SR_B3', 'SO_SR_B4', 'SO_SR_B5', 'SO_SR_B6', 'SO_SR_B7', 'SO_ST_B10']
snow_on = snow_on.select(bands).rename(so_bands)#.clip(outline)

L8_glaciers_mod = L8_glaciers.map( lambda image : image.addBands(snow_on) )
L8_glaciers_mod = L8_glaciers_mod.map( lambda image : renameBandsL8(image) )
L8_glaciers_mod = L8_glaciers_mod.map( lambda image : addNDSIl8(image) )
L8_glaciers_mod = L8_glaciers_mod.map( lambda image : addNDWIl8(image) )
L8_glaciers_mod = L8_glaciers_mod.map( lambda image : calcTemp(image) )
L8_glaciers_mod = L8_glaciers_mod.map( lambda image : calcTemp(image) )
L8_glaciers_mod = L8_glaciers_mod.map( lambda image : diffNorm(image) )
print("Number of images after band renaming: ", len(L8_glaciers.getInfo().get('features')))
#print(L8_glaciers.getInfo())

#%%
# we need to convert our random forest asset to a ee.Classifier
# first replace "#" with "\n" in each tree
rf_strings = asset_RF_L8.aggregate_array("tree").map(
  lambda x : ee.String(x).replace( "#", "\n", "g") )

# then convert the strings to a RF calssifier
RF_classifier = ee.Classifier.decisionTreeEnsemble(rf_strings)

#%%
# classify all the images, copying the metadata to the classification images
def classify_image(image):
    ided = image.classify(RF_classifier)
    ided = ided.copyProperties(image)
    return ided 
    
L8_id = L8_glaciers_mod.map( lambda image: classify_image(image) )
print('Number of images after IDing: ',len(L8_id.getInfo().get('features')))

#%%
# collapse the image collection into a single, multi-band image
single_image = L8_id.toBands()



#%%
# clip image to RGI outlines only, then set nans
single_image_clipped = single_image.clipToCollection(asset_rgi01_Alaska)
single_image_clipped = single_image_clipped.unmask(99)
single_image_clipped = single_image_clipped.set({'nodata':99})

# set the list of original band names as a property
id_list = L8_glaciers.aggregate_array('system:id')
single_image_clipped = single_image_clipped.set({'img_ids':id_list})

# can also do a list of other properties, like cloud cover
cloud_list = L8_glaciers.aggregate_array('CLOUD_COVER_LAND')
single_image_clipped = single_image_clipped.set({'CLOUD_COVER_LAND':cloud_list})

# save to a csv if you choose
if save_metadata:
    out_df = pd.DataFrame( {'id':id_list.getInfo(), 'cloud_cover_land':cloud_list.getInfo()} )
    out_path = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop','AGVA','classified','meta csv',description+'.csv')
    out_df.to_csv(out_path)

#%%
# export the ided image collection to google drive
# tasks = geetools.batch.Export.imagecollection.toDrive(
#     region = L8_glaciers.geometry().bounds(),
#     collection = L8_id_clipped,
#     folder = 'L8 Classified Images',
#     namePattern = '{id}',
#     scale = 30,
#     dataType = 'uint8',
#     maxPixels = int(1e13),
#     crs = 'EPSG:32606'
#     )

#%%
# export the multiband band image
if run_id_export:
    task = ee.batch.Export.image.toDrive(
        image = single_image_clipped,
        region = L8_glaciers.geometry().bounds(),
        folder = 'L8 Classified Images',
        scale = 30,
        maxPixels = int(1e13),
        crs = single_image.getInfo().get('bands')[0].get('crs'),
        description = description,
        skipEmptyTiles = True
        )

    task.start()
    print('Classified image stack export started')

#%%
#print(single_image.getInfo())
print(single_image.getInfo().get('bands')[0].get('crs'))

single_cloud_image = single_cloud_image.toByte()
if run_cloudmask_export:
    #print("sorry, not working right now")
    
    task = ee.batch.Export.image.toDrive(
        image = single_cloud_image,
        region = L8_glaciers.geometry().bounds(),
        folder = 'L8 cloud',
        scale = 30,
        maxPixels = int(1e13),
        crs = single_cloud_image.getInfo().get('bands')[0].get('crs'),
        description = description_cloud,
        skipEmptyTiles = True
        )

    task.start()
    print('Cloud masked image stack export started')

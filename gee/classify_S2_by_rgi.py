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
run_id_export = 1
run_scl_export = 0 # this isn't working with landsat collection 2 currently
run_cloudprob_export = 1
save_metadata = 1

date_start = '2015-01-01'
date_end = '2023-01-01'

merge_dtis = 0

# list of all the sensing orbit numbers you want to use
# [1,101,15,115,72,29,129,86,43,143,100,57,14,114,71,28,128,85,42] # from west to east
# [1,14,15,28,29,42,43,57,71,72,85,86,100,101,114,115,128,129,143] # in numeric order
# wolv: 43 and 143

# rgi outlines
asset_rgi01_Alaska = ee.FeatureCollection('projects/lzeller/assets/01_rgi60_Alaska')

# option to subset to only some
asset_rgi01_Alaska = asset_rgi01_Alaska.filter(ee.Filter.eq('Name','Gulkana Glacier'))
asset_rgi01_Alaska_1km = asset_rgi01_Alaska.filter(ee.Filter.gte('Area',1))

# simple outline
asset_simpleoutline = ee.FeatureCollection('projects/lzeller/assets/AGVAsimplearea')  # eventually redo this to be areas within 5km of rgi outlines >0.5km

# subregion outlines
asset_subregions = ee.FeatureCollection('projects/lzeller/assets/Alaska_RGI_Subregions')



### load in rnadom forest classifier, do initial conversion from string to ee.classifier
# asset_RF_S2 = ee.FeatureCollection('projects/lzeller/assets/random_forest_S2')
asset_RF_S2 = ee.FeatureCollection('projects/lzeller/assets/random_forest_S2_TOA')

# first replace "#" with "\n" in each tree
rf_strings = asset_RF_S2.aggregate_array("tree").map(
  lambda x : ee.String(x).replace( "#", "\n", "g") )

# then convert the strings to a RF calssifier
RF_classifier = ee.Classifier.decisionTreeEnsemble(rf_strings)


# function to rename Sentinel-2 bands
def renameBandsS2(image):
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    renamed_bands = ['coastal', 'blue', 'green', 'red', 're1', 're2', 're3', 'nir', 're4', 'vapor', 'swir1', 'swir2']
    return image.select(bands).rename(renamed_bands);

# function to add ndwi, ndsi to an image after bands are renamed
def add_ndwi_ndsi(image):
    ndwi = image.normalizedDifference(['green', 'nir']).rename('ndwi')
    ndsi = image.normalizedDifference(['green', 'swir1']).rename('ndsi')
    return image.addBands([ndwi, ndsi])

# lists of names of each band that is going to be used
bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'];
renamed_bands = ['coastal', 'blue', 'green', 'red', 're1', 're2', 're3', 'nir', 're4', 'vapor', 'swir1', 'swir2']
raw_predictors = ['coastal', 'blue', 'green', 'red', 're1', 're2', 're3', 'nir', 're4', 'vapor', 'swir1', 'swir2', 'ndwi', 'ndsi']
so_bands = ['SO_coastal', 'SO_blue', 'SO_green', 'SO_red', 'SO_re1', 'SO_re2', 'SO_re3', 'SO_nir', 'SO_re4', 'SO_vapor', 'SO_swir1', 'SO_swir2', 'SO_ndsi', 'SO_ndwi']
dn_bands = ['DN_coastal', 'DN_blue', 'DN_green', 'DN_red', 'DN_re1', 'DN_re2', 'DN_re3', 'DN_nir', 'DN_re4', 'DN_vapor', 'DN_swir1', 'DN_swir2', "DN_ndsi", "DN_ndwi"]

# load snow-on 'baseline' mosaic
snow_on = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                  .filterDate('2019-01-01', '2023-01-01')
                  .filter(ee.Filter.dayOfYear(50,121))
                  .filterBounds(asset_simpleoutline)
                  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE',10))
                  .median())

# rename the snow-on bands, add ndwi and ndsi
snow_on = renameBandsS2(snow_on)
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

# # add ndwi, ndsi to the snow-on image
# def add_ndwi_ndsi_so(image):
#     ndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('ndwi')
#     ndsi = image.normalizedDifference(['SR_B3', 'SR_B6']).rename('ndsi')
#     return image.addBands([ndwi, ndsi])

# # rename bands in the image you want to classify
# def rename_orig(image):
#     image = image.select(renamed_bands).rename(renamed_bands)
#     return image

# # rename the snow on bands
# def rename_so(image):
#     image = image.select(bands).rename(so_bands)
#     return image




# function to return the geometry of a given orbit
def mosaic_orbit_number_geom(orbit):
    ic = ( ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterDate('2020-01-01', '2020-02-01')
                  .filterBounds(asset_simpleoutline) )
    
    subset_ic = ic.filter(ee.Filter.eq("SENSING_ORBIT_NUMBER",orbit))
    orbit_geom = subset_ic.geometry().dissolve()#.intersection(asset_subregions)
  
    # make it into a feature and return
    feature = ee.Feature(orbit_geom, {"SENSING_ORBIT_NUMBER":orbit})
  
    return feature

# function to return the geometry of a given image collection
def get_ic_geometry(ic):
    return ic.geometry().dissolve()

# function to redraw subregion boundaries to bbox of glacier within them (to limit size)
def redraw_boundary(region):
    subset_rgi = asset_rgi01_Alaska.filterBounds(region.geometry())
    subset_bounds = subset_rgi.map( lambda f : f.setGeometry(f.geometry().bounds()) )
    subset_bounds = subset_bounds.geometry().bounds()
    
    return region.setGeometry(subset_bounds)

# # rename the snow-on bands
# snow_on = renameBandsS2(snow_on)

# # add ndwi, ndsi to the snow_on
# snow_on = add_ndwi_ndsi(snow_on)

# remake subregion geometries
asset_subregions = asset_subregions.map( lambda f : redraw_boundary(f))

#%%

# iterate through each rgi outline, sending them off to be analyzed
rgi_names = asset_rgi01_Alaska.aggregate_array('RGIId').getInfo()
# print(rgi_names)
n_features = len(rgi_names)

rgi_list = asset_rgi01_Alaska.toList(n_features) 

# now for each subregion or interest, clip single_image_clipped to that geometry and then export
for i in range(0, len(rgi_names)):
    
    # grab this feature
    rgi_i = ee.Feature(rgi_list.get(i)) 
    # print(rgi_i.getInfo())
    
    # get rgi name
    rgi_name = rgi_names[i]
    
    # create folder and file names
    description = f'S2_{rgi_name}_{date_start}_{date_end}_{merge_dtis}'
    description_cloud = f'S2_{rgi_name}_{date_start}_{date_end}_{merge_dtis}_cloud'
    description_scl = f'S2_{rgi_name}_{date_start}_{date_end}_{merge_dtis}_scl'
    folder_img = 'S2_Classified_Raw'
    folder_scl = 'S2_Cloud_Raw'
    folder_cloudsprobs = 'S2_CloudProb_Raw'
    
    # Load image collection which we will want to classify
    # S2_images = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    S2_images = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                      .filterDate(date_start, date_end) # filter by date range
                      .filterBounds(rgi_i.geometry())
                      .sort('system:time_start')) 
    
    # add cloud probability image as property to images
    S2_clouds = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                        .filterDate(date_start, date_end) # filter by date range
                        .filterBounds(rgi_i.geometry())
                        .sort('system:time_start'))
    
    S2_clouds = (ee.ImageCollection(ee.Join.saveFirst('cloud_mask').apply(**{
          'primary': S2_images,
          'secondary': S2_clouds,
          'condition': ee.Filter.equals(**{
              'leftField': 'system:index',
              'rightField': 'system:index' })
        })))
    
    # get a list of all the datatake_identifiers (individual pass overs) that there are
    if merge_dtis:
        dtis = S2_images.aggregate_array("DATATAKE_IDENTIFIER").distinct()
    else:
        dtis = S2_images.aggregate_array("DATATAKE_IDENTIFIER")
    # print("Number of DTIs:", len(dtis.getInfo()))
    
    ### function to classify snow/firn/ice using random forest
    def classify_image_rf(image):
        
        # rename bands, add ndwi, ndsi
        image = renameBandsS2(image) 
        image = add_ndwi_ndsi(image)
        image = add_DN_bands(image)
        
        # classify with rf
        ided = image.classify(RF_classifier).unmask(99)
        
        # override areas with ndsi<0
        
        # copy properties
        ided = ided.copyProperties(image)
        
        return ided 
    
    ### function to grab the scene classification map from sentinel-2 SR images
    def get_SCL(image):
        scl = image.select("SCL").unmask(99)
        return scl
    
    ### function to get the cloud_probability images
    def get_cloud_prob(image):
        
        # take the property image and format it into its own ee.image
        cloud_prob = ee.Image(image.get('cloud_mask')).select('probability').rename('cloud_prob')
        
        # get all the metadata and properties to be the same as the original
        cloud_prob = ee.Image(image).addBands(ee.Image(cloud_prob)).set('cloud_mask', None).select('cloud_prob')
        
        return cloud_prob.unmask(199)
    
    
    ##### run id export for this glacier if you want
    if run_id_export:
        
        # apply function to classify snow/firn/ice in full image collection
        # S2_identified = S2_images.map( lambda i : classify_image(i)) 
        S2_identified = S2_images.map( lambda i : classify_image_rf(i)) 
        
        # define function to merge classified images that have the same DATATAKE_IDENTIFIER
        def mosaic_DTI(dti):
            
            # subset to only this dti
            subset_ic = S2_identified.filter(ee.Filter.eq("DATATAKE_IDENTIFIER",dti))
   
            mosaic = subset_ic.mosaic() # mosaic them
            mosaic = mosaic.rename([ee.String(dti).slice(0,-3)]) # rename the band to dti value
            mosaic = mosaic.set({"DATATAKE_IDENTIFIER":dti})# copy properties
          
            return mosaic
        
        # merge classified images by dti
        if merge_dtis:
            s2_swaths = ee.ImageCollection(dtis.map( lambda d : mosaic_DTI(d)) )
        
        else:
            s2_swaths = S2_identified # test not doing the swath merging
        
        # collapse the classified image collection into a single, multi-band image
        single_image = s2_swaths.toBands()
        
        single_image = single_image.set({'dtis':dtis}) # set the list of dtis as a property
        single_image = single_image.unmask(99) # set masked pixels to 99
        single_image = single_image.clip(rgi_i.geometry()).unmask(99) # clip to rgi outline
        
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
        print('Classified image stack export started', f"{description}")
        
        
    # run cloud probability export for this glacier if you want
    if run_cloudprob_export:
         
        # apply function to grab SCL from each image
        S2_cloud_probs = S2_clouds.map( lambda i : get_cloud_prob(i))
        
        # function to merge SCLs that have the same DATATAKE_IDENTIFIER
        def mosaic_DTI_scl(dti):
            # subset to only this dti
            subset_ic = S2_SCLs.filter(ee.Filter.eq("DATATAKE_IDENTIFIER",dti))
            
            # mosaic them
            mosaic = subset_ic.mosaic()
            
            # rename the band to dti value
            mosaic = mosaic.rename([ee.String(dti).slice(0,-3)])
            
            # you can copy properties over to the mosaiced image, and it will be stored in the image properties.
            mosaic = mosaic.set({"DATATAKE_IDENTIFIER":dti})
          
            return mosaic
        
        # merge SCLs by dti
        if merge_dtis:
            s2_swaths = ee.ImageCollection(dtis.map( lambda d : mosaic_DTI_scl(d)) )
        else:
            s2_swaths = S2_cloud_probs # test not doing the swath merging
        
        # collapse the classified image collection into a single, multi-band image
        single_image = s2_swaths.toBands()
        
        single_image = single_image.set({'dtis':dtis}) # set the list of dtis as a property
        single_image = single_image.unmask(199) # set masked pixels to 99
        single_image = single_image.clip(rgi_i.geometry()).unmask(199) # clip to rgi outline
        
        # export the image to drive
        task = ee.batch.Export.image.toDrive(
            image = single_image, #regional_clipped_image,
            region = rgi_i.geometry(), # region.bounds()
            folder = folder_cloudsprobs,
            scale = 10,
            maxPixels = int(1e13),
            crs = 'EPSG:3338',
            crsTransform = [10,0,0,0,-10,0],
            description = description_scl,
            skipEmptyTiles = True
            )
        
        task.start()
        print('Cloud Probability image stack export started', f"{description_cloud}")
         
         
    # run scl export for this glacier if you want
    if run_scl_export:
        
        # apply function to grab SCL from each image
        S2_SCLs = S2_images.map( lambda i : get_SCL(i))
        
        # function to merge SCLs that have the same DATATAKE_IDENTIFIER
        def mosaic_DTI_scl(dti):
            # subset to only this dti
            subset_ic = S2_SCLs.filter(ee.Filter.eq("DATATAKE_IDENTIFIER",dti))
            
            # mosaic them
            mosaic = subset_ic.mosaic()
            
            # rename the band to dti value
            mosaic = mosaic.rename([ee.String(dti).slice(0,-3)])
            
            # you can copy properties over to the mosaiced image, and it will be stored in the image properties.
            mosaic = mosaic.set({"DATATAKE_IDENTIFIER":dti})
          
            return mosaic
        
        # merge SCLs by dti
        if merge_dtis:
            s2_swaths = ee.ImageCollection(dtis.map( lambda d : mosaic_DTI_scl(d)) )
        else:
            s2_swaths = S2_SCLs # test not doing the swath merging
        
        # collapse the classified image collection into a single, multi-band image
        single_image = s2_swaths.toBands()
        
        single_image = single_image.set({'dtis':dtis}) # set the list of dtis as a property
        single_image = single_image.unmask(99) # set masked pixels to 99
        single_image = single_image.clip(rgi_i.geometry()).unmask(99) # clip to rgi outline
        
        # export the image to drive
        task = ee.batch.Export.image.toDrive(
            image = single_image, #regional_clipped_image,
            region = rgi_i.geometry(), # region.bounds()
            folder = folder_scl,
            scale = 10,
            maxPixels = int(1e13),
            crs = 'EPSG:3338',
            crsTransform = [10,0,0,0,-10,0],
            description = description_scl,
            skipEmptyTiles = True
            )
        
        task.start()
        print('SCL image stack export started', f"{description_scl}")
        
        
    if save_metadata:
        
        # create initial df with datatake_identifier, sensing_orbit_number as columns
        out_df = pd.DataFrame( {'datatake_identifier':dtis.getInfo(), 'RGIId':rgi_name} )
        
        # add columns for satellite (a/b), date    GS[SS]_[YYYYMMDDTHHMMSS]_[RRRRRR]_N[xx.yy]
        out_df['satellite'] = [ i[:4] for i in out_df['datatake_identifier'] ]
        out_df['date'] = [ i[5:13] for i in out_df['datatake_identifier'] ]
        
        out_path = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop','AGVA','classified images','meta csv','S2', description+'.csv')
        out_df.to_csv(out_path)
        print('Metadata exported to local machine')
        
        #%%
    
        
        
        
    
    
    
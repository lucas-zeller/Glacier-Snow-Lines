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
import numpy as np


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

PCA = 1 # if 1: uses RF predictor with only 4 PCs.  if 0: will use only the raw and DN bands

# wrs_row = 18
# wrs_path = 68
cloud_max = 90
date_start = '2013-01-01'
date_end = '2021-12-30'

# open file containing all row/path combinations you want to analyze
fp = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop','AGVA','AGVA rows and paths.csv')
df = pd.read_csv(fp)

# rgi outlines
asset_rgi01_Alaska = ee.FeatureCollection('projects/lzeller/assets/01_rgi60_Alaska')

# simple outline
asset_simpleoutline = ee.FeatureCollection('projects/lzeller/assets/AGVAsimplearea')

# load snow-on 'baseline' mosaic
snow_on = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                  .filter(ee.Filter.dayOfYear(50,121))
                  .filterBounds(asset_simpleoutline)
                  .filter(ee.Filter.lte('CLOUD_COVER_LAND',10))
                  .median())
    
# iterate through each row/path, sending them off to be analyzed
c=0
for i,row in df.iterrows():
    c+=1
    #if c>1: continue
    wrs_row = int(row['row'])
    wrs_path = int(row['path'])

    # format export options
    description = 'P{}_R{}_{}_{}_{}'.format(wrs_path,wrs_row,date_start,date_end,cloud_max)
    description_cloud = 'P{}_R{}_{}_{}_{}_cloud'.format(wrs_path,wrs_row,date_start,date_end,cloud_max)
    folder_img = 'L8 Classified Images'
    folder_cloud = 'L8 Cloud'
    
    print()
    print('Row:',wrs_row,' Path:',wrs_path, c,'of',len(df))
    #%%
    # load all the GEE cloud assets that we will be using
    
    # Random forest classifier
    if PCA==0:
        asset_RF_L8 = ee.FeatureCollection('projects/lzeller/assets/RF_model_SO')
        
    # if we are using PCs, it is a different classifier, different folder names, etc...
    elif PCA==1:
        asset_RF_L8 = ee.FeatureCollection('projects/lzeller/assets/RF_model_SO_PC')
        description = description
        folder_img = folder_img + " PC"
        #folder_cloud = folder_cloud + " PC"
    
    # try to catch errors
    else:
        print("What predictors are you using?")
        asset_RF_L8 = 0
        
    
    #%%
    ### functions which will be called later
    
    # for renaming bands ('SO_' indicates snow-on)
    bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ndwi', 'ndsi'];
    renamed_bands = ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndwi', 'ndsi']
    so_bands = ['SO_coastal', 'SO_blue', 'SO_green', 'SO_red', 'SO_nir', 'SO_swir1', 'SO_swir2', 'SO_ndwi', 'SO_ndsi'];
    dn_bands = ['DN_coastal', 'DN_blue', 'DN_green', 'DN_red', 'DN_nir', 'DN_swir1', 'DN_swir2', 'DN_ndwi', 'DN_ndsi']
    pc_input_bands = ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndwi', 'ndsi', 'DN_coastal', 'DN_blue', 'DN_green', 'DN_red', 'DN_nir', 'DN_swir1', 'DN_swir2', 'DN_ndwi', 'DN_ndsi']
    
    # add ndwi, ndsi to the image you want to classify
    def original_ndwi_ndsi(image):
        ndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('ndwi')
        ndsi = image.normalizedDifference(['SR_B3', 'SR_B6']).rename('ndsi')
        return image.addBands([ndwi, ndsi])
    
    # add ndwi, ndsi to the snow-on image
    def so_ndwi_ndsi(image):
        ndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('ndwi')
        ndsi = image.normalizedDifference(['SR_B3', 'SR_B6']).rename('ndsi')
        return image.addBands([ndwi, ndsi])
    
    # rename bands in the image you want to classify
    def rename_orig(image):
        image = image.select(bands).rename(renamed_bands)
        return image
    
    # rename the snow on bands
    def rename_so(image):
        image = image.select(bands).rename(so_bands)
        return image
    
    # calculate the diff-norm image, add bands to image you want to classify
    def diff_norm(image):
        orig = image.select(renamed_bands)
        so = image.select(so_bands)
        diff_norm_image = orig.divide(so)
        diff_norm_image = diff_norm_image.select(renamed_bands).rename(dn_bands)
        return_image = image.addBands(diff_norm_image)
        return return_image
    
    # return a cloud mask image for each image
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
    
    # scale bands to each have mean=0, sd=1 (using same scaler as we used when training the RF)
    def scaleBands(image):    # (band-mean)/sd
        # define the mean and sd of each band (from RF_training_PCA.py, 5/9/2022)
        means = [33443.4, 34169.1, 33541, 32557.7, 25839.2, 7172.18, 7623.75, 0.140562, 0.629977, 0.666081,0.682462, 0.710574, 0.698219, 0.615067, 0.762485, 0.739348, 2.35308, 0.947847]
        sds = [9868.29, 9625.1, 8806.68, 9047.56, 8561.65, 568.862, 454.257, 0.0689007, 0.0885184, 0.190364, 0.18677, 0.181278, 0.187646, 0.192129, 0.069388, 0.0545672, 1.16816, 0.145545]
        
        # create multiband image from each, where each band has a constant value
        mean_image = ee.Image.constant(means)
        sd_image = ee.Image.constant(sds)
        
        # take the correct bands from input image, subtract mean, divide by sd
        return_image = image.select(pc_input_bands).subtract(mean_image).divide(sd_image)
        
        return return_image
    
    # add PCs as bands to image
    def addPCs(image):
        # normalize the bands ot mean=0, sd=1
        scaled_image = scaleBands(image)
        
        # load eigenvectors, eigenvalues
        eigenvectors = np.load('eigenvectors.npy').tolist()
        eigenvalues = np.load('eigenvalues.npy').tolist()
        
        # turn bands into array image
        array_image = scaled_image.select(pc_input_bands).toArray()
        
        # multiply the array image by eigenvectors
        principalComponents = ee.Image(ee.Array(eigenvectors)).matrixMultiply(array_image.toArray(1))
        
        # turn pc array back into multiband image
        pcImage = principalComponents.arrayProject([0]).arrayFlatten([['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18' ]])
        
        # add these PC bands to the original image
        return_image = image.addBands(pcImage)
        
        return return_image
    
    
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
    #print()

    
    #%%
    # add ndwi, ndsi to original images
    L8_glaciers_mod = L8_glaciers.map( lambda image : original_ndwi_ndsi(image) )
    
    # rename original bands
    L8_glaciers_mod = L8_glaciers_mod.map( lambda image : rename_orig(image) )
    
    # add ndsi, ndwi to snow on image
    snow_on_mod = so_ndwi_ndsi(snow_on)
    
    # rename snow on bands
    snow_on_mod = rename_so(snow_on_mod)
    
    # add snow on bands to training images
    L8_glaciers_mod = L8_glaciers_mod.map( lambda image : image.addBands(snow_on_mod))
    
    # create diff_norm image, add bands
    L8_glaciers_mod = L8_glaciers_mod.map( lambda image : diff_norm(image))
    
    # add pcs to image
    L8_glaciers_mod = L8_glaciers_mod.map( lambda image : addPCs(image))
    
    #print("Number of images after band renaming: ", len(L8_glaciers_mod.getInfo().get('features')))
    print("Number of bands per image: ", len(L8_glaciers_mod.getInfo().get('features')[0].get('bands')))
    # print("List of band names:")
    # c=1
    # for b in L8_glaciers_mod.getInfo().get('features')[0].get('bands'):
    #     print(c,b.get('id'))
    #     c+=1
    
    
    #%%
    # convert our random forest asset to a ee.Classifier
    # first replace "#" with "\n" in each tree
    rf_strings = asset_RF_L8.aggregate_array("tree").map(
      lambda x : ee.String(x).replace( "#", "\n", "g") )
    
    # then convert the strings to a RF calssifier
    RF_classifier = ee.Classifier.decisionTreeEnsemble(rf_strings)
    
    # classify all the images, copying the metadata to the classification images
    def classify_image(image):
        ided = image.classify(RF_classifier)
        ided = ided.copyProperties(image)
        return ided 
        
    L8_id = L8_glaciers_mod.map( lambda image: classify_image(image) )
    print('Number of images after IDing: ',len(L8_id.getInfo().get('features')))
    
    
    # collapse the image collection into a single, multi-band image
    single_image = L8_id.toBands()
    
    #%%
    #make a cloud-mask collection for the images
    cloud_collection = L8_glaciers.map( lambda image : maskL8T2(image) )
    
    # combine multiple images into single band image
    single_cloud_image = cloud_collection.toBands().clipToCollection(asset_rgi01_Alaska)
    
    # convert cloud mask to 8bit image
    single_cloud_image = single_cloud_image.toByte()
    
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
        out_path = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop','AGVA','classified images','meta csv',description+'.csv')
        out_df.to_csv(out_path)
        print('Metadata exported to local machine')
    
    
    #%%
    # export the multiband band image
    if run_id_export:
        task = ee.batch.Export.image.toDrive(
            image = single_image_clipped,
            region = L8_glaciers.geometry().bounds(),
            folder = folder_img,
            scale = 30,
            maxPixels = int(1e13),
            crs = single_image.getInfo().get('bands')[0].get('crs'),
            description = description,
            skipEmptyTiles = True
            )
    
        task.start()
        print('Classified image stack export started')
    
    #%%
    # export cloud mask
    if run_cloudmask_export:
        #print("sorry, not working right now")
        
        task = ee.batch.Export.image.toDrive(
            image = single_cloud_image,
            region = L8_glaciers.geometry().bounds(),
            folder = folder_cloud,
            scale = 30,
            maxPixels = int(1e13),
            crs = single_cloud_image.getInfo().get('bands')[0].get('crs'),
            description = description_cloud,
            skipEmptyTiles = True
            )
    
        task.start()
        print('Cloud masked image stack export started')

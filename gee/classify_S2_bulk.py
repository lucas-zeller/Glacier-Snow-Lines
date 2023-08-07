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
run_id_export = 0
run_scl_export = 1 # this isn't working with landsat collection 2 currently
save_metadata = 1

date_start = '2021-01-01'
date_end = '2021-12-31'

regions_override = [10]
# we are likely going to need to separate this out into individual years
# in order to stay below GEE user memory limits

# list of all the sensing orbit numbers you want to use
# [1,101,15,115,72,29,129,86,43,143,100,57,14,114,71,28,128,85,42] # from west to east
# [1,14,15,28,29,42,43,57,71,72,85,86,100,101,114,115,128,129,143] # in numeric order
# wolv: 43 and 143

# rgi outlines
asset_rgi01_Alaska = ee.FeatureCollection('projects/lzeller/assets/01_rgi60_Alaska')

# option to subset to only some
# asset_rgi01_Alaska = asset_rgi01_Alaska.filter(ee.Filter.eq('Name','Wolverine Glacier'))

# simple outline
asset_simpleoutline = ee.FeatureCollection('projects/lzeller/assets/AGVAsimplearea')  # eventually redo this to be areas within 5km of rgi outlines >0.5km

# subregion outlines
asset_subregions = ee.FeatureCollection('projects/lzeller/assets/Alaska_RGI_Subregions')

# load snow-on 'baseline' mosaic
snow_on = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterDate('2019-01-01', '2023-12-31')
                  .filter(ee.Filter.dayOfYear(50,121))
                  .filterBounds(asset_simpleoutline)
                  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE',10))
                  .median())

### load in rnadom forest classifier, do initial conversion from string to ee.classifier
asset_RF_S2 = ee.FeatureCollection('projects/lzeller/assets/random_forest_S2')

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

# lists of names of each band that is going to be used
bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'];
renamed_bands = ['coastal', 'blue', 'green', 'red', 're1', 're2', 're3', 'nir', 're4', 'vapor', 'swir1', 'swir2']
so_bands = ['SO_coastal', 'SO_blue', 'SO_green', 'SO_red', 'SO_re1', 'SO_re2', 'SO_re3', 'SO_nir', 'SO_re4', 'SO_vapor', 'SO_swir1', 'SO_swir2']
dn_bands = ['DN_coastal', 'DN_blue', 'DN_green', 'DN_red', 'DN_re1', 'DN_re2', 'DN_re3', 'DN_nir', 'DN_re4', 'DN_vapor', 'DN_swir1', 'DN_swir2', "DN_ndsi", "DN_ndwi"]
raw_predictors = ['coastal', 'blue', 'green', 'red', 're1', 're2', 're3', 'nir', 're4', 'vapor', 'swir1', 'swir2', 'ndwi', 'ndsi']

# function to add ndwi, ndsi to the image you want to classify
def add_ndwi_ndsi(image):
    ndwi = image.normalizedDifference(['green', 'nir']).rename('ndwi')
    ndsi = image.normalizedDifference(['green', 'swir1']).rename('ndsi')
    return image.addBands([ndwi, ndsi])

# add ndwi, ndsi to the snow-on image
def add_ndwi_ndsi_so(image):
    ndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('ndwi')
    ndsi = image.normalizedDifference(['SR_B3', 'SR_B6']).rename('ndsi')
    return image.addBands([ndwi, ndsi])

# rename bands in the image you want to classify
def rename_orig(image):
    image = image.select(renamed_bands).rename(renamed_bands)
    return image

# rename the snow on bands
def rename_so(image):
    image = image.select(bands).rename(so_bands)
    return image

# calculate the diff-norm image, add bands to image you want to classify
def add_DN_bands(image):
    orig = image.select(raw_predictors)
    so = image.select(raw_predictors)
    DN_image = orig.divide(so)
    DN_image = DN_image.select(raw_predictors).rename(dn_bands)
    return_image = image.addBands(DN_image)
    return return_image


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

# function to redraw subregion boundaries to bbox of glacier within them
def redraw_boundary(region):
    subset_rgi = asset_rgi01_Alaska.filterBounds(region.geometry())
    subset_bounds = subset_rgi.map( lambda f : f.setGeometry(f.geometry().bounds()) )
    subset_bounds = subset_bounds.geometry().bounds()
    
    return region.setGeometry(subset_bounds)

# rename the snow-on bands
snow_on = renameBandsS2(snow_on)

# add ndwi, ndsi to the snow_on
snow_on = add_ndwi_ndsi(snow_on)

# remake subregion geometries
asset_subregions = asset_subregions.map( lambda f : redraw_boundary(f))

#%%
    
# iterate through each sensing orbit number, sending them off to be analyzed
sons = [43,143]
sons=[43]
c=0
for sensing_orbit_number in sons:
    c+=1
    description = f'S2_{sensing_orbit_number:03d}_{date_start}_{date_end}'
    description_scl = f'S2_{sensing_orbit_number:03d}_{date_start}_{date_end}_cloud'
    folder_img = 'S2_Classified_Raw'
    folder_scl = 'S2_Cloud_Raw'

    print()
    print(f"{sensing_orbit_number}   {c} of {len(sons)}")
    
    # Load image collection which we will want to classify, filtering by cloud cover
    S2_images = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterDate(date_start, date_end) # filter by date range
                      .filterBounds(asset_simpleoutline)
                      .filter(ee.Filter.eq("SENSING_ORBIT_NUMBER",sensing_orbit_number)) # filter by this orbit number
                      .sort('system:time_start')) 
    
    # get a list of all the datatake_identifiers (individual pass overs) that there are
    dtis = S2_images.aggregate_array("DATATAKE_IDENTIFIER").distinct()
    print("Number of DTIs:", len(dtis.getInfo()))
    
    # get the rough geometry of this sensing_orbit_number (as a feature)
    son_geometry = mosaic_orbit_number_geom(sensing_orbit_number)
    
    # subset regions to only those with glacier outlines overlapping this rough geometry 
    # rgi_son = asset_rgi01_Alaska.filterBounds(son_geometry.geometry())
    subregions_of_interest = asset_subregions.filterBounds(son_geometry.geometry())
    
    # get the names/numbers of those regions, aggregate to list (to iterate), count them
    names = subregions_of_interest.aggregate_array('id').getInfo()
    # print(names)
    
    n_features = len(names)
    subregions_list = subregions_of_interest.toList(n_features+1) 
    #print(names)
    
    ### function to classify snow/firn/ice
    def classify_image(image):
        
        # rename bands, add ndwi, ndsi
        image = renameBandsS2(image) 
        image = add_ndwi_ndsi(image)
        
        # normalize by snow-on mosaic
        normalized = image.divide(snow_on) # now each will range 0 to 5+. 0=all brightness lost, 1=identical to snow, >1=brighter than snow
        
        # grab nir
        nir = normalized.select('nir')
        
        # say it is snow if nir, normalized by snow_on, is greater than 0.6
        snow = nir.gt(0.6).selfMask().rename('snow') # snow==1
        not_snow = nir.lte(0.6).selfMask().multiply(2).rename('not_snow') #not_snow==2
        
        # blend the two classification masks together
        ided = snow.blend(not_snow).rename('class')
        # ided = snow.selfMask().unmask(2)
        ided = ided.copyProperties(image)
        
        return ided 
    
    ### function to classify snow/firn/ice using random forest
    def classify_image_rf(image):
        
        # classify all the images, copying the metadata to the classification images
        # def classify_image(image):
        #     ided = image.classify(RF_classifier)
        #     ided = ided.copyProperties(image)
        #     return ided 
        
        # rename bands, add ndwi, ndsi
        image = renameBandsS2(image) 
        image = add_ndwi_ndsi(image)
        
        # add DN bands
        image = add_DN_bands(image)
        
        # classify with rf
        ided = image.classify(RF_classifier)
        
        # override areas with ndsi<0
        
        # copy properties
        ided = ided.copyProperties(image)
        
        return ided 
    
    
    ### function to grab the scene classification map from sentinel-2 SR images
    def get_SCL(image):
        scl = image.select("SCL")
        return scl
    
    
    
    
    #################################################################
    ### classify images and export to google drive, if you choose ###
    #################################################################
    if run_id_export:
        
        # apply function to classify snow/firn/ice in full image collection
        # S2_identified = S2_images.map( lambda i : classify_image(i)) 
        S2_identified = S2_images.map( lambda i : classify_image_rf(i)) 
        
        ## each image has 1=snow, 2=not snow, and then original image mask remains
        
        # define function to merge classified images that have the same DATATAKE_IDENTIFIER
        def mosaic_DTI(dti):
            
            # subset to only this dti
            subset_ic = S2_identified.filter(ee.Filter.eq("DATATAKE_IDENTIFIER",dti))
            
            # mosaic them
            mosaic = subset_ic.mosaic()
            
            # rename the band to dti value
            mosaic = mosaic.rename([ee.String(dti).slice(0,-3)])
          
            # you can copy properties over to the mosaiced image, and it will be stored in the image properties.
            mosaic = mosaic.set({"DATATAKE_IDENTIFIER":dti, "SENSING_ORBIT_NUMBER":sensing_orbit_number})
          
            return mosaic
        
        # merge classified images by dti
        s2_swaths = ee.ImageCollection(dtis.map( lambda d : mosaic_DTI(d)) )
        # print("Number of swaths made: ", len(s2_swaths.getInfo().get('features'))) 
        
        # collapse the classified image collection into a single, multi-band image
        single_image = s2_swaths.toBands()
        
        # set the list of dtis as a property, as a fallback to make sure we can trace everything back
        single_image = single_image.set({'dtis':dtis})
        
        # clip to rgi outlines
        # single_image_clipped = single_image.clipToCollection(asset_rgi01_Alaska)
        # single_image_clipped = single_image_clipped.unmask(99)
        # single_image_clipped = single_image_clipped.set({'nodata':99})
        
        # # test not clipping to rgi
        # single_image_clipped = single_image
        
        # now for each subregion or interest, clip single_image_clipped to that geometry and then export
        for i in range(0, len(names)):
            
            # if you want to just do a single region
            if names[i] in regions_override: 
                print(names[i])
            else: continue
        
            # grab this feature
            feature_i = ee.Feature(subregions_list.get(i)) 
            
            # grab geometry of the region
            region = feature_i.geometry()
        
            # clip to this geometry
            regional_clipped_image = single_image.clip(region).unmask(99) 

            # clip to rgi
            # I don't see any reason to do this, so let's not watse compute resources
            # regional_clipped_image = regional_clipped_image.clipToCollection(asset_rgi01_Alaska).unmask(98)
            
            # export the image to drive
            task = ee.batch.Export.image.toDrive(
                image = regional_clipped_image, #regional_clipped_image,
                region = region.bounds(), # region.bounds()
                folder = folder_img,
                scale = 10,
                maxPixels = int(1e13),
                crs = 'EPSG:3338',
                crsTransform = [10,0,0,0,-10,0],
                description = f"{description}_R{names[i]:02d}",
                skipEmptyTiles = True
                )
            
            task.start()
            print('Classified image stack export started', f"{description}_R{names[i]:02d}")
            
            

    ##############################################################
    ### Process SCL and export to google drive, if you choose ###
    ##############################################################
    if run_scl_export:
        # print('nah')
        # continue
        
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
            mosaic = mosaic.set({"DATATAKE_IDENTIFIER":dti, "SENSING_ORBIT_NUMBER":sensing_orbit_number})
          
            return mosaic
        
        # merge SCLs by dti
        s2_swaths_scl = ee.ImageCollection(dtis.map( lambda d : mosaic_DTI_scl(d)) )
        # print("Number of SCL swaths made: ", len(s2_swaths_scl.getInfo().get('features'))) 
        
        # collapse the SCL image collection into a single, multi-band image
        single_image = s2_swaths_scl.toBands()
        
        # set the list of dtis as a property, as a fallback to make sure we can trace everything back
        single_image = single_image.set({'dtis':dtis})
        
        # clip to rgi outlines
        # single_image_clipped = single_image.clipToCollection(asset_rgi01_Alaska).unmask(99)
        # single_image_clipped = single_image_clipped.set({'nodata':99})
        
        # now for each subregion or interest, clip single_image_clipped to that geometry and then export
        for i in range(0, len(names)):
            
            # if you want to just 
            if names[i] in regions_override: 
                print(names[i])
            else: continue
            
            # grab this feature
            feature_i = ee.Feature(subregions_list.get(i)) 
            
            # grab geometry of the region
            region = feature_i.geometry()
        
            # clip to this geometry
            regional_clipped_image = single_image.clip(region).unmask(99) 

            # clip to rgi
            regional_clipped_image = regional_clipped_image.clipToCollection(asset_rgi01_Alaska).unmask(98)
            
            # export the image to drive
            task = ee.batch.Export.image.toDrive(
                image = regional_clipped_image,
                region = region.bounds(),
                folder = folder_scl,
                scale = 10,
                maxPixels = int(1e13),
                crs = 'EPSG:3338',
                crsTransform = [10,0,0,0,-10,0],
                description = f"{description_scl}_R{names[i]:02d}",
                skipEmptyTiles = True
                )
            
            task.start()
            print('SCL image stack export started', f"{description_scl}_R{names[i]:02d}")

            
  
        
    ###########################################################
    ### save image-specific metadata to a csv if you choose ###
    ###########################################################
    if save_metadata:
        
        # create initial df with datatake_identifier, sensing_orbit_number as columns
        out_df = pd.DataFrame( {'datatake_identifier':dtis.getInfo(), 'sensing_orbit_number':sensing_orbit_number} )
        
        # add columns for satellite (a/b), date    GS[SS]_[YYYYMMDDTHHMMSS]_[RRRRRR]_N[xx.yy]
        out_df['satellite'] = [ i[:4] for i in out_df['datatake_identifier'] ]
        out_df['date'] = [ i[5:13] for i in out_df['datatake_identifier'] ]
        
        out_path = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop','AGVA','classified images','meta csv','S2',description+'.csv')
        out_df.to_csv(out_path)
        print('Metadata exported to local machine')
        #%%
        
        
    #     # now clip the single image to each of the subregions
    #     regions_single_image = asset_subregions.map( lambda r : single_image.clip(r.geometry()) ) 
        
        
        
        
    #     #%%
        
        
    #     # clip image to RGI outlines only, then set nans
    #     single_image_clipped = single_image.clipToCollection(asset_rgi01_Alaska)
    #     single_image_clipped = single_image_clipped.unmask(99)
    #     single_image_clipped = single_image_clipped.set({'nodata':99})
        
    #     # set the list of dtis as a property, as a fallback to make sure we can trace everything back
    #     single_image_clipped = single_image_clipped.set({'dtis':dtis})
    
    
    #     ##### this is the point where we are going to clip it to subregions before exporting
        
    #     # # subset subregions collection to only regions that area within this swath
    #     subregions_of_interest = asset_subregions.filterBounds(single_image.geometry())
        
        
    #     # filterBounds was throwing a user memory limit error, so let's just 
    #     # run it for all of the subregions
    #     subregions_of_interest = asset_subregions
        
    #     # get list of which regions remain
    #     names = subregions_of_interest.aggregate_array('id').getInfo()
    #     print(names)
        
    #     # check how many images this is
    #     n_features = len(names)
        
    #     # convert subregion featurecollection to list of features
    #     subregions_list = subregions_of_interest.toList(n_features+1) 
        
    #     # now for each subregion remaining, clip single_image_clipped to that geometry and then export
    #     for i in range(0, n_features):
        
    #         # grab this feature
    #         feature_i = ee.Feature(subregions_list.get(i)) 
            
    #         # clip to this geometry
    #         regional_clipped_image = single_image_clipped.clip(feature_i.geometry())
            
    #         # export the image to drive
    #         task = ee.batch.Export.image.toDrive(
    #             image = regional_clipped_image,
    #             region = feature_i.geometry(),   # should clip this to the asset_simpleoutline to get smaller files
    #             folder = folder_img,
    #             scale = 10,
    #             maxPixels = int(1e13),
    #             crs = 'EPSG:3338',
    #             crsTransform = [10,0,0,0,-10,0],
    #             description = f"{description}_R{names[i]:02d}",
    #             skipEmptyTiles = True
    #             )
            
    #         # # export the image to drive
    #         # task = ee.batch.Export.image.toDrive(
    #         #     image = single_image_clipped,
    #         #     region = S2_images.geometry().intersection(asset_simpleoutline).bounds(),   # should clip this to the asset_simpleoutline to get smaller files
    #         #     folder = folder_img,
    #         #     scale = 10,
    #         #     maxPixels = int(1e13),
    #         #     crs = 'EPSG:3338',
    #         #     crsTransform = [10,0,0,0,-10,0],
    #         #     description = description,
    #         #     skipEmptyTiles = True
    #         #     )
        
    #         task.start()
    #         print('Classified image stack export started')
    
    
    
    # #### processing of SCL (ESA-provided scene classification and cloud id)
    # if run_scl_export:
        
    #     # apply function to grab SCL from each image
    #     S2_SCLs = S2_images.map( lambda i : get_SCL(i))
        
    #     # function to merge SCLs that have the same DATATAKE_IDENTIFIER
    #     def mosaic_DTI_scl(dti):
    #         subset_ic = S2_SCLs.filter(ee.Filter.eq("DATATAKE_IDENTIFIER",dti))
    #         mosaic = subset_ic.mosaic()
          
    #         # you can copy properties over to the mosaiced image, and it will be stored in the image properties.
    #         # date, dti, sensing_orbit_number, satellite a/b
    #         mosaic = mosaic.set({"DATATAKE_IDENTIFIER":dti, "SENSING_ORBIT_NUMBER":sensing_orbit_number})
          
    #         return mosaic
        
    #     # merge SCLs by dti
    #     s2_swaths_scl = ee.ImageCollection(dtis.map( lambda d : mosaic_DTI_scl(d)) )
    #     # print("Number of SCL swaths made: ", len(s2_swaths_scl.getInfo().get('features'))) 
        
    #     # collapse the SCL image collection into a single, multi-band image
    #     single_image = s2_swaths_scl.toBands()
        
    #     # clip image to RGI outlines only, then set nans
    #     single_image_clipped = single_image.clipToCollection(asset_rgi01_Alaska)
    #     single_image_clipped = single_image_clipped.unmask(99)
    #     single_image_clipped = single_image_clipped.set({'nodata':99})
        
    #     # set the list of dtis as a property, as a fallback to make sure we can trace everything back
    #     single_image_clipped = single_image_clipped.set({'dtis':dtis})
    
    #     # export to drive
    #     task = ee.batch.Export.image.toDrive(
    #         image = single_image_clipped,
    #         region = S2_images.geometry().intersection(asset_simpleoutline).bounds(),   # should clip this to the asset_simpleoutline to get smaller files
    #         folder = folder_scl,
    #         scale = 10,
    #         maxPixels = int(1e13),
    #         crs = 'EPSG:3338',
    #         crsTransform = [10,0,0,0,-10,0],
    #         description = description_scl,
    #         skipEmptyTiles = True
    #         )
    
    #     task.start()
    #     print('Classified image stack export started')
    
    # ### save image-specific metadata to a csv if you choose
    # if save_metadata:
        
    #     # create initial df with datatake_identifier, sensing_orbit_number as columns
    #     out_df = pd.DataFrame( {'datatake_identifier':dtis.getInfo(), 'sensing_orbit_number':sensing_orbit_number} )
        
    #     # add columns for satellite (a/b), date    GS[SS]_[YYYYMMDDTHHMMSS]_[RRRRRR]_N[xx.yy]
    #     out_df['satellite'] = [ i[:4] for i in out_df['datatake_identifier'] ]
    #     out_df['date'] = [ i[5:13] for i in out_df['datatake_identifier'] ]
        
    #     out_path = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop','AGVA','classified images','meta csv','S2',description+'.csv')
    #     out_df.to_csv(out_path)
    #     print('Metadata exported to local machine')
        
        
        
        
        
        
        
        
        
        
        
        
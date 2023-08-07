# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 18:24:38 2023

@author: lzell
"""

import os
import rasterio as rio
import numpy as np
import shapely
import pyproj
import geopandas as gpd
from rasterio.merge import merge as riomerge
from rasterio.plot import show as rioshow
import matplotlib.pyplot as plt
from rasterio.warp import calculate_default_transform, reproject, Resampling
from osgeo import gdal

# set folder paths, etc...
folder_AGVA = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA")
path_regions = os.path.join(folder_AGVA, "RGI", "S2_subregions", "subregions.shp")

# open subregions shapefile
regions_gdf = gpd.read_file(path_regions)

#%%

### first we are going to reproject every one of the hugonnet thinning images
all_ranges = [[2000,2005], [2005,2010], [2010,2015], [2015,2020]]

for a in all_ranges:    
    
    # set path to the folder holding all the files you want
    years_to_do = f"01_02_rgi60_{a[0]}-01-01_{a[1]}-01-01"
    folder_thinning = os.path.join(folder_AGVA, "DEMs","Hugonnet_thinning", years_to_do, 'dhdt')
    
    # list out the image names in the folder
    all_image_names = os.listdir(folder_thinning)
    
    # get the bounding box for each of the images
    bounding_boxes = []
    
    for i in range(len(all_image_names)):
        dataset = rio.open(os.path.join(folder_thinning, all_image_names[i]))
        (left, bottom, right, top) = dataset.bounds
        
        # create shapely geometry from bounds  shapely.geometry.box(minx, miny, maxx, maxy, ccw=True)
        bounding_box = shapely.geometry.box(left, bottom, right, top)
        
        # reproject to epsg:4326
        target_crs = pyproj.CRS('EPSG:4326')
        initial_crs = pyproj.CRS(dataset.crs)
        
        project = pyproj.Transformer.from_crs(initial_crs, target_crs, always_xy=True).transform
        bounding_box_reproj = shapely.ops.transform(project, bounding_box)
    
        bounding_boxes.append(bounding_box_reproj)
    
    # put the image names and boxes into a geopandas gdf
    all_images_gdf = gpd.GeoDataFrame({'name':all_image_names, 'geometry':bounding_boxes}, crs='EPSG:4326')
    
    # quick image to make sure this looks right
    # fig,axs = plt.subplots()
    # all_images_gdf.boundary.plot(ax=axs)
    # regions_gdf.boundary.plot(ax=axs, color='black')
    
    # so now we have a list of all the images
    # open each and reproject to alaska albers
    image_list = [os.path.join(folder_thinning, i) for i in all_images_gdf['name']]
    output_list = [os.path.join(folder_AGVA, "DEMs","Hugonnet_thinning", 'Alaska Albers', years_to_do, 'dhdt', i)
                   for i in all_images_gdf['name']]
    
    for i in range(len(image_list)):
        # if i>1:continue
        input_raster = gdal.Open(image_list[i])
        output_raster = output_list[i]
        warp = gdal.Warp(output_raster,input_raster,dstSRS='EPSG:3338')
        warp = None 
        input_raster = None

    ### then for each subregion, go through and mosaic the reprojected images that overlaps them
    for i in regions_gdf['id']:
        
        # print to track progress
        print(years_to_do, i)
        
        subregion = i
        subset_regions = regions_gdf[regions_gdf['id']==subregion]
        
        # then subset to images that overlap this
        subset_images = all_images_gdf[all_images_gdf.intersects(subset_regions.geometry.values[0])]
        
        # create image paths from names
        subset_paths = [os.path.join(folder_AGVA, "DEMs","Hugonnet_thinning", 'Alaska Albers', years_to_do, 'dhdt', i)
                       for i in subset_images['name']]
        
        # now go through and use gdal.translate to merge all of them
        out_path = os.path.join(folder_AGVA, "DEMs","Hugonnet_thinning", 'Alaska Albers', years_to_do, 'dhdt', f"Region_{subregion:02d}.tif")
        
        vrt = gdal.BuildVRT("merged.vrt", subset_paths)
        translated = gdal.Translate(out_path, vrt, outputType=gdal.GDT_Float32, creationOptions = ['PREDICTOR=2','COMPRESS=LZW'])
        vrt = None
        translated = None

#%%

# function to reproject and align an input image with another 'match' image
def reproj_match(infile, match, outfile):
    """Reproject a file to match the shape and projection of existing raster. 
    
    Parameters
    ----------
    infile : (string) path to input file to reproject
    match : (string) path to raster with desired shape and projection 
    outfile : (string) path to output file tif
    """
    # open input
    with rio.open(infile) as src:
        #src_transform = src.transform
        
        # open input to match
        with rio.open(match) as match:
            dst_crs = match.crs
            
            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,     # input CRS
                dst_crs,     # output CRS
                match.width,   # input width
                match.height,  # input height 
                *match.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
            )

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": 0})
        
        # open output
        with rio.open(outfile, "w", **dst_kwargs) as dst:
            # iterate through bands and write using reproject function
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.cubic)
    return rio.open(outfile).read(1)

#%%

# set folder paths, etc...
folder_dhdt = os.path.join(folder_AGVA, 'DEMs', 'Hugonnet_thinning', 'Alaska Albers')

# we want to do this for the 10m and 30m dems
scales = [10]
for s in scales:
    folder_dem = os.path.join(folder_AGVA, "DEMs", f"{s}m_COP_GLO30")
    
    ### now for every region in each year range, go through and reproject/match the mosaic to
    ### the original dem
    for a in all_ranges:    
        
        # set path to the folder holding all the files you want
        years_to_do = f"01_02_rgi60_{a[0]}-01-01_{a[1]}-01-01"
        
        # go through each subregion
        for i in regions_gdf['id']:
            
            # print to track progress
            print(years_to_do, i)
            subregion = i
            
            if i in [9,10]:
                print('good')
            else: continue
    
            # set path to the file of the original dem
            path_dem_original = os.path.join(folder_dem, f"Region_{subregion:02d}.tif")
    
            # set path to the file of the dhdt mosaic covering this region
            path_dhdt_original = os.path.join(folder_dhdt, years_to_do, 'dhdt', f"Region_{subregion:02d}.tif")
            
            # set destination file path
            path_output = os.path.join(folder_AGVA, 'DEMs', "10m_thinning", years_to_do, "dhdt", f"Region_{subregion:02d}.tif")
            
            # send these off to the above function
            test = reproj_match(path_dhdt_original, path_dem_original, path_output)

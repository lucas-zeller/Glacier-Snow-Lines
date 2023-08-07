# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:05:51 2022

@author: lzell
"""

import os
import rasterio as rio
import numpy as np
from rasterio.merge import merge as riomerge
from rasterio.plot import show as rioshow
import matplotlib.pyplot as plt
from osgeo import gdal
import glob
import subprocess
import geopandas as gpd
import pandas as pd
from rasterio.mask import mask
from datetime import datetime
import shapely
from shapely import geometry
from mpl_toolkits.axes_grid1 import make_axes_locatable
import multiprocessing

# iterate through every row/path image
# iterate through each glacier in that image
# for each date, calculate area of firn, snow, ice, and cloud mask
# save the ~74 long csv with columns of: glacier ID, row, path, snow pixels, firn pixels, ice pixels, cloud pixels, glacier area,
#%%
# base folder path
agva_folder = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop','AGVA')
os.chdir(agva_folder)

# folders holding the images (PC vs non-PC)
images_folder = os.path.join(agva_folder, 'classified images', 'L8 Classified Images PC')
#images_folder = os.path.join(agva_folder, 'classified images', 'L8 Classified Images no PC')

# csv folder
csv_folder = os.path.join(images_folder, 'csvs')

# load rgi outlines
rgi_path = os.path.join(agva_folder, 'RGI', '01_rgi60_Alaska', '01_rgi60_Alaska.shp')
rgi_gdf = gpd.read_file(rgi_path)

# filter to only glacier >0.5 km2
rgi_gdf = rgi_gdf[rgi_gdf.Area >= 0.5]

# load csv that has the row/path combinations
rp_fp = os.path.join(agva_folder,'AGVA rows and paths.csv')
rp_df = pd.read_csv(rp_fp)

# rp_df = rp_df[rp_df['row']==18]
# rp_df = rp_df[rp_df['path']==68]

#%%

# function that will be called
def analyze_images(index,rgi,image_path, cloud_path, meta_path, wrs_row, wrs_path):

    # open image, clipping it to the RGI glacier outline
    with rio.open(image_path) as src:
        image_clip, out_transform = mask(dataset=src, shapes=[rgi.geometry], crop=True, nodata=99)
        out_meta = src.meta
        
    image = image_clip.astype(float)
    image[image==99] = np.nan
    
    # find area of the image, in km2
    area_found = np.count_nonzero(~np.isnan(image[0,:,:]))*30*30 / (1000*1000)
    area_rgi = rgi.Area
    #print(area_found)
    ### if the area found is <80% of the rgi area, skip it
    if area_found < area_rgi*0.8: 
        for y in range(2015,2022):
            rgi_image.loc[index,"min "+str(y)] = np.nan
            rgi_image.loc[index,"min date "+str(y)] = np.nan
            rgi_image.loc[index,"min aar "+str(y)] = np.nan
        return 0
    
    # open cloud image, clipping to the RGI outline
    # 1=usable data, 0=unusable or off glacier 
    with rio.open(cloud_path) as src:
        cloud_image_clip, cloud_out_transform = mask(dataset=src, shapes=[rgi.geometry], crop=True, nodata=99)
        out_meta = src.meta
    cloud_free_pixels = np.sum(cloud_image_clip==1, axis=(1,2))
    cloud_free_area = cloud_free_pixels*30*30 / (1000*1000)
    
    ### otherwise, calculate the are of firn, snow, and ice in each year (each band)
    
    # open metadata as csv
    img_data = pd.read_csv(meta_path)
    img_data['date'] = img_data.apply(lambda row: datetime.strptime(str(row.id[-8:]), '%Y%m%d'), axis=1)
    img_data['year'] = img_data.apply(lambda row: row.date.year, axis=1)
    #img_data = img_data[img_data['year']>=2015]
    
    # make copy df to store per-image info
    rgi_data = img_data.copy()
    
    # calculate area of snow, firn and ice in each image, add to df
    rgi_data['snow pixels'] = np.sum(image==0, axis=(1,2))
    rgi_data['firn pixels'] = np.sum(image==1, axis=(1,2))
    rgi_data['ice pixels'] = np.sum(image==2, axis=(1,2))
    rgi_data['cloudfree pixels'] = cloud_free_pixels
    
    # add rgi area to df
    rgi_data['glacier area'] = rgi.Area
    rgi_data['RGIId'] = rgi.RGIId
    
    
    ### new from here
    # save the df as a csv, in the correct folder with correct name
    csv_fp = os.path.join(csv_folder, rgi.RGIId+".csv")
    
    # if it doesn't exist, simply save this file
    if not os.path.exists(csv_fp):
        rgi_data.to_csv(csv_fp, index=False)
    
    # if it does exist, open the pre-existing csv, add rows, and then save
    else:
        #print('double',csv_fp)
        # open master df
        master_df = pd.read_csv(csv_fp)
        
        # drop unnamed columns
        master_df.drop(master_df.columns[master_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    
        # concatenate all the rows together
        new_df = pd.concat([master_df,rgi_data], ignore_index=True)
        
        # resave
        new_df.to_csv(csv_fp, index=False)

    
    # folder_name = "P{}_R{}".format(wrs_path, wrs_row)
    # file_name = "P{}_R{}_{}.csv".format(wrs_path, wrs_row, rgi.RGIId)
    
    # folder_path = os.path.join(images_folder, 'Glacier csvs', folder_name)
    # out_name = os.path.join(folder_path, file_name)
    
    # # check if folder exists
    # if os.path.exists(folder_path):
    #     rgi_data.to_csv(out_name)
    # else:
    #     os.makedirs(folder_path)
    #     rgi_data.to_csv(out_name)
    
    # #print("saved to",out_name)
    return 1

#%%
# iterate through each row/path, sending them off to be analyzed
c=0
for i,row in rp_df.iterrows():
    
    # options to only test out one or two images
    #if c>3: continue
    
    # print statement to track progress and keep you sane while it's running
    c+=1
    print(c,"of",len(rp_df))
    
    wrs_row = int(row['row'])
    wrs_path = int(row['path'])
    
    # load image name and path
    img_name = "P{}_R{}_2013-01-01_2021-12-30_90".format(wrs_path, wrs_row)
    image_path = os.path.join(images_folder, img_name+".tif")
    print(img_name)
    
    # load cloud mask image
    cloud_name = "P{}_R{}_2013-01-01_2021-12-30_90_cloud.tif".format(wrs_path, wrs_row)
    cloud_path = os.path.join(agva_folder, 'classified images', 'L8 Cloud', cloud_name)
    
    # load image metadata
    meta_name = "P{}_R{}_2013-01-01_2021-12-30_90".format(wrs_path, wrs_row)
    meta_path = os.path.join(agva_folder, 'classified images', 'meta csv', meta_name+'.csv')

    # open rio image object, get crs
    image_rio = rio.open(image_path)
    image_crs = image_rio.crs
    image_bounds = image_rio.bounds
    
    # make image bounds into a shapely object (to filter the outlines shapefile with)
    image_bbox = geometry.box(*image_bounds)
    xmin,ymin,xmax,ymax = (image_bounds[0],image_bounds[1],image_bounds[2],image_bounds[3])
    
    # reproject outlines to the image crs
    rgi_gdf_crs = rgi_gdf.to_crs(image_crs)
    
    # filter outlines to those that overlap our image
    rgi_image = rgi_gdf_crs.cx[xmin:xmax, ymin:ymax].copy()
    print(len(rgi_image), " glaciers >0.5 km2 are possibly in this image")
    
    # send each outline along with the image to be analyzed
    c2=0
    for index, rgi in rgi_image.iterrows(): 
        r = analyze_images(index, rgi, image_path, cloud_path, meta_path, wrs_row, wrs_path)
        c2+=r
    print(c2, " glaciers >0.5 km2 are actually in this image")
    print()

#%%
# lastly, iterate through all the saved csvs, sorting them by date
print("Reordering by date")            
for master_file in os.scandir(csv_folder):
    master_df = pd.read_csv(master_file)
    # drop unnamed columns
    master_df.drop(master_df.columns[master_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    master_df.sort_values('date',inplace=True)
    master_df.to_csv(master_file, index=False)

#%%
# # add empty columns to df
# for y in range(2015,2022):
#     rgi_image["min "+str(y)] = np.nan
#     rgi_image["min aar "+str(y)] = np.nan
#     rgi_image["min date "+str(y)] = np.nan

# def analyze_images(input_list): # input = [index, row]
#     index = input_list[0]
#     rgi = input_list[1]
    
#     with rio.open(image_path) as src:
#         image_clip, out_transform = mask(dataset=src, shapes=[rgi.geometry], crop=True, nodata=99)
#         out_meta = src.meta
        
#     image = image_clip.astype(float)
#     image[image==99] = np.nan
    
#     # find area of the image, in km2
#     area_found = np.count_nonzero(~np.isnan(image[0,:,:]))*30*30 / (1000*1000)
#     area_rgi = rgi.Area
    
#     ### if the area found is <80% of the rgi area, skip it
#     if area_found < area_rgi*0.8: 
#         for y in range(2015,2022):
#             rgi_image.loc[index,"min "+str(y)] = np.nan
#             rgi_image.loc[index,"min date "+str(y)] = np.nan
#             rgi_image.loc[index,"min aar "+str(y)] = np.nan
#         return [index, np.nan, np.nan, np.nan]
    
#     ### otherwise, calculate the minimum accumulation area in each year
    
#     # make copy df to store per-image info
#     rgi_data = img_data.copy()
    
#     # calculate area of snow, firn and ice in each image
#     rgi_data['area_snow'] = np.sum(image==0, axis=(1,2)) * 30 * 30 / 1000000 
#     rgi_data['area_firn'] = np.sum(image==1, axis=(1,2)) * 30 * 30 / 1000000 
#     rgi_data['area_ice'] = np.sum(image==2, axis=(1,2)) * 30 * 30 / 1000000 
    
#     rgi_data['area_snow_rel'] = np.sum(image==0, axis=(1,2)) * 30 * 30 / (1000000*area_rgi) 
#     rgi_data['area_firn_rel'] = np.sum(image==1, axis=(1,2)) * 30 * 30 / (1000000*area_rgi)
#     rgi_data['area_ice_rel'] = np.sum(image==2, axis=(1,2)) * 30 * 30 / (1000000*area_rgi)

#     # option to filter by cloudy percentage if you want
#     df_to_use = rgi_data[rgi_data['cloud_cover_land']<100]
    
#     # calculate minimum each year
#     years = [2015,2016,2017,2018,2019,2020,2021]
#     dfs_by_year = []
#     max_n = 0
    
#     for y in years:
#         df_sub = df_to_use[df_to_use.year == y]
#         dfs_by_year.append(df_sub)
#         if df_sub.shape[0]>max_n: max_n=df_sub.shape[0]
    
#     # find minimum accumulation area in each year
#     min_aa = []
#     min_aa_date = []
    
#     y=2015
#     for df in dfs_by_year:
#         areas = df['area_snow']+df['area_firn']
#         areas_rel = df['area_snow_rel']+df['area_firn_rel']
        
#         rgi_image.loc[index, "min "+str(y)] = np.nanmin(areas)
#         rgi_image.loc[index, "min aar "+str(y)] = np.nanmin(areas_rel)
#         rgi_image.loc[index, "min date "+str(y)] = max(df[df['area_snow']+df['area_firn']==np.nanmin(areas)].date)
#         y+=1
#         #return([index, np.nanmin(areas), np.nanmin(areas_rel), max(df[df['area_snow']+df['area_firn']==np.nanmin(areas)].date)])
#         # min_aa.append(np.nanmin(areas))
#         # min_aa_date.append(max(df[df['area_snow']+df['area_firn']==np.nanmin(areas)].date))

# # test out multithreading
# multithread = 0

# if multithread == 0:
      
#     # loop through each geometry
#     c=0
#     total = len(rgi_image)
    
#     for index, rgi in rgi_image.iterrows():
#         if c%10==0:
#             print(c,'out of',total,'analyzed')
#         c+=1
        
#         analyze_images([index,rgi])
        
#         # with rio.open(image_path) as src:
#         #     image_clip, out_transform = mask(dataset=src, shapes=[rgi.geometry], crop=True, nodata=99)
#         #     out_meta = src.meta
            
#         # image = image_clip.astype(float)
#         # image[image==99] = np.nan
        
#         # # find area of the image, in km2
#         # area_found = np.count_nonzero(~np.isnan(image[0,:,:]))*30*30 / (1000*1000)
#         # area_rgi = rgi.Area
        
#         # ### if the area found is <80% of the rgi area, skip it
#         # if area_found < area_rgi*0.8: 
#         #     for y in range(2015,2022):
#         #         rgi_image.loc[index,"min "+str(y)] = np.nan
#         #         rgi_image.loc[index,"min date "+str(y)] = np.nan
#         #         rgi_image.loc[index,"min aar "+str(y)] = np.nan
#         #     continue
        
#         # ### otherwise, calculate the minimum accumulation area in each year
        
#         # # make copy df to store per-image info
#         # rgi_data = img_data.copy()
        
#         # # calculate area of snow, firn and ice in each image
#         # rgi_data['area_snow'] = np.sum(image==0, axis=(1,2)) * 30 * 30 / 1000000 
#         # rgi_data['area_firn'] = np.sum(image==1, axis=(1,2)) * 30 * 30 / 1000000 
#         # rgi_data['area_ice'] = np.sum(image==2, axis=(1,2)) * 30 * 30 / 1000000 
        
#         # rgi_data['area_snow_rel'] = np.sum(image==0, axis=(1,2)) * 30 * 30 / (1000000*area_rgi) 
#         # rgi_data['area_firn_rel'] = np.sum(image==1, axis=(1,2)) * 30 * 30 / (1000000*area_rgi)
#         # rgi_data['area_ice_rel'] = np.sum(image==2, axis=(1,2)) * 30 * 30 / (1000000*area_rgi)
    
#         # # option to filter by cloudy percentage if you want
#         # df_to_use = rgi_data[rgi_data['cloud_cover_land']<100]
        
#         # # calculate minimum each year
#         # years = [2015,2016,2017,2018,2019,2020,2021]
#         # dfs_by_year = []
#         # max_n = 0
        
#         # for y in years:
#         #     df_sub = df_to_use[df_to_use.year == y]
#         #     dfs_by_year.append(df_sub)
#         #     if df_sub.shape[0]>max_n: max_n=df_sub.shape[0]
        
#         # # find minimum accumulation area in each year
#         # min_aa = []
#         # min_aa_date = []
        
#         # y=2015
#         # for df in dfs_by_year:
#         #     areas = df['area_snow']+df['area_firn']
#         #     areas_rel = df['area_snow_rel']+df['area_firn_rel']
            
#         #     rgi_image.loc[index, "min "+str(y)] = np.nanmin(areas)
#         #     rgi_image.loc[index, "min aar "+str(y)] = np.nanmin(areas_rel)
#         #     rgi_image.loc[index, "min date "+str(y)] = max(df[df['area_snow']+df['area_firn']==np.nanmin(areas)].date)
            
#         #     # min_aa.append(np.nanmin(areas))
#         #     # min_aa_date.append(max(df[df['area_snow']+df['area_firn']==np.nanmin(areas)].date))
            
#         #     y+=1



# else:
#     # make a list so that you can use mulitprocessing
#     list_of_rows = []
#     # for index, rgi in rgi_image.iterrows():
#     #         list_of_rows.append([index,rgi])  

#     # if __name__ == '__main__':
#     #     pool = multiprocessing.Pool(4)
#     #     result = pool.map(analyze_images, list_of_rows)
        
#     #     for r in result:
#     #         rgi_image.loc[r[0], "min "+str(y)] = np.nanmin(areas)
#     #         rgi_image.loc[r[0], "min aar "+str(y)] = np.nanmin(areas_rel)
#     #         rgi_image.loc[r[0], "min date "+str(y)] = 0
#     #rgi_image.apply(lambda row: analyze_images_lambda(row), axis=1)

# #%%
# # per-glacier mean and median aar of all years
# rgi_image['mean aar'] = rgi_image.apply(lambda row: np.nanmean([row['min aar 2015'],
#                                                                 row['min aar 2016'],
#                                                                 row['min aar 2017'],
#                                                                 row['min aar 2018'],
#                                                                 row['min aar 2019'],
#                                                                 row['min aar 2020'],
#                                                                 row['min aar 2021'],
#                                                                 ]), axis=1)

# rgi_image['median aar'] = rgi_image.apply(lambda row: np.nanmedian([row['min aar 2015'],
#                                                                     row['min aar 2016'],
#                                                                     row['min aar 2017'],
#                                                                     row['min aar 2018'],
#                                                                     row['min aar 2019'],
#                                                                     row['min aar 2020'],
#                                                                     row['min aar 2021'],
#                                                                     ]), axis=1)

# # per year variation from median
# for y in years:
#     rgi_image[str(y)+' aar variation'] = rgi_image.apply(lambda row: row['min aar '+str(y)] - row['median aar'], axis=1)

# #%%
# # calculate mean of each column each year
# means = []
# medians = []
# aar_vars = []

# for y in years:
#     mea = np.nanmean(rgi_image["min aar "+str(y)].astype(float))
#     med = np.nanmedian(rgi_image["min aar "+str(y)].astype(float))
#     var = np.nanmean(rgi_image[str(y)+' aar variation'].astype(float))
#     means.append(mea)
#     medians.append(med)
#     aar_vars.append(var)

# #%%
# # scatterplot showing the average AAR of all glaciers plotted against the wolverine ELAs
# fig,ax = plt.subplots(1,2, figsize=(8,3))
# ax[0].scatter(means,wolv_ela*-1,marker='X', c='tab:red', s=50)
# ax[1].scatter(medians,wolv_ela*-1,marker='X', c='tab:red', s=50)

# ax[0].set_xlabel('Mean AAR')
# ax[1].set_xlabel('Median AAR')
# ax[0].set_ylabel('Wolv ELA')
# ax[1].set_ylabel('Wolv ELA')

# plt.tight_layout() 

# #%%
# # scatterplot showing the average AAR variation plotted against wolverine ELAs
# fig,ax = plt.subplots( figsize=(4,3))
# ax.scatter(aar_vars,wolv_ela*-1,marker='X', c='tab:red', s=50)

# ax.set_xlabel('AAR variation')
# ax.set_ylabel('Wolv ELA')

# plt.tight_layout() 

# #%%
# # map showing the outlines colored by mean aar of all years
# fig,ax = plt.subplots(figsize=(10,8))
# rgi_image.plot(ax=ax, column="median aar", legend=True, cmap='viridis', vmin=0, vmax=1, legend_kwds={'label': "Average AAR 2015-2021"})

# #%%
# # maps showing the annual variation of each glacier each year
# fig,axs = plt.subplots(3,3, figsize=(12,12), sharex=True, sharey=True)
# c=0
# for y in years:
#     a = axs[c//3,c%3]
    
#     if y==2021: #add with colorbar
#         divider = make_axes_locatable(axs[2,1])
#         cax = divider.append_axes("left", size="5%", pad=0.1)
#         rgi_image.plot(ax=a, column=str(y)+' aar variation', cmap='coolwarm_r', vmin=-0.3, vmax=0.3, legend=True, cax=cax)
#     else:
#         rgi_image.plot(ax=a, column=str(y)+' aar variation', cmap='coolwarm_r', vmin=-0.3, vmax=0.3)
    
#     a.set_title(y)
#     a.axis('off')
#     c+=1
# axs[2,1].axis('off')
# axs[2,2].axis('off')
# plt.tight_layout()
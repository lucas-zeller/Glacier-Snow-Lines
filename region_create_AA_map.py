# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:05:09 2022

@author: lzell
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from datetime import datetime
import matplotlib as mpl
import scipy.stats
import rasterio as rio
import pyproj
from shapely.ops import transform
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling, calculate_default_transform

# load csv of annual AA each year for all the glaciers
# subset to glaciers within an AOI that is defined by a shapefile
# for each glacier within that subset, open and clip each annual AA image to glacier extent
# take per-pixel mode
# save the image as a single-band geotiff in a new folder

# set folder/file path
agva_folder = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop','AGVA')
imgs_folder = os.path.join(agva_folder, "classified images", "L8 Classified Images PC")
meta_folder = os.path.join(agva_folder, "classified images", "meta csv")
data_path = os.path.join(agva_folder, 'AA observations csv', 'PC', 'annual_minimums_temp.csv')
rgi_path = os.path.join(agva_folder, 'RGI', '01_rgi60_Alaska', '01_rgi60_Alaska.shp')
aoi_folder = os.path.join(agva_folder, 'aois')
csv_folder = os.path.join(agva_folder, 'AA observations csv', 'PC')
temp_folder = os.path.join(agva_folder, 'Regional Images temp folder')
os.chdir(agva_folder)

# open rgi shapefile and observed AA datafile
AAs_df = pd.read_csv(data_path)
rgi_gdf = gpd.read_file(rgi_path)#.set_index('RGIId')

# select only rows in rgi_gdf that are in AAs_df
rgi_gdf = rgi_gdf[rgi_gdf['RGIId'].isin(AAs_df['RGIId'])]

# merge the two dataframes
rgi_gdf = rgi_gdf.merge(AAs_df, on='RGIId')

# background image
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
outlines_bad = world[world['name'].isin(['United States of America', 'Canada'])]
#states_gdf = gpd.read_file('http://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_5m.json')
states_gdf = gpd.read_file(os.path.join(agva_folder,'alaska_outline.json'))
outlines = states_gdf[states_gdf['NAME'].isin(['Alaska'])]
new_outline = gpd.read_file(os.path.join(agva_folder,'us_outline_20m.json'))

# reproject to better crs
rgi_gdf = rgi_gdf.to_crs(epsg=3338)
outlines = outlines.to_crs(epsg=3338)

#%%
# open aoi file
aoi = gpd.read_file(os.path.join(aoi_folder, 'glacier_bay', 'POLYGON.shp')).to_crs(epsg=3338)

# subset rgi_gdf to aoi
rgi_gdf = rgi_gdf[rgi_gdf.within(aoi.geometry[0])]

# filter out obs where we had 0 observations
rgi_gdf = rgi_gdf.replace(-1,np.nan)
rgi_gdf = rgi_gdf.replace('-1',np.nan)

# do some further calculations
aar_cols = ['min_AAR_2013', 'min_AAR_2014','min_AAR_2015', 'min_AAR_2016', 'min_AAR_2017', 'min_AAR_2018', 'min_AAR_2019', 'min_AAR_2020', 'min_AAR_2021']
aa_cols = ['min_AA_2013', 'min_AA_2014','min_AA_2015', 'min_AA_2016', 'min_AA_2017', 'min_AA_2018', 'min_AA_2019', 'min_AA_2020', 'min_AA_2021']

# add doy as column
def get_doy(row, year):
    date = str(row['min_date_'+str(y)])
    if date == '-1': 
        doy = np.nan
    elif date == 'nan':
        doy = np.nan
    else:
        date = date[:10]
        date_format = datetime.strptime(date, '%Y-%m-%d')
        doy = date_format.timetuple().tm_yday
    return doy
    
for y in range(2013,2022):
    rgi_gdf['doy'+str(y)] = rgi_gdf.apply(lambda row: get_doy(row, y), axis=1)
    
# optionally, filter out glaicer years where we had only 1 observation (n_obs)
def remove_1s(row, year):
    if row["n_obs_"+str(year)]==1:
        row['min_AAR_'+str(year)] = np.nan
        row['min_AA_'+str(year)] = np.nan
    return row

# for y in range(2013,2022):
#     rgi_gdf = rgi_gdf.apply(lambda row: remove_1s(row, y), axis=1)

rgi_gdf['mean_AAR'] = rgi_gdf[aar_cols].mean(axis=1)
rgi_gdf['mean_AA'] = rgi_gdf[aa_cols].mean(axis=1)

# add doy as column
for y in range(2013,2022):
    rgi_gdf['doy'+str(y)] = rgi_gdf.apply(lambda row: get_doy(row, y), axis=1)

#%%
# annual variation from mean AAR for each glacier, each year
fig,axs = plt.subplots(3,3, figsize=(15,9), sharex=True, sharey=True)

# rgi_gdf['centroid_column'] = rgi_gdf.centroid
# rgi_gdf = rgi_gdf.set_geometry('centroid_column')

n=0
for y in range(2013,2022):

    r=n//3
    c=n%3
    print(y,r,c)
    ax = axs[r,c]
    
    subset = rgi_gdf#[rgi_gdf['O2Region']=='2']
    color_col = subset['min_AAR_'+str(y)]-subset['mean_AAR']
    #color_col = rgi_gdf['doy'+str(y)]
    
    outlines.boundary.plot(ax=ax, color='0.7', alpha=1, zorder=-1)    
    subset.plot(ax=ax, column=color_col, alpha=0.9, legend=False, cmap='coolwarm_r', vmin=-20, vmax=20)
    
    ax.set_title(str(y)+"  ", loc='right', y=1.0, pad=-20, fontsize=18)
    ax.set_xlim(-1.39e+5, 1.59e+6)
    ax.set_ylim(7.25e+5, 1.627e+6)
    ax.set_xticks([])
    ax.set_yticks([])
    
    n+=1

plt.tight_layout()

#%%
# now we need a function that take a glacier outline and date, goes to find the classified image
# taken on that date, and clips it to the glacier outline

def clip_to_rgi(image_id, rgi_geom, rgiid):
    
    # define image path
    path = image_id[-14:-12]
    row = image_id[-11:-9]    
    img_path = os.path.join(imgs_folder, "P{}_R{}_2013-01-01_2021-12-30_90.tif".format(path,row))
    #print(img_path)
    
    # find what band number it is supposed to be
    meta_csv = pd.read_csv(os.path.join(meta_folder, "P{}_R{}_2013-01-01_2021-12-30_90.csv".format(path,row)))
    meta_row = meta_csv[meta_csv['id'] == image_id]
    band = meta_row.index[0]
    #print(band)
    
    # find the image crs
    with rio.open(img_path) as src:
        img_crs = src.crs
        print(img_crs)
    if img_crs != 'EPSG:32608':
        return 0

    # reproject the glacier outline to the correct crs
    # initiate the transformer
    project = pyproj.Transformer.from_proj(
        pyproj.Proj('epsg:3338'), # source coordinate system
        pyproj.Proj(img_crs)) # destination coordinate system
    
    # reproject outline
    rgi_reproj = transform(project.transform, rgi_geom)
    
    # clip image to outline
    with rio.open(img_path) as src:
        image_clip, out_transform = mask(dataset=src, shapes=[rgi_reproj], crop=True, nodata=99)
        out_meta = src.meta
    
    # take only the correct band
    image_clip = image_clip[band,:,:]

    # save the image to correct folder
    out_meta.update({"driver": "GTiff",
                 "height": image_clip.shape[0],
                 "width": image_clip.shape[1],
                 'count':1,
                 "transform": out_transform})

    if not os.path.isdir(os.path.join(temp_folder, rgiid)):
        os.mkdir(os.path.join(temp_folder, rgiid))
        
    out_path = os.path.join(temp_folder, rgiid, image_id[-20:]+".tif")
    with rio.open(out_path, "w", **out_meta) as dest:
        dest.write(image_clip, 1)
    
    return image_clip



def clip_to_outline(rgiid, date, geom):
    # open csv of all images taken of the glacier
    csv = pd.read_csv(os.path.join(csv_folder, str(rgiid)+".csv"))
    
    # select the one corresponding with the passed date
    correct_row = csv[csv['date']==date]
    
    # extract landsat image ID
    if len(correct_row) > 0:
        img_id = correct_row['id'].iloc[0]
        
        # open that landsat image, clip to the glacier outline
        clipped = clip_to_rgi(img_id, geom, rgiid)
    
    else:
        clipped = 0
    
    return clipped

#temp1 = rgi_gdf.iloc[15]
temp1 = rgi_gdf[rgi_gdf['RGIId']=='RGI60-01.21013'].iloc[0]

imgs = []
for y in range(2013,2022):
    print(y)
    t = clip_to_outline(temp1['RGIId'], temp1['min_date_'+str(y)], temp1['geometry'])
    
    if isinstance(t,int):
        print('skipped')
        continue
    else:
        imgs.append(t)

# the images are not all in the same crs. they need to all be reprojected to the same UTM
# corrdinates before stacking. need to think a little more carefully about how to do this.
img_all = np.dstack(imgs)

img_mode = scipy.stats.mode(img_all, axis=2)[0]

plt.figure()
plt.imshow(img_all[:,:,4], vmin=-1, vmax=4, cmap='Blues')

plt.figure()
plt.imshow(img_mode, vmin=-1, vmax=4, cmap='Blues')

#%%
def reproj(infile, match, outfile):
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
                src.width,   # input width
                src.height,  # input height 
                *src.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
            )

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": 99})
        #print("Coregistered to shape:", dst_height,dst_width,'\n Affine',dst_transform)
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
                    resampling=Resampling.nearest)
        #return src.read(1)
        
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
                           "nodata": 99})
        #print("Coregistered to shape:", dst_height,dst_width,'\n Affine',dst_transform)
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
                    resampling=Resampling.nearest)
        #return src.read(1)

folder = os.path.join(temp_folder, 'RGI60-01.20422')
i = os.path.join(folder, 'LC08_060019_20180905.tif')
m = os.path.join(folder, 'LC08_059019_20131103.tif')
o1 = os.path.join(folder, 'LC08_060019_20180905_re1.tif')
o2 = os.path.join(folder, 'LC08_060019_20180905_re2.tif')
reproj(i,m,o1)
reproj_match(o1, m, o2)

#%%
check = rio.open(o2).read(1)
plt.figure()
plt.imshow(check, vmin=-1, vmax=4, cmap='Blues')
    
#%%


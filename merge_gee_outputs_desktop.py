# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 08:59:25 2022

@author: lzell

https://www.youtube.com/watch?v=sBBMKbAj8XE
https://gdal.org/drivers/raster/gtiff.html
https://gdal.org/programs/gdal_translate.html
https://gdal.org/programs/gdal_merge.html
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
import shutil

save = 1
show_fig = 0 # only ever do this if you are doing a SINGLE image

### define folder with all images that you want merged

# non-PC classified images
big_folder = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop','AGVA','classified images','L8 Classified Images')
length = 32

# PC classified images
# big_folder = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop','AGVA','classified images','L8 Classified Images PC')
# length = 35

# # cloud mask images
# big_folder = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop','AGVA','classified images','L8 Cloud')
# length = 38




# set working directory to a different sub-folder to hold the final images
merged_folder = os.path.join(big_folder)
unmerged_folder = os.path.join(big_folder,'unmerged')
os.chdir(merged_folder)

# get list of the 100+ base image names
names = []
for entry in os.scandir(unmerged_folder):
    name = entry.name[:length]
    if name == 'merged':
        continue
    if name in names: 
        continue
    else:
        names.append(name)



#%%

# iterate through each of the names
names = names[:] # if you want to test it on a single image

c=0
for n in names:
    LS_name = n

    # load the paths to each image segment into a list
    img_paths = []
    for entry in os.scandir(unmerged_folder):
        if entry.name.startswith(LS_name):
            path = os.path.join(unmerged_folder,entry.name)
            img_paths.append(path)

    # define the path to save merged file to
    out_path = os.path.join(merged_folder,LS_name + '.tif')
    
    # if it's just a single image, then it doesn't need to be merged
    # just copy the file to the merged folder
    if len(img_paths)==1:
        shutil.copy(img_paths[0], out_path)
    
    # if it's >1 files to merge, use gdal
    elif len(img_paths)>1:
        # use gdal to merge all those image segments into a single file, saving it with the correct name
        if save:   
            vrt = gdal.BuildVRT("merged.vrt", img_paths)
            #gdal.Translate(out_name, vrt, outputType=gdal.GDT_Byte, creationOptions = ['PREDICTOR=2','COMPRESS=LZW'])
            gdal.Translate(out_path, vrt, outputType=gdal.GDT_Float32, creationOptions = ['PREDICTOR=2','COMPRESS=LZW'])
            vrt = None
    
    # if there are no files to emrge, you have a problem
    else:
        print("Error with",n,"No files found to merge.")

    c+=1
    print(c,"of",len(names),"saved")
#%%
    # if you want to show the figure, do it here
    if show_fig & len(names)==1:
        
        srcs = []
        for entry in os.scandir(unmerged_folder):
            if entry.name.startswith(LS_name):
                srcs.append(rio.open(os.path.join(unmerged_folder,entry.name)))
    
        fig,axs = plt.subplots(2,2, figsize=(10,10), sharex=True, sharey=True)
        
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        
        c=0
        for s in srcs:
            a = axs[c//2,c%2]
            rioshow((srcs[c],1), ax=a,vmin=0,vmax=99)
            
            xmins.append(min(a.get_xlim()))
            xmaxs.append(max(a.get_xlim()))
            ymins.append(min(a.get_ylim()))
            ymaxs.append(max(a.get_ylim()))
            
            c+=1
        
        a.set_xlim(min(xmins),max(xmaxs))
        a.set_ylim(min(ymins),max(ymaxs))
        
        plt.tight_layout()
        
        
        fig,axs = plt.subplots(figsize=(10,10))
        for s in srcs:
            rioshow((s,1), ax=axs,vmin=0,vmax=99)
        axs.set_xlim(min(xmins),max(xmaxs))
        axs.set_ylim(min(ymins),max(ymaxs))



#%%

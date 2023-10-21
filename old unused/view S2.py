# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 09:34:38 2023

@author: lzell
"""

import os
import rasterio as rio
import numpy as np
from rasterio.merge import merge as riomerge
from rasterio.plot import show as rioshow
import matplotlib.pyplot as plt

cloud = 0

# set path to the folder holding those images
merged_folder_path = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA","classified images","S2_Classified_Merged")
if cloud: merged_folder_path = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA","classified images","S2_Cloud_Merged")

# list out the image names in the folder
all_image_names = os.listdir(merged_folder_path)

#%%
# go through each image and plot it up
fig,axs = plt.subplots()

xmin = 1e9
xmax = 0
ymin = 1e9
ymax = 0

c=0
for i in all_image_names:
    if i[-3:] != "tif": continue
    # if c>1: continue

    image_path = os.path.join(merged_folder_path, i)
    print(image_path)

    # open image with rasterio
    with rio.open(image_path) as src:
        
        # get bounding box
        bounds = src.bounds
        # print(src.bounds)
        xmin = min(xmin, bounds[0])
        xmax = max(xmax, bounds[2])
        ymin = min(ymin, bounds[1])
        ymax = max(ymax, bounds[3])

        # read first band, set 99 to nan
        band1 = src.read(1)
        
        if cloud:
            band1[band1==99] = 13
            band1[band1==98] = 12
    
            # plot, with extent controlled
            axs.imshow(band1, extent=[bounds[0],bounds[2],bounds[1],bounds[3]], vmin=0, vmax=13)
        
        else:
            band1[band1==99] = 4
            band1[band1==98] = 3
            
            masked_array = np.ma.array(band1, mask=(band1==3)) 
    
            # plot, with extent controlled
            # axs.imshow(band1, extent=[bounds[0],bounds[2],bounds[1],bounds[3]], vmin=0, vmax=4)
            axs.imshow(masked_array, extent=[bounds[0],bounds[2],bounds[1],bounds[3]], vmin=0, vmax=4, interpolation='none')
        
        # # set 99 to nan
        # rioshow(src,1, ax=axs)

    c+=1

axs.set_xlim(xmin, xmax)
axs.set_ylim(ymin, ymax)
plt.axis('scaled')



# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:41:28 2022

@author: lzell
"""
import os

folder = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop','AGVA','classified images','L8 Classified Images PC')
os.chdir(folder)

for entry in os.scandir(folder):
    name = entry.name
    
    if name[-4:] == ".tif":
        new_name = name[:32]+".tif"
        # print(new_name)
        os.rename(name, new_name)
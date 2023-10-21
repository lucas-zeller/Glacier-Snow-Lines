# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:00:57 2022

@author: lzell
"""

import os
import pandas as pd
import numpy as np

# iterate through each Path/Row folder
# iterate through each rgi csv
# if there is no existing 'master' csv for this rgi, make a new one
# else, copy this data into the already existing csv

#%%
# base folder
agva_folder = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop','AGVA')

### define path to where the path/row folders are, and where the master csvs are
# PC
# images_csv_folder = os.path.join(agva_folder, 'classified images', 'L8 Classified Images PC', 'Glacier csvs')
# master_csv_folder = os.path.join(agva_folder, 'AA observations csv', 'PC')

# no PC
images_csv_folder = os.path.join(agva_folder, 'classified images', 'L8 Classified Images no PC', 'Glacier csvs')
master_csv_folder = os.path.join(agva_folder, 'AA observations csv', 'no PC')

#%%

# iterate through each R/P csv folder
rpc = 0
for rp_folder in os.scandir(images_csv_folder):
    
    # for testing on just a couple
    if rpc>9999999: continue
    rpc+=1
    
    # grab the folder name
    rp_folder_name = rp_folder.name
    print(rpc,'of',"95",rp_folder_name)
    
    # iterate through each csv in the folder
    for rgi_csv in os.scandir(rp_folder):
        
        # grab file name, rgi name, filepath to csv
        rgi_csv_name = rgi_csv.name
        rgi_name = rgi_csv_name[8:-4]
        csv_fp = os.path.join(images_csv_folder,rp_folder_name,rgi_csv_name)
        
        # open with pandas
        single_df = pd.read_csv(csv_fp)
        
        # drop unnamed columns
        single_df.drop(single_df.columns[single_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        
        # add row/path as columns
        single_df['row'] = rp_folder_name[5:]
        single_df['path'] = rp_folder_name[1:3]
        
        # see if there is an already existing 'master' csv
        master_csv_fp = os.path.join(master_csv_folder, rgi_name+".csv")
        
        # if it doesn't exist, simply copy this file to the 'master' folder and rename
        if not os.path.exists(master_csv_fp):
            
            # save to csv
            single_df.to_csv(master_csv_fp, index=False)
        
        # otherwise, open the master file, append the new data to it, and resave
        else:
            #print('found a double',rgi_name)
            
            # open master df
            master_df = pd.read_csv(master_csv_fp)
            
            # drop unnamed columns
            master_df.drop(master_df.columns[master_df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        
            # concatenate all the rows together
            new_df = pd.concat([master_df,single_df], ignore_index=True)
            
            # resave
            new_df.to_csv(master_csv_fp, index=False)

print("Reordering by date")            
# at the end, go through and sort all of our master DFs by date
for master_file in os.scandir(master_csv_folder):     
    
    master_df = pd.read_csv(master_file)
    master_df.sort_values('date',inplace=True)
    master_df.to_csv(master_file, index=False)




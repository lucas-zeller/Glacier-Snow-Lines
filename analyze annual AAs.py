# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:59:44 2022

@author: lzell
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# how to define accumulation area
#aa_definition = 'snowfirn'
aa_definition = 'snow'

# filter by months
start_month = 7
end_month = 11

# folder paths
agva_folder = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop','AGVA')
os.chdir(agva_folder)

# with/without pc
master_csv_folder = os.path.join(agva_folder, 'classified images', 'L8 Classified Images PC', 'csvs')
master_csv_folder = os.path.join(agva_folder, 'AA observations csv', 'PC')

# set minimum cloud-free area needed (0-100)
min_cloudfree = 90

# years that will be run through
years = [2013,2014,2015,2016,2017,2018,2019,2020,2021]

# master master dataframe to hold everything
master_master_df = pd.DataFrame()
rows = []

# iterate through each csv
rpc = 0
for fp in os.scandir(master_csv_folder):
    
    # skip first one
    if fp.name[:3] == 'ann': continue

    # option to only test on a few
    if rpc>10000000:
        continue
    rpc+=1

    print(rpc,'of','~9800')
    
    # get rgi id
    rgi_id = fp.name[:-4]
    
    # open as df
    rgi_df = pd.read_csv(fp)
    
    # filter to the months we know the minimum AA observations should be in
    rgi_df['month'] = rgi_df.apply(lambda row: int(str(row.date)[5:7]), axis=1)
    rgi_df = rgi_df[rgi_df['month']>=start_month]
    rgi_df = rgi_df[rgi_df['month']<=end_month]    
    
    # calculate total id'ed pixels
    rgi_df['total_pixels'] =  rgi_df['snow pixels']+rgi_df['firn pixels']+rgi_df['ice pixels']

    # filter out some weird observations where many pixels aren't id'ed
    rgi_df = rgi_df[ rgi_df['total_pixels']*30*30/(1000*1000) >= rgi_df['glacier area']*0.8]
    
    # add cloudy pixel percentage
    rgi_df['cloudfree_pixel_percent'] = (100*rgi_df['cloudfree pixels']/(rgi_df['snow pixels']+rgi_df['firn pixels']+rgi_df['ice pixels'])).astype(int)
    
    # filter by cloudfree area
    rgi_df = rgi_df[rgi_df['cloudfree_pixel_percent']>=min_cloudfree]
    
    # add AA (km2)
    if aa_definition == 'snow':
        rgi_df['AA'] = rgi_df['snow pixels']*30*30/(1000*1000)
    elif aa_definition == 'snowfirn':
        rgi_df['AA'] = (rgi_df['snow pixels']+rgi_df['firn pixels'])*30*30/(1000*1000)
    
    # add AAR (0-100)
    rgi_df['AAR'] = (100*rgi_df['AA']/(rgi_df['total_pixels']*30*30/(1000*1000))).astype(int)
    
    # dictionary to hold all the data you will want saved
    row_to_add = {'RGIId':rgi_id}
    
    # for each year, calculate minimum AA
    for y in years:
        
        # select only rows in this year
        y_df = rgi_df[rgi_df['year']==y]
        
        # find number of observations that year
        n_obs = len(y_df)
        
        if n_obs>0:
            # find minimum AA, index of the minimum AA
            min_AA = np.nanmin(y_df['AA'])
            AA_index = y_df.index[y_df['AA'] == min_AA].tolist()
            
            # get row at the latest index
            min_AA_row = y_df.loc[AA_index[-1],:]
            
            # get AAR, date, etc...
            min_AAR = min_AA_row['AAR']
            min_date = min_AA_row['date']
            
            # calculate number of matching observations
            n_match = len(AA_index)
            
        else:
            min_AA = -1
            min_AAR = -1
            min_date = -1
            n_match = -1
            
        # add each to the dictionary to save
        row_to_add['n_obs_'+str(y)] = n_obs
        row_to_add['min_AA_'+str(y)] = min_AA
        row_to_add['min_AAR_'+str(y)] = min_AAR
        row_to_add['min_date_'+str(y)] = min_date
        row_to_add['n_matching_'+str(y)] = n_match
        

    rows.append(row_to_add)

master_master_df = pd.DataFrame(rows)

#%%
save = 1
save_path = os.path.join(agva_folder, 'AA observations csv', 'PC', 'annual_minimums_temp.csv')
if save:
    master_master_df.to_csv(save_path, index=False)
    
    
    
    
    
    
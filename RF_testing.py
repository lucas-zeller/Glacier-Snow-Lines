# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:22:44 2022

@author: lzell
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from datetime import datetime
import rasterio as rio
from rasterio.mask import mask
import pandas as pd
import geopandas as gpd
import copy
import time
import multiprocessing
import json
import geemap
from geemap import ml
import csv
import ee

from scipy.ndimage import uniform_filter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score


#%%
save = 0 

# define training folder path
training_folder_path = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA","training_data")
#training_folder_path = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA","training_data","old")

df_1 = pd.read_csv(os.path.join(training_folder_path,'LC08_055020_20160827_training_dataset_0snow_1firn_2ice_3rock.csv')).drop('.geo', axis=1)
df_2 = pd.read_csv(os.path.join(training_folder_path,'LC08_067016_20190808_training_dataset_0snow_1firn_2ice_3rock.csv')).drop('.geo', axis=1)
df_3 = pd.read_csv(os.path.join(training_folder_path,'LC08_067017_20160831_training_dataset_0snow_1firn_2ice_3rock.csv')).drop('.geo', axis=1)
df_4 = pd.read_csv(os.path.join(training_folder_path,'LC08_068018_20180929_training_dataset_0snow_1firn_2ice_3rock.csv')).drop('.geo', axis=1)
df_5 = pd.read_csv(os.path.join(training_folder_path,'LC08_070017_20200815_training_dataset_0snow_1firn_2ice_3rock.csv')).drop('.geo', axis=1)

dfs = [df_1, df_2, df_3, df_4, df_5]
print(df_1.columns)

#%%
#rename bands
renames = {'SR_B1':'coastal','SR_B2':'blue','SR_B3':'green','SR_B4':'red','SR_B5':'nir','SR_B6':'swir1','SR_B7':'swir2'}
renames_SO = {'SO_SR_B1':'SO_coastal','SO_SR_B2':'SO_blue','SO_SR_B3':'SO_green','SO_SR_B4':'SO_red','SO_SR_B5':'SO_nir','SO_SR_B6':'SO_swir1','SO_SR_B7':'SO_swir2'}
for d in dfs:
    d.rename(columns=renames,inplace=True)
    d.rename(columns=renames_SO,inplace=True)
    
print(df_1.columns)

#%%

# add columns

def add_ndwi(df):
    df['ndwi'] = (df['green']-df['nir'])/(df['green']+df['nir'])
    df['SO_ndwi'] = (df['SO_green']-df['SO_nir'])/(df['SO_green']+df['SO_nir'])

def add_ndsi(df):
    df['ndsi'] = (df['green']-df['swir1'])/(df['green']+df['swir1'])
    df['SO_ndsi'] = (df['SO_green']-df['SO_swir1'])/(df['SO_green']+df['SO_swir1'])

# def ndwi_times_ndsi(df):
#     ndwi = df['ndwi']+1
#     ndsi = df['ndsi']+1
#     mult = ndwi*ndsi
#     df['ndwi_ndsi'] = mult
    
def convert_temp(df):
    df['temp_c'] = df['ST_B10']*0.00341802+149-273.15
    df['SO_temp_c'] = df['SO_ST_B10']*0.00341802+149-273.15

def diff_norm(df):
    for b in ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndwi', 'ndsi', 'temp_c']:
        snow_off = df[b]
        snow_on = df['SO_'+b]
        diff = snow_on - snow_off
        norm = diff/snow_on# * -1 + 1
        df['dn_'+b] = norm

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float32)

def fix_inf(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

for d in dfs:
    add_ndwi(d)
    add_ndsi(d)
    convert_temp(d)
    diff_norm(d)
    fix_inf(d)
    #d = clean_dataset(d)
    #ndwi_times_ndsi(d)
    
df_master = pd.concat(dfs)
print('Observations')
print('Snow:', len(df_master[df_master['type']==0]))
print('Firn:', len(df_master[df_master['type']==1]))
print('Ice: ', len(df_master[df_master['type']==2]))



#%%
# subsample per image to equal number of snow/ice/firn obs
df_master_balanced = []
c=0
for d in dfs:
    firn = d[d['type']==1]
    count = firn.shape[0]
    snow = d[d['type']==0].sample(n=count, random_state=970)
    ice = d[d['type']==2].sample(n=count, random_state=336)
    bal_df = pd.concat([firn, snow, ice])
    bal_df['image'] = c
    df_master_balanced.append(bal_df)
    c+=1
df_master_balanced = pd.concat(df_master_balanced).drop('system:index',1)
df_master_balanced = df_master_balanced.apply(pd.to_numeric)


#%%
# train rf
#predictor_list = ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndwi', 'ndsi', 'temp_c']
predictor_list = ['coastal', 'blue', 'green', 'red', 'nir', 'ndwi', 'ndsi']
#predictor_list = ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2']
#predictor_list = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndwi', 'ndsi', 'temp_c']
# predictor_list = ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndwi', 'ndsi', 'temp_c', 
#                   'dn_coastal', 'dn_blue', 'dn_green', 'dn_red', 'dn_nir', 'dn_swir1', 'dn_swir2', 'dn_ndwi', 'dn_ndsi', 'dn_temp_c']
predictor_list = ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndwi', 'ndsi',
                  'dn_coastal', 'dn_blue', 'dn_green', 'dn_red', 'dn_nir', 'dn_swir1', 'dn_swir2', 'dn_ndwi', 'dn_ndsi']

test_image = 0
train = df_master_balanced[df_master_balanced['image']==test_image]
test = df_master_balanced[df_master_balanced['image']!=test_image]

# lat = 61.7
# train = df_master_balanced[df_master_balanced['latitude']<lat]
# test = df_master_balanced[df_master_balanced['latitude']>=lat]

df_master_balanced = df_master_balanced[df_master_balanced['type']<3]

#train_preds, test_preds, train_truth, test_truth = train_test_split(df_master_balanced[predictor_list], df_master_balanced['type'], test_size=0.33, random_state=336)

train_preds = train[predictor_list].astype(float)
train_truth = train['type'].astype(float)
test_preds = test[predictor_list].astype(float)
test_truth = test['type'].astype(float)

# make our best model with the given parameters
RFC = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10, 
                             min_samples_leaf=10, max_features=7, n_jobs=-1, oob_score=True, random_state = 336)
RF_fit = RFC.fit(train_preds, train_truth)

# # save the model to disk
# save_model=0
# if save_model:
#     filename = os.path.join(folder_path,'RF_model_ATS780.sav')
#     pickle.dump(RF_fit, open(filename, 'wb'))



# classify the test data with this fitted RF
pred_RF_test = RF_fit.predict(test_preds)
pred_RF_train = RF_fit.predict(train_preds)


cm_test = confusion_matrix(test_truth, pred_RF_test)
cm_train = confusion_matrix(train_truth, pred_RF_train)

print('cm train')
print(cm_train)
print('cm test')
print(cm_test)

print("OOB score:",RF_fit.oob_score_)
print("Train accuracy:",accuracy_score(train_truth,pred_RF_train))
print("Test accuracy:",accuracy_score(test_truth,pred_RF_test))

colors = ['tab:cyan','tab:blue','tab:green','tab:red','hotpink','tab:orange','gold','cornflowerblue','powderblue','firebrick','tab:cyan','tab:blue','tab:green','tab:red','hotpink','tab:orange','gold','cornflowerblue','powderblue','firebrick']
colors = colors[:len(predictor_list)+1]
names = predictor_list

importances = pd.Series(RF_fit.feature_importances_, index=names)


fig, ax = plt.subplots(figsize=(8,6), dpi=100)
plt.grid(axis='y', zorder=-1)
ax.bar(range(len(predictor_list)),height=importances, zorder=10, color=colors)
ax.set_xticks(range(len(predictor_list)))
ax.set_xticklabels(importances.index, rotation=60)
plt.ylabel('Importance')
plt.tight_layout()


#%%
# save to string
#https://geemap.org/ml/
#https://github.com/giswqs/geemap/blob/master/examples/notebooks/local_rf_training.ipynb
def rf_strings(estimator, feature_names):
    estimators = np.squeeze(estimator.estimators_)
    class_labels = estimator.classes_
    trees = []
    for e in estimators:
        s = ml.tree_to_string(e, feature_names=feature_names, labels=class_labels)
        trees.append(s)
    return trees

estimators = np.squeeze(RF_fit.estimators_)
s = ml.tree_to_string(estimators[0], feature_names=predictor_list, labels=RF_fit.classes_)

save=1
if save:
    out_file = os.path.join(training_folder_path,'RF_csv2.csv')
    out_file2 = os.path.join(training_folder_path,'RF_txt.txt')
    
    #trees = ml.rf_to_strings(RF_fit, predictor_list,processes=1)
    trees = rf_strings(RF_fit, predictor_list)
    
    
    # trees_df = pd.DataFrame (trees, columns = ['trees'])
    # trees_df['extra'] = 1
    
    # trees_df.to_csv(out_file, index=False)
    # ml.trees_to_csv(trees, out_file)
#%%

if save:    
    # get user id
    ee.Initialize()
    user_id = geemap.ee_user_id()
    user_id
    
    # name asset
    asset_id = user_id + "/random_forest_strings_test"
    asset_id = "projects/lzeller/assets/random_forest_strings_with_SO"
    asset_id

    # kick off an export process so it will be saved to the ee asset
    ml.export_trees_to_fc(trees, asset_id)
    
    #%% 
    # string_out = "["
    
    # for s in trees:
    #     string_out = string_out + s + ","
    # string_out = string_out + "["
    
    # results = trees
    # with open(out_file,'w', newline='') as result_file:
    #     wr = csv.writer(result_file)
    #     for val in results:
    #         wr.writerow([val[:-1]])


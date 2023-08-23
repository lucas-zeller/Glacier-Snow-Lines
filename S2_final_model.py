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
import pandas as pd
import copy
import time
import geemap
from geemap import ml
import ee

from scipy.ndimage import uniform_filter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score, f1_score

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

#%%
# # Trigger the authentication flow.
# ee.Authenticate()

# # Initialize the library.
# ee.Initialize()

#%%
upload_model = 1
model_name = "random_forest_S2_TOA"

# define training folder path
training_folder_path = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA","training_data",'S2')

# open all the training datasets, concat into single pandas df
df_1 = pd.read_csv(os.path.join(training_folder_path,'GS2A_20190901T200911_021904_N02_training_dataset_1snow_2firn_3ice_4rock_5shadow_6water.csv'))
df_2 = pd.read_csv(os.path.join(training_folder_path,'GS2A_20200909T194951_027252_N02_training_dataset_1snow_2firn_3ice_4rock_5shadow_6water.csv'))
df_3 = pd.read_csv(os.path.join(training_folder_path,'GS2A_20210829T211521_032315_N03_training_dataset_1snow_2firn_3ice_4rock_5shadow_6water.csv'))
df_4 = pd.read_csv(os.path.join(training_folder_path,'GS2A_20220810T213541_037263_N04_training_dataset_1snow_2firn_3ice_4rock_5shadow_6water.csv'))
df_5 = pd.read_csv(os.path.join(training_folder_path,'GS2A_20220918T202121_037820_N04_training_dataset_1snow_2firn_3ice_4rock_5shadow_6water.csv'))
df_6 = pd.read_csv(os.path.join(training_folder_path,'GS2B_20190822T210029_012853_N02_training_dataset_1snow_2firn_3ice_4rock_5shadow_6water.csv'))
df_7 = pd.read_csv(os.path.join(training_folder_path,'GS2B_20200822T212529_018087_N02_training_dataset_1snow_2firn_3ice_4rock_5shadow_6water.csv'))
df_8 = pd.read_csv(os.path.join(training_folder_path,'GS2B_20210815T204019_023206_N03_training_dataset_1snow_2firn_3ice_4rock_5shadow_6water.csv'))

dfs = [df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8]

for i in range(len(dfs)):
    dfs[i]['image'] = i
df_all = pd.concat(dfs)

# rename bands
bands_orig = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 
              'SO_B1', 'SO_B2', 'SO_B3', 'SO_B4', 'SO_B5', 'SO_B6', 'SO_B7', 'SO_B8', 'SO_B8A', 'SO_B9', 'SO_B11', 'SO_B12']
bands_new = ['coastal', 'blue', 'green', 'red', 're1', 're2', 're3', 'nir', 're4', 'vapor', 'swir1', 'swir2',
             'SO_coastal', 'SO_blue', 'SO_green', 'SO_red', 'SO_re1', 'SO_re2', 'SO_re3', 'SO_nir', 'SO_re4', 'SO_vapor', 'SO_swir1', 'SO_swir2']
bands_raw = ['coastal', 'blue', 'green', 'red', 're1', 're2', 're3', 'nir', 're4', 'vapor', 'swir1', 'swir2', 'ndwi', 'ndsi']

bands_dict = {bands_orig[i]:bands_new[i] for i in range(len(bands_orig))}
df_all = df_all.rename(columns=bands_dict)

# add ndsi, ndwi, dn
df_all['ndwi'] = (df_all['green']-df_all['nir'])/(df_all['green']+df_all['nir'])
df_all['SO_ndwi'] = (df_all['SO_green']-df_all['SO_nir'])/(df_all['SO_green']+df_all['SO_nir'])
df_all['ndsi'] = (df_all['green']-df_all['swir1'])/(df_all['green']+df_all['swir1'])
df_all['SO_ndsi'] = (df_all['SO_green']-df_all['SO_swir1'])/(df_all['SO_green']+df_all['SO_swir1'])

for i in bands_raw:
    df_all[f"DN_{i}"] = df_all[f"{i}"]/df_all[f"SO_{i}"]

# remove bad data
df_all = df_all.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

print("All the data is loaded")

#%%
# define predictor variable names and target variable name
predictors = [
                'coastal', 'blue', 'green', 'red', 're1', 're2', 're3', 'nir', 're4', 'vapor', 'swir1', 'swir2', 'ndsi', 'ndwi',
                'SO_coastal', 'SO_blue', 'SO_green', 'SO_red', 'SO_re1', 'SO_re2', 'SO_re3', 'SO_nir', 'SO_re4', 'SO_vapor', 'SO_swir1', 'SO_swir2', 'SO_ndsi', 'SO_ndwi',
                'DN_coastal', 'DN_blue', 'DN_green', 'DN_red', 'DN_re1', 'DN_re2', 'DN_re3', 'DN_nir', 'DN_re4', 'DN_vapor', 'DN_swir1', 'DN_swir2', 'DN_ndsi', 'DN_ndwi',
              ]
              
target = 'type'

# select the data
X = df_all[predictors]
Y = df_all[target]
                 
# set class weights for the classifier
weights = {1:1, 2:1, 3:1, 4:0.5, 5:0.2, 6:0.2}
            
# initiate and fit the classifier. model parameters were previously chosen from cross-validation search
RFC = RandomForestClassifier(n_estimators=50, max_depth=15, min_samples_split=500, min_samples_leaf=100,
                             class_weight=weights, max_features='sqrt', n_jobs=-1, oob_score=True, random_state = 336)
RF_fit = RFC.fit(X, Y) 

print("The model has been fit")        

# get oob score, accuracy, etc...
oob = RF_fit.oob_score_
print(f"OOB score: {oob}")

# make figure showing predictor importances, if you want           
make_importances_figures = 0
if make_importances_figures:
    importances = pd.Series(RF_fit.feature_importances_, index=predictors)
    
    fig, ax = plt.subplots(figsize=(9,5), dpi=100)
    ax.bar(range(len(predictors)),height=importances, zorder=10)
    
    plt.grid(axis='y', zorder=1)
    ax.set_xticks(range(len(predictors)))
    ax.set_xticklabels(importances.index, rotation=60)
    
    plt.ylabel('Importance')
    plt.title(f'Test Image: {test_image}')
    plt.tight_layout()


#%%
##### export the predictor to to GEE
# https://geemap.org/ml/
# https://github.com/giswqs/geemap/blob/master/examples/notebooks/local_rf_training.ipynb

# function to turn trees into strings that are readable by GEE
def rf_strings(estimator, feature_names):
    estimators = np.squeeze(estimator.estimators_)
    class_labels = estimator.classes_
    trees = []
    for e in estimators:
        s = ml.tree_to_string(e, feature_names=feature_names, labels=class_labels)
        trees.append(s)
    return trees


if upload_model:
    
    # convert from tree to string
    trees = rf_strings(RF_fit, predictors)
    
    # create a null geometry point for feature collection.
    null_island = ee.Geometry.Point([0, 0])

    # create a list of features, each feature being a single tree
    # encode return values (\n) as #, use to parse later
    features = [ee.Feature(null_island, {"tree": tree.replace("\n", "#")}) for tree in trees]
    
    # cast as feature collection
    fc = ee.FeatureCollection(features)
    
    # name asset
    user_id = "lzeller"
    asset_id = f"projects/{user_id}/assets/{model_name}"
    description = model_name
    
    # create export task and start
    task = ee.batch.Export.table.toAsset(
        collection=fc, description=description, assetId=asset_id
    )
    task.start()

    print("Random Forest Export Started")


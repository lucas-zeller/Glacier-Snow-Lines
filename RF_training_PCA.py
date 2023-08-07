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
from pickle import dump
from pickle import load

from scipy.ndimage import uniform_filter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#%%
save = 0 

# define training folder path
training_folder_path = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA","training_data")
#training_folder_path = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA","training_data","old")

pkl_path = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA",'scripts','gee')

df_1 = pd.read_csv(os.path.join(training_folder_path,'LC08_055020_20160827_training_dataset_0snow_1firn_2ice_3rock.csv')).drop('.geo', axis=1)
df_2 = pd.read_csv(os.path.join(training_folder_path,'LC08_067016_20190808_training_dataset_0snow_1firn_2ice_3rock.csv')).drop('.geo', axis=1)
df_3 = pd.read_csv(os.path.join(training_folder_path,'LC08_067017_20160831_training_dataset_0snow_1firn_2ice_3rock.csv')).drop('.geo', axis=1)
df_4 = pd.read_csv(os.path.join(training_folder_path,'LC08_067018_20190824_training_dataset_0snow_1firn_2ice_3rock.csv')).drop('.geo', axis=1)
df_5 = pd.read_csv(os.path.join(training_folder_path,'LC08_068018_20180929_training_dataset_0snow_1firn_2ice_3rock.csv')).drop('.geo', axis=1)
df_6 = pd.read_csv(os.path.join(training_folder_path,'LC08_070017_20200815_training_dataset_0snow_1firn_2ice_3rock.csv')).drop('.geo', axis=1)

dfs = [df_1, df_2, df_3, df_4, df_5, df_6]
#print(df_1.columns)

c=0
for d in dfs:
    d['image'] = c
    c+=1

#%%
    
df_master = pd.concat(dfs)
df_master = df_master.loc[df_master['type']<=2]
# print('Observations')
# print('Snow:', len(df_master[df_master['type']==0]))
# print('Firn:', len(df_master[df_master['type']==1]))
# print('Ice: ', len(df_master[df_master['type']==2]))

band_list = ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndwi', 'ndsi',
                  'DN_coastal', 'DN_blue', 'DN_green', 'DN_red', 'DN_nir', 'DN_swir1', 'DN_swir2', 'DN_ndwi', 'DN_ndsi']

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
    df_master_balanced.append(bal_df)
    c+=1
df_master_balanced = pd.concat(df_master_balanced).drop('system:index',1)
df_master_balanced = df_master_balanced.apply(pd.to_numeric)

df_master_balanced = df_master

#%%
# PCA
# shape the values correctly
x = df_master_balanced[band_list].values

### scale the features to mean=0, sd=1
# create scaler object
scaler = StandardScaler()
#scaler = load(open('scaler.pkl', 'rb'))

# fit the scaler
#scaler = scaler.fit(x)

# load pre-saved scaler
scaler = load(open(os.path.join(pkl_path,'scaler.pkl'), 'rb'))

# these can be drawn in to GEE to be used for standardizing data
means = scaler.mean_
sds = scaler.scale_

# save scaler to use elsewhere
#dump(scaler, open('scaler.pkl', 'wb'))

# scale the inputs
x_scaled = scaler.transform(x)
# x_scaled_test = (x-means)/sds
# x_scaled_diff = x_scaled-x_scaled_test

#%%
# initiate PCA
n=18
pca_obj = PCA(n_components=n)

# perform PCA on your data

# load-presaved PCA object (with eigenvalues/vectors already calculated)
pca_obj = load(open(os.path.join(pkl_path,'PCA_obj.sav'), 'rb'))
#pca_obj = pca_obj.fit(x_scaled)
pcs = pca_obj.transform(x_scaled)

save_pca = 0
if save_pca:
    filename = 'PCA_obj.sav'
    dump(pca_obj, open(filename, 'wb'))

# print(pca_obj.explained_variance_ratio_)
# print(pca_obj.singular_values_)
# np.save('eigenvalues', pca_obj.singular_values_)
# np.save('eigenvectors', pca_obj.components_)

# add pcs to df
for i in range(n):
    df_master_balanced['PC'+str(i+1)] = pcs[:,i]

# shape into df
cols = []
for i in range(n):
    cols.append("PC"+str(i+1))
pc_df = pd.DataFrame(data = pcs, columns=cols)
pc_df['type'] = df_master_balanced['type'].values
pc_df['img'] = df_master_balanced['image'].values

df_snow = pc_df[pc_df.type==0]
df_firn = pc_df[pc_df.type==1]
df_ice = pc_df[pc_df.type==2]
dfs = [df_snow, df_firn, df_ice]



#%%
# train rf
test_image = 4

# normal bands only
predictor_list = ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndwi', 'ndsi', 'DN_coastal', 'DN_blue', 'DN_green', 'DN_red', 'DN_nir', 'DN_swir1', 'DN_swir2', 'DN_ndwi', 'DN_ndsi']

# normal plus 4 PCs
#predictor_list = ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndwi', 'ndsi', 'DN_coastal', 'DN_blue', 'DN_green', 'DN_red', 'DN_nir', 'DN_swir1', 'DN_swir2', 'DN_ndwi', 'DN_ndsi', 'PC1', 'PC2', 'PC3', 'PC4']

# PCs only
#predictor_list = ['PC1', 'PC2', 'PC3', 'PC4']


# train = df_master_balanced[df_master_balanced['image']!=test_image]
# test = df_master_balanced[df_master_balanced['image']==test_image]

train = df_master_balanced
test = df_master_balanced

train_preds = train[predictor_list].astype(float)
train_truth = train['type'].astype(float)
test_preds = test[predictor_list].astype(float)
test_truth = test['type'].astype(float)

# make our best model with the given parameters
RFC = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10, 
                             min_samples_leaf=10, max_features=5, n_jobs=-1, oob_score=True, random_state = 336)
RF_fit = RFC.fit(train_preds, train_truth)

# # save the model to disk
save_model=0
if save_model:
    filename = os.path.join(pkl_path,'RF_model_SO.sav')
    dump(RF_fit, open(filename, 'wb'))

def rf_strings(estimator, feature_names):
    estimators = np.squeeze(estimator.estimators_)
    class_labels = estimator.classes_
    trees = []
    for e in estimators:
        s = ml.tree_to_string(e, feature_names=feature_names, labels=class_labels)
        trees.append(s)
    return trees

save_strings=0
if save_strings:    
    # get user id
    ee.Initialize()
    user_id = geemap.ee_user_id()
    user_id
    
    # name asset
    asset_id = user_id + "/random_forest_strings_test"
    asset_id = "projects/lzeller/assets/RF_model_SO"
    asset_id
    
    # create string trees
    trees = rf_strings(RF_fit, predictor_list)
    
    # kick off an export process so it will be saved to the ee asset
    ml.export_trees_to_fc(trees, asset_id)
    
#%%

# classify the test data with this fitted RF
pred_RF_test = RF_fit.predict(test_preds)
pred_RF_train = RF_fit.predict(train_preds)


cm_test = confusion_matrix(test_truth, pred_RF_test)
cm_train = confusion_matrix(train_truth, pred_RF_train)

# print('cm train')
# print(cm_train)
# print('cm test')
# print(cm_test)
print('Image:',test_image)
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

save=0
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



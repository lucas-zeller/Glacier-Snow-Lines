import os
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.mask import mask
from rasterio.windows import from_bounds, Window
from scipy.ndimage import uniform_filter, map_coordinates, convolve, binary_fill_holes
from xml.dom import minidom

def extract_ELAs(xr_class_map, xr_dem, step=10, width=20):
    '''
    Parameters
    ----------
    xr_class_map : xarray dataarray
        2d dataarray showing the distribution of snow (1) and not-snow(0) at a single timestep.
    xr_dem : xarray dataarray
        2d dataarray, with identical size/shape to xr_class_map, showing elevation.
    step : int, optional
        DESCRIPTION. The default is 10.
    width : int, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    list
        elevations where the snow and ice fraction are equal. the ELA/SLA.

    '''
    
    # lets get the minimum and maximum elevation on the glacier
    z_min = np.nanmin(xr_dem)
    z_max = np.nanmax(xr_dem)

    # get the centers of each elevation band
    z_bands = np.arange( np.ceil(z_min/step)*step-width, np.ceil(z_max/step)*step-width, step) 

    snow_fractions = []
    for z in z_bands:
        
        # subset the snow/ice class to this elevation
        band_subset = xr_class_map.where( ( xr_dem>=(z-width) ) & ( xr_dem<(z+width) ) )
        
        # calculate mean. this will give % of this area that is snow
        band_mean = np.nanmean(band_subset)
        
        # append
        snow_fractions.append(band_mean)

    # format to numpy array
    snow_fractions = np.array(snow_fractions)

    ### now lets find the elevation(s) where the snow fraction crosses 0.5
    sf_centered = snow_fractions-0.5 #center around 0.5
    idx_crossing = np.where(np.diff(np.sign(sf_centered))==2)[0] #find indices where it goes from - to +

    # interpolate to find elevation partway between z_crossing where the crossing occurs
    def interp_cross(idx_c):
        z_c = z_bands[idx_c]
        slope = ( sf_centered[idx_c+1] - sf_centered[idx_c] ) / (step) # calculate slope between this point and the next
        crossing = z_c - sf_centered[idx_c]/slope #calculate where that line crosses 0
        return crossing
    
    z_crossing = [interp_cross(i) for i in idx_crossing]

    idx_zero = np.where(sf_centered==0)[0] #also get indices where it is exactly 0.5
    z_zero = [z_bands[i] for i in idx_zero] #these elevations are exactly 0.5
    
    # append it all together
    all_elas = z_crossing+z_zero
    if len(all_elas)==0:
        all_elas = [z_min]
    
    return all_elas
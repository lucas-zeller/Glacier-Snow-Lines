import os
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio as rio
import xarray as xr
import rioxarray as riox
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.mask import mask
from rasterio.windows import from_bounds, Window
from scipy.ndimage import uniform_filter, map_coordinates, convolve, binary_fill_holes
from xml.dom import minidom
import matplotlib.pyplot as plt



def extract_all_ELAs(xr_class_map, xr_dem, xr_mask, step=20, width=1, p_snow=0.5):
    '''
    Parameters
    ----------
    xr_class_map : xarray dataarray
        2d dataarray showing the distribution of snow (1) and not-snow(0) at a single timestep.
    xr_dem : xarray dataarray
        2d dataarray, with identical size/shape to xr_class_map, showing elevation.
    xr_mask : xarray dataarray
        2d dataarray showing binary mask of glacier surface
    step : int, optional
        DESCRIPTION. The default is 20. The step size, in m, between zones
    width : int, optional
        DESCRIPTION. The default is 1. How many zones to buffer in each direction to smooth

    Returns
    -------
    list
        elevations where the snow and ice fraction are equal. the ELA/SLA.

    '''
    # extract the day/time for each obs
    times = xr_class_map.time.values
    
    # lets get the minimum and maximum elevation on the glacier
    z_min = np.nanmin(xr_dem.where(xr_dem>0))
    z_max = np.nanmax(xr_dem.where(xr_dem>0))
    # print(z_min, z_max)

    # get the centers of each elevation band
    z_bands = np.arange( np.ceil(z_min/step)*step-(step*width), np.ceil(z_max/step)*step, step) 
    # print(z_bands)
    
    # get the min, max of each elevation band
    z_mins =  np.arange( np.ceil(z_min/step)*step-(step*width), np.ceil(z_max/step)*step, step)
    z_maxs =  z_mins+(step*width)

    ### now reclassify dem into these bands, labeled 1,2,3,... 
    dem = xr_dem.copy()
    for i in range(len(z_mins)):
        dem = xr.where( (dem>=z_mins[i]) & (dem<z_maxs[i]), i+1, dem )

    # fig,ax = plt.subplots()
    # ax.imshow(xr_dem[0])
    
    # we need to rename the dem 'time' dimension to not be 'time' for this
    dem = dem.rename({'time':'time2'})
    
    # # for each band value in dem, we want to count the number of snow pixels in each time step
    df_grouped_snow = xr.Dataset({'dem': dem, 'class': xr_class_map}).to_dataframe().groupby(['dem','time']).agg({"class": "sum"}).reset_index()
    
    # now count number of total pixels within each band
    df_grouped_total = []
    for i in range(len(z_mins)):
        count = np.nansum(xr.where(dem==i+1, 1, 0))
        df_grouped_total.append([i+1,count])
    df_grouped_total = pd.DataFrame(df_grouped_total, columns=['dem','count'])
    
    # drop zone 0 from both
    df_grouped_snow = df_grouped_snow[df_grouped_snow['dem']>0]
    df_grouped_total = df_grouped_total[df_grouped_total['dem']>0]
    
    ### now for each zone, calculate the % snow cover (with 1 zone buffer) on each date
    snow_fractions = []
    for i in range(len(z_mins)):
        # if i>10:continue
        
        # define the range of zones we are looking at
        zone_min = max(1,i)
        zone_max = min(len(z_mins),i+2)
        # print(i, zone_min, zone_max)
        
        # subset snow count to these zones, group by date, sum
        count_snow = df_grouped_snow[ (df_grouped_snow['dem']>=zone_min) & (df_grouped_snow['dem']<=zone_max)]
        count_snow = count_snow.groupby('time').agg({'class':'sum'})
        
        # then do the same for total count
        count_total = df_grouped_total[ (df_grouped_total['dem']>=zone_min) & (df_grouped_total['dem']<=zone_max)]
        count_total = np.nansum(count_total['count'])
        
        # divide to get snow cover fraction of this zone at each time step
        snow_frac = count_snow['class'].values/count_total
        # gives np array of the fractional snow cover in this zone on each date (no date index is preserved)
    
        snow_fractions.append(snow_frac)
        
    snow_fractions = np.array(snow_fractions).T
   
    # plt.figure()
    # plt.imshow(snow_fractions)
    
    ### now lets find the elevation(s) where the snow fraction crosses 0.5, for each time step
    sf_centered = snow_fractions-p_snow #center around 0.5
    idx_crossing = np.nonzero( np.diff( np.sign(sf_centered) )==2 )

    # format into pandas df
    idx_crossing = pd.DataFrame({'time':idx_crossing[0], 'z':idx_crossing[1]})
    
    # function to interpolate to find elevation partway between z_crossing where the crossing occurs 
    def interp_cross(idx_c):
        i_t = idx_c['time'] #time index
        i_z = idx_c['z'] #ele index
        
        # calculate line equation and where it crosses 0.5
        slope = ( sf_centered[i_t][i_z+1] - sf_centered[i_t][i_z] ) / (step) # calculate slope between this point and the next
        crossing = z_bands[i_z] - sf_centered[i_t][i_z]/slope #calculate where that line crosses 0
        
        return [times[i_t], crossing] # return time and elevation of crossing
    
    # apply function. this takes only a small fraction of the time that the first step does
    z_crossing = idx_crossing.apply(interp_cross, axis=1, raw=False, result_type='expand').rename(columns={0:'time', 1:'z'})

    # also get the bands where there is an exact 50/50 snow/ice split
    idx_zero = np.nonzero( sf_centered==0 )
    idx_zero = pd.DataFrame({'time':idx_zero[0], 'z':idx_zero[1]})
    z_zero = idx_zero.apply(lambda x: [ times[x['time']], z_bands[x['z']] ], axis=1, raw=False, result_type='expand' ).rename(columns={0:'time', 1:'z'})
    
    # append together, sort by time, and return
    all_elas = pd.concat([z_crossing, z_zero], ignore_index=True)#.sort_values('time')
    
    # for dates where there are no elas found, decide if ela is above glacier (9999) or below (0)
    last_elas = []
    for d in times:
        
        # count number of ela obs on this date, if zero, go to next
        num_obs = len(all_elas[all_elas['time']==d])
        if num_obs>0:
            continue
        
        # else decide if ela should be 0 or 9999, based on total snow fraction on glacier surface
        else:
            # get the observation from this date, count snow, divide by glacier area
            snow_c = xr_class_map.sel(time=d).sum()
            snow_frac = snow_c/xr_mask.sum()
            
            if snow_frac>0.5:
                last_elas.append([d,-1])
            else:
                last_elas.append([d,9999])
    
    # format to df, append, return
    last_elas = pd.DataFrame(last_elas, columns=['time','z'])
    all_elas = pd.concat([all_elas, last_elas], ignore_index=True)#.sort_values('time')
    
    return all_elas


def choose_one_ELA(xr_class_map, xr_dem, df_elas):
    
    # calculate the aar for each date
    area = xr_class_map.max(dim='time').sum()
    aars = xr_class_map.sum(dim=('y','x'))/area
    
    # for each AAR, get the "ideal" ela
    elevations = xr_dem.values.flatten()
    elevations = elevations[elevations>0]
    elas = np.percentile(elevations, (1-aars.values)*100)
    
    # make df for date, aar, ela
    df_ideal = pd.DataFrame({'time':xr_class_map.time, 'aar':aars, 'ela':elas})
    
    ### for obs in df_elas with multiple ela estimations, pick the best one
    # first initiate a df with unique dates
    unique_dates = pd.DataFrame(df_elas['time'].drop_duplicates())
    
    # now for each date, select best ELA
    list_best = []
    def best_choice(row):
        
        # see how many obs there are on this date
        all_obs = df_elas[df_elas['time'] == row['time']]
        
        if len(all_obs)==1:
            return all_obs['z'].values[0] # if only one ela, then that's the best
        else:
            options = all_obs['z'].values # get all our ela options
            ideal = df_ideal[df_ideal['time']==row['time']]['ela'].values[0] # get the 'ideal' ela
            
            # return the option that is closest to ideal
            return options[(np.abs(options - ideal)).argmin()]
        
    unique_dates['ela'] = unique_dates.apply(best_choice, axis=1, raw=False, result_type='expand')
    
    return unique_dates


# wrap the above two functions into a single call
def get_the_ELAs(xr_class_map, xr_dem, xr_mask, step=20, width=1, p_snow=0.5):
    all_ELAs = extract_all_ELAs(xr_class_map, xr_dem, xr_mask, step=step, width=width, p_snow=p_snow)
    best_ELAs = choose_one_ELA(xr_class_map, xr_dem, all_ELAs).sort_values('time')
    return best_ELAs
 

# function to get the "idealized" aar-ela relationship
def idealized_ELA_AAR(xr_dem, xr_mask, step=0.01):
    # for each 1% change in aar, see what the corresponding ideal ela would be
    
    # get a list of the elevation of all pixels
    elevations = xr_dem.values.flatten()
    elevations = elevations[elevations>0]
    
    # make list of aars we want
    aars = np.arange(0,1+step,step)
    elas = np.percentile(elevations, (1-aars)*100)
    
    # make into df
    df = pd.DataFrame({'aar':aars, 'ela':elas})
    
    return df



# function to tell you what S2 subregion a given rgi outline is in
# return int
def get_S2_subregion(rgi_geom):
    
    # open the s2 subregions shapefile
    path_subregions = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA","RGI","S2_subregions","subregions.shp")
    gdf_subregions = gpd.read_file(path_subregions).to_crs("EPSG:3338")
    
    # find the subregion that intersects the rgi_geom
    correct_subregion = gdf_subregions[gdf_subregions['geometry'].contains(rgi_geom)]['id'].values[0]
    
    return correct_subregion



def get_base_DEM(rgi_geom, subregion=-1):
    
    if subregion==-1:
        subregion = get_S2_subregion(rgi_geom)
    
    # set folder paths, etc...
    folder_AGVA = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA")
    path_dem_base = os.path.join(folder_AGVA, "DEMs", '10m_COP_GLO30', f"region_{subregion:02d}_10m.tif")
    
    # open base dem, each of the dhdt products
    def open_xr(path):
        xr_da = riox.open_rasterio(path).rio.write_nodata(0)
        xr_da = xr_da.rio.clip([rgi_geom], from_disk=True, drop=True)
        return xr_da 
    
    dem_base = open_xr(path_dem_base)/10 # divide by ten for scaling factor
    dem_base = dem_base.rename({"band":"time"})
    dem_base['time'] = pd.to_datetime(["2013-01-01"])
    
    return dem_base

# function to get the time-varying dem of a glacier, 2000-2023
# return xarray dataarray, clipped to glacier extent, with single DEM for each year
def get_time_varying_DEM(rgi_geom, subregion=-1):
    
    if subregion==-1:
        subregion = get_S2_subregion(rgi_geom)
    
    # set folder paths, etc...
    folder_AGVA = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA")
    path_dem_base = os.path.join(folder_AGVA, "DEMs", '10m_COP_GLO30', f"region_{subregion:02d}_10m.tif")
    path_dhdt_00_05 = os.path.join(folder_AGVA, 'DEMs', "10m_thinning", "01_02_rgi60_2000-01-01_2005-01-01", "dhdt", f"Region_{subregion:02d}.tif")
    path_dhdt_05_10 = os.path.join(folder_AGVA, 'DEMs', "10m_thinning", "01_02_rgi60_2005-01-01_2010-01-01", "dhdt", f"Region_{subregion:02d}.tif")
    path_dhdt_10_15 = os.path.join(folder_AGVA, 'DEMs', "10m_thinning", "01_02_rgi60_2010-01-01_2015-01-01", "dhdt", f"Region_{subregion:02d}.tif")
    path_dhdt_15_20 = os.path.join(folder_AGVA, 'DEMs', "10m_thinning", "01_02_rgi60_2015-01-01_2020-01-01", "dhdt", f"Region_{subregion:02d}.tif")
    
    # open base dem, each of the dhdt products
    def open_xr(path):
        xr_da = riox.open_rasterio(path).rio.write_nodata(0)
        xr_da = xr_da.rio.clip([rgi_geom], from_disk=True, drop=True)
        return xr_da 
    
    dem_base = open_xr(path_dem_base)/10 # divide by ten for scaling factor
    dhdt_00_05 = open_xr(path_dhdt_00_05)
    dhdt_05_10 = open_xr(path_dhdt_05_10)
    dhdt_10_15 = open_xr(path_dhdt_10_15)
    dhdt_15_20 = open_xr(path_dhdt_15_20)
    
    # calculate time-varying dem for each year
    time_varying_dem = []
    base = 2013 #the raw dem represents this year
    for y in range(2000,2024):
        
        # calculate numbers years off from 2013
        dy = y-base
        
        # from this, calculate how much to multiply each of the dem products
        # I can't explain in words how this work, but trust me that I thought through it and it is good
        f10 = min( 2, max(dy,-3)) 
        f15 = max( dy-f10, 0)
        f05 = max( min(dy-f10,0), -5)
        f00 = max( min(dy-f10-f05,0), -5)
        
        #print(2000+i, dy, f"{f00}:{f05}:{f10}:{f15}")
        dem_new = ((dem_base) + (dhdt_00_05*f00*10) + 
                                (dhdt_05_10*f05*10) + 
                                (dhdt_10_15*f10*10) + 
                                (dhdt_15_20*f15*10)  ).astype(int)
        
        # set the time variable
        dem_new = dem_new.rename({"band":"time"})
        dem_new['time'] = pd.to_datetime([f"{y}-01-01"])
        
        # append to list
        time_varying_dem.append(dem_new)
    
    # append all the dems together, sort by time
    time_varying_dem = xr.concat(time_varying_dem, dim="time").sortby('time')
    
    # make off-glacier 0
    time_varying_dem = xr.where(dem_base>0, time_varying_dem, 0)
    
    return time_varying_dem



   







# def extract_ELAs(xr_class_map, xr_dem, step=20, width=20):
#     '''
#     Parameters
#     ----------
#     xr_class_map : xarray dataarray
#         2d dataarray showing the distribution of snow (1) and not-snow(0) at a single timestep.
#     xr_dem : xarray dataarray
#         2d dataarray, with identical size/shape to xr_class_map, showing elevation.
#     step : int, optional
#         DESCRIPTION. The default is 10.
#     width : int, optional
#         DESCRIPTION. The default is 20.

#     Returns
#     -------
#     list
#         elevations where the snow and ice fraction are equal. the ELA/SLA.

#     '''
    
#     # lets get the minimum and maximum elevation on the glacier
#     z_min = np.nanmin(xr_dem.where(xr_dem>0))
#     z_max = np.nanmax(xr_dem.where(xr_dem>0))

#     # get the centers of each elevation band
#     z_bands = np.arange( np.ceil(z_min/step)*step-width, np.ceil(z_max/step)*step-width, step) 

#     snow_fractions = []
#     for z in z_bands:
        
#         # subset the snow/ice class to this elevation
#         band_subset = xr_class_map.where( ( xr_dem>=(z-width) ) & ( xr_dem<(z+width) ) )
        
#         # calculate mean. this will give % of this area that is snow
#         band_mean = np.nanmean(band_subset)
        
#         # append
#         snow_fractions.append(band_mean)

#     # format to numpy array
#     snow_fractions = np.array(snow_fractions)

#     ### now lets find the elevation(s) where the snow fraction crosses 0.5
#     sf_centered = snow_fractions-0.5 #center around 0.5
#     idx_crossing = np.where(np.diff(np.sign(sf_centered))==2)[0] #find indices where it goes from - to +

#     # interpolate to find elevation partway between z_crossing where the crossing occurs
#     def interp_cross(idx_c):
#         z_c = z_bands[idx_c]
#         slope = ( sf_centered[idx_c+1] - sf_centered[idx_c] ) / (step) # calculate slope between this point and the next
#         crossing = z_c - sf_centered[idx_c]/slope #calculate where that line crosses 0
#         return crossing
    
#     z_crossing = [interp_cross(i) for i in idx_crossing]

#     idx_zero = np.where(sf_centered==0)[0] #also get indices where it is exactly 0.5
#     z_zero = [z_bands[i] for i in idx_zero] #these elevations are exactly 0.5
    
#     # append it all together
#     all_elas = z_crossing+z_zero
#     if len(all_elas)==0:
#         all_elas = [z_min]
    
#     return all_elas


# def extract_all_ELAs_old(xr_class_map, xr_dem, step=20, width=20):
#     '''
#     Parameters
#     ----------
#     xr_class_map : xarray dataarray
#         2d dataarray showing the distribution of snow (1) and not-snow(0) at a single timestep.
#     xr_dem : xarray dataarray
#         2d dataarray, with identical size/shape to xr_class_map, showing elevation.
#     step : int, optional
#         DESCRIPTION. The default is 10.
#     width : int, optional
#         DESCRIPTION. The default is 20.

#     Returns
#     -------
#     list
#         elevations where the snow and ice fraction are equal. the ELA/SLA.

#     '''
#     # extract the day/time for each obs
#     times = xr_class_map.time.values
    
#     # lets get the minimum and maximum elevation on the glacier
#     z_min = np.nanmin(xr_dem.where(xr_dem>0))
#     z_max = np.nanmax(xr_dem.where(xr_dem>0))

#     # get the centers of each elevation band
#     z_bands = np.arange( np.ceil(z_min/step)*step-width, np.ceil(z_max/step)*step-width, step) 
    
#     # define function to get snow fraction for a given band (z)
#     def get_snow_fraction(z):
#         print(z)
        
#         # subset the snow/ice class to this elevation
#         band_means = xr_class_map.where( ( xr_dem>=(z-width) ) & ( xr_dem<(z+width) ) )
        
#         # calculate mean. this will give % of this area that is snow throughout each time
#         band_means = band_means.mean(dim=('y','x'), skipna=True).values[:,0]
        
#         return band_means
    
#     # lets try to do this as list comprehension instead. transpose to format how my brain thinks
#     snow_fractions = np.array([get_snow_fraction(z) for z in z_bands]).T

#     ### now lets find the elevation(s) where the snow fraction crosses 0.5, for each time step
#     sf_centered = snow_fractions-0.5 #center around 0.5
#     idx_crossing = np.nonzero( np.diff( np.sign(sf_centered) )==2 )

#     # format into pandas df
#     idx_crossing = pd.DataFrame({'time':idx_crossing[0], 'z':idx_crossing[1]})
    
#     # function to interpolate to find elevation partway between z_crossing where the crossing occurs 
#     def interp_cross(idx_c):
#         i_t = idx_c['time'] #time index
#         i_z = idx_c['z'] #ele index
        
#         # calculate line equation and where it crosses 0.5
#         slope = ( sf_centered[i_t][i_z+1] - sf_centered[i_t][i_z] ) / (step) # calculate slope between this point and the next
#         crossing = z_bands[i_z] - sf_centered[i_t][i_z]/slope #calculate where that line crosses 0
        
#         return [times[i_t], crossing] # return time and elevation of crossing
    
#     # apply function. this takes only a small fraction of the time that the first step does
#     z_crossing = idx_crossing.apply(interp_cross, axis=1, raw=False, result_type='expand').rename(columns={0:'time', 1:'z'})

#     # also get the bands where there is an exact 50/50 snow/ice split
#     idx_zero = np.nonzero( sf_centered==0 )
#     idx_zero = pd.DataFrame({'time':idx_zero[0], 'z':idx_zero[1]})
#     z_zero = idx_zero.apply(lambda x: [ times[x['time']], z_bands[x['z']] ], axis=1, raw=False, result_type='expand' )
    
#     # append together, sort by time, and return
#     all_elas = pd.concat([z_crossing, z_zero], ignore_index=True).sort_values('time')
    
#     return all_elas



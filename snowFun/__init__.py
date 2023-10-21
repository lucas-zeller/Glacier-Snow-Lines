import os
import pandas as pd
import polars as pl
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
import time

def test():
    print('test')

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
    dem = dem.rename({'time':'time2'}).astype('uint16').load()#.chunk({'time':-1, 'y':50, 'x':50})

    # now count number of total pixels within each band
    df_grouped_total = []
    for i in range(len(z_mins)):
        count = np.nansum(xr.where(dem==i+1, 1, 0))
        df_grouped_total.append([i+1,count])
    df_grouped_total = pd.DataFrame(df_grouped_total, columns=['dem','count'])
    
#     print(dem.shape)
#     print(xr_class_map.shape)
    
    ## looks like this line is the main bottleneck in this function 
    # # for each band value in dem, we want to count the number of snow pixels in each time step
   
    polars = 0
    daskk = 0
    onebyone = 1
    
    # test using polars for this
    if polars:
        st = time.time()
        # do the calculation
        df_grouped_snow = ( pl.from_pandas( ( xr.Dataset({'dem': dem, 'class': xr_class_map})
                                                   .to_dataframe()
                                                   .reset_index(level='time', inplace=False)[['dem', 'time','class']] ))
                              .groupby(['dem','time'])
                              .agg(pl.sum('class'))#.alias('class_sum'))
                          ).to_pandas()
        et = time.time()
        
    elif daskk:
        st = time.time()
        # do the calculation
        df_grouped_snow = ( xr.Dataset({'dem': dem, 'class': xr_class_map})
                                  .to_dask_dataframe()
                                  .groupby(['dem', 'time'])['class'].sum().reset_index()
                          ).compute()
        et = time.time()
        
    elif onebyone: 
        st = time.time()
        
        # go through day by day and sum snow in each elevation band each day
        df_grouped_snow = []
        
        for i in range(len(times)):
            
            # grab the obs on this date
            t=times[i]
            obs=xr_class_map.sel(time=t).load()
            
            # sum the number of snow pixels within each elevation band
            sum_snow_today = ( xr.Dataset({'dem': dem, 'class': obs})
                               .to_dataframe()
                               .reset_index()[['dem','class']]
                               .groupby(['dem'])
                               .agg({"class": "sum"})
                               .reset_index() )
            
            sum_snow_today['time']=t
            
            df_grouped_snow.append(sum_snow_today)
            
        df_grouped_snow = pd.concat(df_grouped_snow)

        et = time.time()
    
    else:
        st = time.time()
        df_grouped_snow = ( xr.Dataset({'dem': dem, 'class': xr_class_map})
                               .to_dataframe()
                               .reset_index(level='time', inplace=False)[['dem', 'time','class']]
                               .groupby(['dem','time'])
                               .agg({"class": "sum"})
                               .reset_index()
                         ) 
        et = time.time()
     
    # get the execution time
#     elapsed_time = et - st
#     print('Execution time:', elapsed_time, 'seconds')
#     return df_grouped_snow
    
    
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
    
    glacier_pixels = xr_mask.sum().values
    
    ### for dates where no ela is found, we assume it is above or below glacier elevation. decide which one.
    # first, find which dates these are
    bad_times = []
    for d in times:
        if len(all_elas[all_elas['time']==d])==0:
            bad_times.append(d)
    
    # then, for each of those dates, see if there is > or < 50% snow
    snow_fracs = xr_class_map.sel(time=bad_times).sum(dim=['x','y'], skipna=True)/glacier_pixels
    
#     print(3)
    
    # for snow_fracs>0.5, make it 9999, otherwise -1
    snow_fracs = xr.where(snow_fracs>0.5, -1, 9999)
    last_elas = pd.DataFrame({'time':snow_fracs.time, 'z':snow_fracs.values})
    # last_elas = snow_fracs.to_dataframe()
    # print(last_elas)
    all_elas = pd.concat([all_elas, last_elas], ignore_index=True)
#     print(4)
    
    return all_elas


def choose_one_ELA(xr_snow, xr_dem, xr_mask, df_elas):
    
#     print(5)
    ### 5-6 is the biggest bottleneck here
    
    # calculate the aar for each date, from the snow distribution
    glacier_pixels = xr_mask.sum().values # total number of pixels on the glacier 
    aars = xr_snow.sum(dim=('y','x'))/((xr_snow.notnull()).sum(dim=['x','y'])) # percentage of pixels which are snow in each time step
    
    # for each  AAR, get the "ideal" ela
#     elevations = xr_dem.where(xr_mask>0, np.nan).dropna().values.flatten()
    elevations = xr_dem.where(xr_mask>0, np.nan).values.ravel()
    elevations = elevations[~np.isnan(elevations)]
#     print(5.1)
#     elevations = elevations[elevations>0]
    elas = np.percentile(elevations, (1-aars.values)*100)
#     print(5.2)
    # make df for date, observed aar, and the ideal ela given the observed aar
    df_results = pd.DataFrame({'time':xr_snow.time, 'aar':aars, 'ela_ideal':elas})
    
#     print(6)
    
    # Group df_elas by "time" and aggregate the "z" values into lists
    elas_grouped = df_elas.groupby("time")["z"].agg(list).reset_index()

    # Merge df_ideal and elas_grouped on the "time" column
#     print(df_results['time'].dtype)
#     print(elas_grouped['time'].dtype)
#     print(elas_grouped)
    
    # sometimes we will have no 'good' observations for this year (idk why), so we need to return essentiall an empty df
    if len(elas_grouped)>0:
        df_results = df_results.merge(elas_grouped, on="time", how="left")
    else:
        return pd.DataFrame({'time':['1900-01-01'], 'aar':[0], 'ela_ideal':[9999], 'z':[-1], 'ela':[-1]})
#         return print('huh')
        
    
    ### now we have a column in df_ideal ("z") that has a list of all elas observed on this date
    
    # Custom function to find the closest value in a list
    def find_closest_value(z_list, ela_ideal_value):
        return min(z_list, key=lambda x: abs(x - ela_ideal_value))

    # Apply the custom function to find the closest "z" value for each row
    df_results["ela"] = df_results.apply(lambda row: find_closest_value(row["z"], row["ela_ideal"]), axis=1)
#     print(df_results)
    return df_results


# wrap the above two functions into a single call
def get_the_ELAs(xr_class_map, xr_dem, xr_mask, step=20, width=1, p_snow=0.5):
    all_ELAs = extract_all_ELAs(xr_class_map, xr_dem, xr_mask, step=step, width=width, p_snow=p_snow)
    best_ELAs = choose_one_ELA(xr_class_map, xr_dem, xr_mask, all_ELAs).sort_values('time')
#     print(best_ELAs)
    return best_ELAs


def extract_band_SCA(xr_class_map, xr_dem, xr_mask, step=10):
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
        DESCRIPTION. The default is 10. The step size, in m, between zones

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

    # get the centers of each elevation band
    z_bands = np.arange( np.ceil(z_min/step)*step, np.ceil(z_max/step)*step-step, step) 
    
    # get the min, max of each elevation band
    z_mins =  z_bands-(step/2)
    z_maxs =  z_bands+(step/2)
    
    # clean up any outliers in the dem that are at the extremes outside of our bands
    dem = xr_dem.copy()
    dem = xr.where( dem>=z_maxs[-1], 0, dem)
    dem = xr.where( dem<z_mins[0], 0, dem)
    
    ### now reclassify dem into these bands, labeled 1,2,3,... 
    for i in range(len(z_mins)):
        dem = xr.where( (dem>=z_mins[i]) & (dem<z_maxs[i]), i+1, dem )

    # we need to rename the dem 'time' dimension to not be 'time' for this
    dem = dem.rename({'time':'time2'}).astype('uint16').load()#.chunk({'time':-1, 'y':50, 'x':50})

    # now count number of total pixels within each band (starting in band=1)
    band_labels = []
    total_pix_per_bands = []
    for i in range(len(z_mins)):
        count = np.nansum(xr.where(dem==i+1, 1, 0))
        total_pix_per_bands.append(count)
        band_labels.append(i+1)
    
    
    ## for each band value in dem, we want to count the number of snow pixels in each time step
    # go through day by day and sum snow in each elevation band each day
    
    # initialize df to hold results. make band_n the index
    df_all_obs = pd.DataFrame({'band_n':band_labels, 'z_min':z_mins,'z_max':z_maxs, 'total_pixels':total_pix_per_bands}).set_index('band_n')

    df_all_obs = [band_labels, z_mins, z_maxs, total_pix_per_bands]
    df_all_obs = [pd.Series(z_mins, index=band_labels),
                  pd.Series(z_maxs, index=band_labels), 
                  pd.Series(total_pix_per_bands, index=band_labels)]
    df_all_columns = ['z_min', 'z_max', 'total_pixels']
    
    for i in range(len(times)):
#         if i>0: continue
        # grab the obs on this date
        t=times[i]
        obs=xr_class_map.sel(time=t).load()

        # sum the number of snow pixels within each elevation band
        sum_snow_each_band = ( xr.Dataset({'dem': dem, 'class': obs})
                           .to_dataframe()
                           .reset_index()[['dem','class']]
                           .groupby(['dem'])
                           .agg({"class": "sum"})
                           .reset_index() )
            
        # drop band 0 (this is off-glacier areas)
        real_obs = sum_snow_each_band[sum_snow_each_band['dem']>0]
        
        # rename and format index
        real_obs = real_obs.rename(columns={"dem":"band_n"}).set_index('band_n')['class']
#         print(real_obs)
        
        # add to results df list
        df_all_obs.append(real_obs)
        df_all_columns.append(str(t)[:10])
    
    # format to dataframe (and transpose)
    df_all_obs = pd.DataFrame(df_all_obs).T

    # Set the column names
    df_all_obs.columns = df_all_columns

    # return
    return df_all_obs
    

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
    path_subregions = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate', 'Desktop', "AGVA", "RGI", "S2_subregions", "subregions.shp")
    gdf_subregions = gpd.read_file(path_subregions).to_crs("EPSG:3338")
    
    # find the subregion that intersects the rgi_geom
    correct_subregion = gdf_subregions[gdf_subregions['geometry'].contains(rgi_geom.values[0])]['id'].values[0]
    
    return correct_subregion



# get the dem of a glacier for a given year
def get_year_DEM(single_geometry, year, subregion=-1):
    
    # find which subregion we're in
    if subregion==-1:
        subregion = get_S2_subregion(single_geometry)
    
    # set folder paths, etc...
    folder_AGVA = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA")
    path_dem_base = os.path.join(folder_AGVA, "DEMs", '10m_COP_GLO30', f"region_{subregion:02d}_10m.tif")
    path_dhdt_00_05 = os.path.join(folder_AGVA, 'DEMs', "10m_thinning", "01_02_rgi60_2000-01-01_2005-01-01", "dhdt", f"Region_{subregion:02d}.tif")
    path_dhdt_05_10 = os.path.join(folder_AGVA, 'DEMs', "10m_thinning", "01_02_rgi60_2005-01-01_2010-01-01", "dhdt", f"Region_{subregion:02d}.tif")
    path_dhdt_10_15 = os.path.join(folder_AGVA, 'DEMs', "10m_thinning", "01_02_rgi60_2010-01-01_2015-01-01", "dhdt", f"Region_{subregion:02d}.tif")
    path_dhdt_15_20 = os.path.join(folder_AGVA, 'DEMs', "10m_thinning", "01_02_rgi60_2015-01-01_2020-01-01", "dhdt", f"Region_{subregion:02d}.tif")
    
    # open the 2013 dem
    dem_base = get_base_DEM(single_geometry, subregion=subregion)
    
    # function to open and clip with rioxarray
    def open_xr(path):
        xr_da = riox.open_rasterio(path).rio.clip(single_geometry, from_disk=True, drop=True)
        return xr_da   
   
    # calculate numbers years off from 2013
    dy = year-2013

    # from this, calculate how much to multiply each of the dem products
    # I can't explain in words how this work, but trust me that I thought through it and it is good
    f10 = min( 2, max(dy,-3)) 
    f15 = max( dy-f10, 0)
    f05 = max( min(dy-f10,0), -5)
    f00 = max( min(dy-f10-f05,0), -5)
    
    # open the dhdt products that are needed
    dhdt_00_05 = open_xr(path_dhdt_00_05).rename({"band":"time"}) if f00 else 0
    dhdt_05_10 = open_xr(path_dhdt_05_10).rename({"band":"time"}) if f05 else 0
    dhdt_10_15 = open_xr(path_dhdt_10_15).rename({"band":"time"}) if f10 else 0
    dhdt_15_20 = open_xr(path_dhdt_15_20).rename({"band":"time"}) if f15 else 0

    #print(2000+i, dy, f"{f00}:{f05}:{f10}:{f15}")
    dem_new = ((dem_base) + (dhdt_00_05*f00*10) + 
                            (dhdt_05_10*f05*10) + 
                            (dhdt_10_15*f10*10) + 
                            (dhdt_15_20*f15*10)  ).astype(int)

    # set the time variable
    dem_new = dem_new#.rename({"band":"time"})
    dem_new['time'] = pd.to_datetime([f"{year}-01-01"])
    
    return dem_base



# get the base dem (2013)
def get_base_DEM(single_geometry, subregion=-1):
    
    if subregion==-1:
        subregion = get_S2_subregion(single_geometry)
    
    # set folder paths, etc...
    folder_AGVA = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA")
    path_dem_base = os.path.join(folder_AGVA, "DEMs", '10m_COP_GLO30', f"region_{subregion:02d}_10m.tif")
    
    # open base dem, each of the dhdt products
    def open_xr(path):
        # I can't recall why I was intent on using the write_nodata(0) call here, but it is causing errors in the array sizes (off by 1 at times) so I am removing it
#         xr_da = riox.open_rasterio(path).rio.write_nodata(0)
#         xr_da = xr_da.rio.clip(single_geometry, from_disk=True, drop=True)
        xr_da = riox.open_rasterio(path).rio.clip(single_geometry, from_disk=True, drop=True)
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



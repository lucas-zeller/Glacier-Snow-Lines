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
from shapely.geometry import box
import matplotlib.pyplot as plt
import time
import cartopy.crs as ccrs

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
    z_bands = np.arange( np.ceil(z_min/step)*step+step/2, np.ceil(z_max/step)*step-step/2, step) 
    # print(z_bands)
    
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
    all_correct_subregion = gdf_subregions[gdf_subregions['geometry'].contains(rgi_geom.values[0])]#['id'].values[0]
    
    ### handling the cases of the overlapping rgi subregions
    
    # if no overlap, then just return the single correct subregion
    if len(all_correct_subregion)==1:
        correct_subregion = all_correct_subregion['id'].values[0]
    
    # else, return the one whose intersection is greatest
    else:
        correct_subregion = all_correct_subregion['id'].values[0] 
    # print(correct_subregion)
    return correct_subregion



# get the dem of a glacier for a given year
def get_year_DEM(single_geometry, year, subregion=-1, smoothed=1):
    
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
    # print(dem_base)
    
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
    
    ### smoothe dhdt products if requested
    if smoothed:
        dhdt_00_05 = make_smooth_dhdt(dhdt_00_05,dem_base)
        dhdt_05_10 = make_smooth_dhdt(dhdt_05_10,dem_base)
        dhdt_10_15 = make_smooth_dhdt(dhdt_10_15,dem_base)
        dhdt_15_20 = make_smooth_dhdt(dhdt_15_20,dem_base)

    # we should not be storing the x/y data as float. cast these to int
    for item in [dhdt_00_05,dhdt_05_10,dhdt_10_15,dhdt_15_20]:
        if not isinstance(item, int):
            item['y'] = item.y.astype(int)
            item['x'] = item.x.astype(int)
    
    
    dem_new = ((dem_base) + (dhdt_00_05*f00*0.01) + 
                            (dhdt_05_10*f05*0.01) + 
                            (dhdt_10_15*f10*0.01) + 
                            (dhdt_15_20*f15*0.01)  ).astype(int)
    
    # dem_new = dem_new.rename({"band":"time"})
    dem_new['time'] = pd.to_datetime([f"{year}-01-01"])
    
    return dem_new


# function to make the dhdt product an elevation-dependent dhdt, rather than 2d
def make_smooth_dhdt(dhdt_product, dem_base):
    
    if isinstance(dhdt_product, int):
        return 0
    
    # plt.imshow(dhdt_product[0])
    
    # make dem to int for this
    dem_base = dem_base.astype(int)
    
    # lets get the minimum and maximum elevation on the glacier
    z_min = np.nanmin(dem_base.where(dem_base>0))
    z_max = np.nanmax(dem_base.where(dem_base>0))
    
    # get list of full elevation range
    all_zs = np.arange(int(z_min-1),int(z_max+1),1)
    
    # calculate average dhdt for each z (within 100m)
    all_dhdts = []
    for z in all_zs:
        dhdt_avg = xr.where( (dem_base>(z-100)) & (dem_base<(z+100)), dhdt_product, np.nan).mean().values
        all_dhdts.append(dhdt_avg)
    
    # reclassify the dhdt to these new elevation-dependent values
    for i in range(len(all_zs)):
        this_z = all_zs[i]
        new_dhdt = all_dhdts[i]
        dhdt_product = dhdt_product.where(dem_base!=this_z, new_dhdt)
    
    return dhdt_product
    
    
# get the dem of a glacier for a given year
def get_year_DEM_smoothed(single_geometry, year, subregion=-1):
    
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
    # print(dem_base)
    
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

    # we should not be storing the x/y data as float. cast these to int
    for item in [dhdt_00_05,dhdt_05_10,dhdt_10_15,dhdt_15_20]:
        if not isinstance(item, int):
            item['y'] = item.y.astype(int)
            item['x'] = item.x.astype(int)
            
    ### we need to calculate the elevation-dependent dhdt for each of the products
    ### to avoid getting spotty outliers 
    dhdt_10_15 = make_smooth_dhdt(dhdt_10_15,dem_base)
    dhdt_15_20 = make_smooth_dhdt(dhdt_15_20,dem_base)
    
    dem_new = ((dem_base) + (dhdt_00_05*f00*0.01) + 
                            (dhdt_05_10*f05*0.01) + 
                            (dhdt_10_15*f10*0.01) + 
                            (dhdt_15_20*f15*0.01)  ).astype(int)
    
    # dem_new = dem_new.rename({"band":"time"})
    dem_new['time'] = pd.to_datetime([f"{year}-01-01"])
    
    return dem_new



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
    
    dem_base = open_xr(path_dem_base)*0.1 # divide by ten for scaling factor
    dem_base = dem_base.rename({"band":"time"})
    # dem_base['time'] = pd.to_datetime(["2013-01-01"])
    
    # make sure x/y is in integer, not float
    dem_base['y'] = dem_base.y.astype(int)
    dem_base['x'] = dem_base.x.astype(int)
    
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
        dem_new = ((dem_base) + (dhdt_00_05*f00*0.1) + 
                                (dhdt_05_10*f05*0.1) + 
                                (dhdt_10_15*f10*0.1) + 
                                (dhdt_15_20*f15*0.1)  ).astype(int)
        
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



# define function to extract dem values along line geometry
def sample_dem_along_line(snowline, xr_dem, increment=20):
    
    # make iterable if it is a multilinestring
    lines_geom = snowline.geometry.values[0]
    
    if lines_geom.geom_type=="LineString":
        all_lines = [lines_geom]
    else:
        all_lines = [i for i in lines_geom.geoms]

    ### create points every XYZ meters along each line segment
    # list to store all the point locations
    all_points = []
    
    for ls in all_lines:

        # get list of distances along line that you want to sample
        distances = np.arange(0, ls.length, increment) 

        # find the x/y point at each distance
        points = [ls.interpolate(distance) for distance in distances]
        all_points = all_points + points

    # make into list of x and y locations
    x_indexer = xr.DataArray([p.x for p in all_points])
    y_indexer = xr.DataArray([p.y for p in all_points]) #xr.DataArray(centroids.y, dims=["point"])

    # sample the dem at each of these points
    all_zs = xr_dem.sel(x=x_indexer, y=y_indexer, method="nearest")
    
    # return list of these elevations
    return all_zs


# function to create a single-frame base map, and return the figure and axis 
def create_base_map(width=6.5, height=5, hillshade=0, projection=None, closeup=None):
    
    # define folder and file paths
    folder_AGVA = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA")
    folder_plotting = os.path.join(folder_AGVA, 'Plotting')
    
    path_bounds = os.path.join(folder_plotting, "usa_can_boundaries", "boundary_lines.shp")
    boundary_lines = gpd.read_file(path_bounds)
    
    # open ocean shapefile
    path_ocean = os.path.join(folder_plotting, 'ne_10m_ocean', 'ne_10m_ocean.shp')
    ocean = gpd.read_file(path_ocean).to_crs("EPSG:3338")
    
    # # define the extent of the plots.
    # plot_buffer = 100000
    # plot_bounds = rgi_gdf.geometry.total_bounds
    # xlims = ( int(plot_bounds[0]-plot_buffer) , int(plot_bounds[2]+plot_buffer) )
    # ylims = ( int(plot_bounds[1]-plot_buffer) , int(plot_bounds[3]+plot_buffer) )
    
    # manual override because I want to cut out some of the alaska peninsula
    xlims = (-450000, 1661000)
    ylims = (500000, 1652000)
    
    # define what the x and y ticks are going to be
    xticks = np.arange(-300000,1500001,300000)
    yticks = np.arange(600000,1500001,300000)
    xtick_labels = xticks/1000000
    ytick_labels = yticks/1000000
    
    # if you want it even more close up
    if closeup:
        xlims = (-150000, 1600000)
        ylims = (800000, 1580000)
        xticks = np.arange(0,1500001,300000)
        yticks = np.arange(900000,1500001,300000)
        xtick_labels = xticks/1000000
        ytick_labels = yticks/1000000
    
    # create non-ocean (land) geometry
    if hillshade:
        not_ocean = box(*box(xlims[0], ylims[0], xlims[1], ylims[1]).buffer(50000).bounds)
        not_ocean = not_ocean.difference(ocean["geometry"].values[0])
        not_ocean = gpd.GeoSeries( [not_ocean], crs=ocean.crs )
    
    # open background hillshade, if wanted
    if hillshade:
        path_ne = os.path.join(folder_plotting, 'GRAY_HR_SR_OB', 'GRAY_HR_SR_OB_AA_500m.tif') 
        ne_background = riox.open_rasterio(path_ne)
        ne_background = ne_background.sel(x=slice(xlims[0]-5000, xlims[1]+5000),
                                          y=slice(ylims[1]+5000, ylims[0]-5000))
    
    # initiate figure
    fig,axs = plt.subplots(figsize=(width,height), dpi=300, subplot_kw={'projection': projection})
    
    # add background hillshade
    if hillshade:
        ne_background.plot(ax=axs, cmap='gray', vmin=-100, vmax=200, add_colorbar=False, zorder=1)
    
    # add colored shapes overlaying the ocean and land
    ocean.plot(ax=axs, color='cornflowerblue', alpha=0.3, zorder=1.2)
    
    if hillshade:
        not_ocean.plot(ax=axs, color='white', alpha=0.7, zorder=1.3)
    
    # add usa and canada boundaries
    boundary_lines.plot(ax=axs, color='black', linewidth=0.3, zorder=1.4)
    
    # add ocean boundary outline
    ocean.boundary.plot(ax=axs, color='black', linewidth=0.5, alpha=1, zorder=1.35)
    
    # set axis limits
    axs.set_xlim(xlims)
    axs.set_ylim(ylims)
    
    # set axis ticks, format marks inwards
    axs.set_xticks(xticks)
    axs.set_yticks(yticks)
    axs.set_xticklabels(xtick_labels)
    axs.set_yticklabels(ytick_labels, rotation=90, va='center')
    axs.tick_params(axis="x", pad=2, direction="in", width=1, labelsize=5, zorder=2)
    axs.tick_params(axis="y" ,pad=1, direction="in", width=1, labelsize=5, zorder=2)
    
    # set axis labels
    axs.set_xlabel(r'Easting ($ \times 10^6$ m)', size=5, labelpad=0)
    axs.set_ylabel(r'Northing ($ \times 10^6$ m)', size=5, labelpad=0)
    plt.title("")
    
    plt.tight_layout()
    
    return (fig, axs)


# function to create a 6-frame base map, and return the figure and axes
def create_annual_base_maps(width=6.5, height=3.25, hillshade=0):
    
    # define folder and file paths
    folder_AGVA = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA")
    folder_plotting = os.path.join(folder_AGVA, 'Plotting')
    
    path_bounds = os.path.join(folder_plotting, "usa_can_boundaries", "boundary_lines.shp")
    boundary_lines = gpd.read_file(path_bounds)
    
    # open ocean shapefile
    path_ocean = os.path.join(folder_plotting, 'ne_10m_ocean', 'ne_10m_ocean.shp')
    ocean = gpd.read_file(path_ocean).to_crs("EPSG:3338")
    
    # # define the extent of the plots.
    # plot_buffer = 100000
    # plot_bounds = rgi_gdf.geometry.total_bounds
    # xlims = ( int(plot_bounds[0]-plot_buffer) , int(plot_bounds[2]+plot_buffer) )
    # ylims = ( int(plot_bounds[1]-plot_buffer) , int(plot_bounds[3]+plot_buffer) )
    
    # manual override because I want to cut out some of the alaska peninsula
    xlims = (-450000, 1661000)
    ylims = (500000, 1652000)
    
    # define what the x and y ticks are going to be
    xticks = np.arange(-300000,1500001,300000)
    yticks = np.arange(600000,1500001,300000)
    xtick_labels = xticks/1000000
    ytick_labels = yticks/1000000
    
    # create non-ocean (land) geometry
    if hillshade:
        not_ocean = box(*box(xlims[0], ylims[0], xlims[1], ylims[1]).buffer(50000).bounds)
        not_ocean = not_ocean.difference(ocean["geometry"].values[0])
        not_ocean = gpd.GeoSeries( [not_ocean], crs=ocean.crs )
    
    # open background hillshade, if wanted
    if hillshade:
        path_ne = os.path.join(folder_plotting, 'GRAY_HR_SR_OB', 'GRAY_HR_SR_OB_AA_500m.tif') 
        ne_background = riox.open_rasterio(path_ne)
        ne_background = ne_background.sel(x=slice(xlims[0]-5000, xlims[1]+5000),
                                          y=slice(ylims[1]+5000, ylims[0]-5000))
    
    # initiate figure
    fig,axs = plt.subplots(2,3, figsize=(width,height), dpi=300)
    
    for ax in axs:
        for a in ax:
            # add background hillshade
            if hillshade:
                ne_background.plot(ax=a, cmap='gray', vmin=-100, vmax=200, add_colorbar=False, zorder=1)
            
            # add colored shapes overlaying the ocean and land
            ocean.plot(ax=a, color='cornflowerblue', alpha=0.3, zorder=1.2)
            
            if hillshade:
                not_ocean.plot(ax=a, color='white', alpha=0.5, zorder=1.3)
            
            # add usa and canada boundaries
            boundary_lines.plot(ax=a, color='black', linewidth=0.1, zorder=1.4)
            
            # add ocean boundary outline
            ocean.boundary.plot(ax=a, color='black', linewidth=0.2, alpha=1, zorder=1.35)
            
            # set axis limits
            a.set_xlim(xlims)
            a.set_ylim(ylims)
            
            # set axis ticks, format marks inwards
            a.set_xticks([])
            a.set_yticks([])
            a.set_xticklabels([])
            a.set_yticklabels([])
            # a.tick_params(axis="x", pad=3, direction="in", width=1, labelsize=6, zorder=2)
            # a.tick_params(axis="y" ,pad=1, direction="in", width=1, labelsize=6, zorder=2)
            
            # set axis labels
            a.set_xlabel("")
            a.set_ylabel("")
            
    axs[0,0].set_title('2018', size='6', pad=0)
    axs[0,1].set_title('2019', size='6', pad=0)
    axs[0,2].set_title('2020', size='6', pad=0)
    axs[1,0].set_title('2021', size='6', pad=0)
    axs[1,1].set_title('2022', size='6', pad=0)
    axs[1,2].set_title('Avg', size='6', pad=0)
    
    # plt.title("")
    plt.tight_layout()
    
    return (fig, axs)


# function to create a 6-frame base map, but rotated to 3 rows, 2 columns
def create_annual_base_maps_rotate(width=6.5, height=5.8, hillshade=0):
    
    # define folder and file paths
    folder_AGVA = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA")
    folder_plotting = os.path.join(folder_AGVA, 'Plotting')
    
    path_bounds = os.path.join(folder_plotting, "usa_can_boundaries", "boundary_lines.shp")
    boundary_lines = gpd.read_file(path_bounds)
    
    # open ocean shapefile
    path_ocean = os.path.join(folder_plotting, 'ne_10m_ocean', 'ne_10m_ocean.shp')
    ocean = gpd.read_file(path_ocean).to_crs("EPSG:3338")
    
    # # define the extent of the plots.
    # plot_buffer = 100000
    # plot_bounds = rgi_gdf.geometry.total_bounds
    # xlims = ( int(plot_bounds[0]-plot_buffer) , int(plot_bounds[2]+plot_buffer) )
    # ylims = ( int(plot_bounds[1]-plot_buffer) , int(plot_bounds[3]+plot_buffer) )
    
    # manual override because I want to cut out some of the alaska peninsula
    xlims = (-450000, 1661000)
    ylims = (500000, 1652000)
    
    # define what the x and y ticks are going to be
    xticks = np.arange(-300000,1500001,300000)
    yticks = np.arange(600000,1500001,300000)
    xtick_labels = xticks/1000000
    ytick_labels = yticks/1000000
    
    # create non-ocean (land) geometry
    if hillshade:
        not_ocean = box(*box(xlims[0], ylims[0], xlims[1], ylims[1]).buffer(50000).bounds)
        not_ocean = not_ocean.difference(ocean["geometry"].values[0])
        not_ocean = gpd.GeoSeries( [not_ocean], crs=ocean.crs )
    
    # open background hillshade, if wanted
    if hillshade:
        path_ne = os.path.join(folder_plotting, 'GRAY_HR_SR_OB', 'GRAY_HR_SR_OB_AA_500m.tif') 
        ne_background = riox.open_rasterio(path_ne)
        ne_background = ne_background.sel(x=slice(xlims[0]-5000, xlims[1]+5000),
                                          y=slice(ylims[1]+5000, ylims[0]-5000))
    
    # initiate figure
    fig,axs = plt.subplots(3,2, figsize=(width,height), dpi=300)
    
    for ax in axs:
        for a in ax:
            # add background hillshade
            if hillshade:
                ne_background.plot(ax=a, cmap='gray', vmin=-100, vmax=200, add_colorbar=False, zorder=1)
            
            # add colored shapes overlaying the ocean and land
            ocean.plot(ax=a, color='cornflowerblue', alpha=0.3, zorder=1.2)
            
            if hillshade:
                not_ocean.plot(ax=a, color='white', alpha=0.5, zorder=1.3)
            
            # add usa and canada boundaries
            boundary_lines.plot(ax=a, color='black', linewidth=0.1, zorder=1.4)
            
            # add ocean boundary outline
            ocean.boundary.plot(ax=a, color='black', linewidth=0.2, alpha=1, zorder=1.35)
            
            # set axis limits
            a.set_xlim(xlims)
            a.set_ylim(ylims)
            
            # set axis ticks, format marks inwards
            a.set_xticks([])
            a.set_yticks([])
            a.set_xticklabels([])
            a.set_yticklabels([])
            # a.tick_params(axis="x", pad=3, direction="in", width=1, labelsize=6, zorder=2)
            # a.tick_params(axis="y" ,pad=1, direction="in", width=1, labelsize=6, zorder=2)
            
            # set axis labels
            a.set_xlabel("")
            a.set_ylabel("")
            
    axs[0,0].set_title('2018', size='6', pad=0)
    axs[0,1].set_title('2019', size='6', pad=0)
    axs[1,0].set_title('2020', size='6', pad=0)
    axs[1,1].set_title('2021', size='6', pad=0)
    axs[2,0].set_title('2022', size='6', pad=0)
    axs[2,1].set_title('Avg', size='6', pad=0)
    
    # plt.title("")
    plt.tight_layout()
    
    return (fig, axs)


# function to create a 4-frame base map, and return the figure and axes
def create_three_base_maps(width=6.5, height=3.8, hillshade=0):
    
    # define folder and file paths
    folder_AGVA = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA")
    folder_plotting = os.path.join(folder_AGVA, 'Plotting')
    
    path_bounds = os.path.join(folder_plotting, "usa_can_boundaries", "boundary_lines.shp")
    boundary_lines = gpd.read_file(path_bounds)
    
    # open ocean shapefile
    path_ocean = os.path.join(folder_plotting, 'ne_10m_ocean', 'ne_10m_ocean.shp')
    ocean = gpd.read_file(path_ocean).to_crs("EPSG:3338")
    
    # # define the extent of the plots.
    # plot_buffer = 100000
    # plot_bounds = rgi_gdf.geometry.total_bounds
    # xlims = ( int(plot_bounds[0]-plot_buffer) , int(plot_bounds[2]+plot_buffer) )
    # ylims = ( int(plot_bounds[1]-plot_buffer) , int(plot_bounds[3]+plot_buffer) )
    
    # manual override because I want to cut out some of the alaska peninsula
    xlims = (-450000, 1661000)
    ylims = (500000, 1652000)
    
    # define what the x and y ticks are going to be
    xticks = np.arange(-300000,1500001,300000)
    yticks = np.arange(600000,1500001,300000)
    xtick_labels = xticks/1000000
    ytick_labels = yticks/1000000
    
    # create non-ocean (land) geometry
    if hillshade:
        not_ocean = box(*box(xlims[0], ylims[0], xlims[1], ylims[1]).buffer(50000).bounds)
        not_ocean = not_ocean.difference(ocean["geometry"].values[0])
        not_ocean = gpd.GeoSeries( [not_ocean], crs=ocean.crs )
    
    # open background hillshade, if wanted
    if hillshade:
        path_ne = os.path.join(folder_plotting, 'GRAY_HR_SR_OB', 'GRAY_HR_SR_OB_AA_500m.tif') 
        ne_background = riox.open_rasterio(path_ne)
        ne_background = ne_background.sel(x=slice(xlims[0]-5000, xlims[1]+5000),
                                          y=slice(ylims[1]+5000, ylims[0]-5000))
    
    # initiate figure
    fig,axs = plt.subplots(2,2, figsize=(width,height), dpi=300)
    
    for a in [axs[0,1], axs[1,1], axs[1,0]]:
    
        # add background hillshade
        if hillshade:
            ne_background.plot(ax=a, cmap='gray', vmin=-100, vmax=200, add_colorbar=False, zorder=1)

        # add colored shapes overlaying the ocean and land
        ocean.plot(ax=a, color='cornflowerblue', alpha=0.3, zorder=1.2)

        if hillshade:
            not_ocean.plot(ax=a, color='white', alpha=0.5, zorder=1.3)

        # add usa and canada boundaries
        boundary_lines.plot(ax=a, color='black', linewidth=0.1, zorder=1.4)

        # add ocean boundary outline
        ocean.boundary.plot(ax=a, color='black', linewidth=0.2, alpha=1, zorder=1.35)

        # set axis limits
        a.set_xlim(xlims)
        a.set_ylim(ylims)

        # set axis ticks, format marks inwards
        a.set_xticks([])
        a.set_yticks([])
        a.set_xticklabels([])
        a.set_yticklabels([])
        # a.tick_params(axis="x", pad=3, direction="in", width=1, labelsize=6, zorder=2)
        # a.tick_params(axis="y" ,pad=1, direction="in", width=1, labelsize=6, zorder=2)

        # set axis labels
        a.set_xlabel("")
        a.set_ylabel("")
    
    # plt.title("")
    plt.tight_layout()
    
    return (fig, axs)


def create_three_base_maps_row(width=6.5, height=2.5, hillshade=0, projection=None):
    
    # define folder and file paths
    folder_AGVA = os.path.join('C:',os.sep,'Users','lzell','OneDrive - Colostate','Desktop',"AGVA")
    folder_plotting = os.path.join(folder_AGVA, 'Plotting')
    
    path_bounds = os.path.join(folder_plotting, "usa_can_boundaries", "boundary_lines.shp")
    boundary_lines = gpd.read_file(path_bounds)
    
    # open ocean shapefile
    path_ocean = os.path.join(folder_plotting, 'ne_10m_ocean', 'ne_10m_ocean.shp')
    ocean = gpd.read_file(path_ocean).to_crs("EPSG:3338")
    
    # # define the extent of the plots.
    # plot_buffer = 100000
    # plot_bounds = rgi_gdf.geometry.total_bounds
    # xlims = ( int(plot_bounds[0]-plot_buffer) , int(plot_bounds[2]+plot_buffer) )
    # ylims = ( int(plot_bounds[1]-plot_buffer) , int(plot_bounds[3]+plot_buffer) )
    
    # manual override because I want to cut out some of the alaska peninsula
    xlims = (-450000, 1661000)
    ylims = (500000, 1652000)
    
    # define what the x and y ticks are going to be
    xticks = np.arange(-300000,1500001,300000)
    yticks = np.arange(600000,1500001,300000)
    xtick_labels = xticks/1000000
    ytick_labels = yticks/1000000
    
    # create non-ocean (land) geometry
    if hillshade:
        not_ocean = box(*box(xlims[0], ylims[0], xlims[1], ylims[1]).buffer(50000).bounds)
        not_ocean = not_ocean.difference(ocean["geometry"].values[0])
        not_ocean = gpd.GeoSeries( [not_ocean], crs=ocean.crs )
    
    # open background hillshade, if wanted
    if hillshade:
        path_ne = os.path.join(folder_plotting, 'GRAY_HR_SR_OB', 'GRAY_HR_SR_OB_AA_500m.tif') 
        ne_background = riox.open_rasterio(path_ne)
        ne_background = ne_background.sel(x=slice(xlims[0]-5000, xlims[1]+5000),
                                          y=slice(ylims[1]+5000, ylims[0]-5000))
    
    # initiate figure
    fig,axs = plt.subplots(1,3, figsize=(width,height), dpi=300, subplot_kw={'projection': projection})
    
    for a in axs:
    
        # add background hillshade
        if hillshade:
            ne_background.plot(ax=a, cmap='gray', vmin=-100, vmax=200, add_colorbar=False, zorder=1)

        # add colored shapes overlaying the ocean and land
        ocean.plot(ax=a, color='cornflowerblue', alpha=0.3, zorder=1.2)

        if hillshade:
            not_ocean.plot(ax=a, color='white', alpha=0.5, zorder=1.3)

        # add usa and canada boundaries
        boundary_lines.plot(ax=a, color='black', linewidth=0.1, zorder=1.4)

        # add ocean boundary outline
        ocean.boundary.plot(ax=a, color='black', linewidth=0.2, alpha=1, zorder=1.35)

        # set axis limits
        a.set_xlim(xlims)
        a.set_ylim(ylims)

        # set axis ticks, format marks inwards
        a.set_xticks([])
        a.set_yticks([])
        a.set_xticklabels([])
        a.set_yticklabels([])
        # a.tick_params(axis="x", pad=3, direction="in", width=1, labelsize=6, zorder=2)
        # a.tick_params(axis="y" ,pad=1, direction="in", width=1, labelsize=6, zorder=2)

        # set axis labels
        a.set_xlabel("")
        a.set_ylabel("")
    
    # plt.title("")
    plt.tight_layout()
    
    return (fig, axs)






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



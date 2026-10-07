#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code implements a preprocessing pipeline for an MLP Model predicting rainfall.  The pipeline is as follows:
    
    1. Days and times of day are converted to sin/cos forms along with longitude and latitude
    2. GOES rain data is forward filled
    3. GOES cloud data NaNs from failed data retrieval are filled with 0's and a column indicating the failure
        to retrieve is added.
    4. A day/night cycle is added.
    5. GOES rain features are lagged 10/20/40 and cloud features are lagged 5/10/25.  
    6. HRRR is lagged 60
    7. Difference and Difference of Differences at distance 10 is computed for GOES rain rate and cloud height
    
    Features are broken into "Rain Features", "Cloud Features", and "Meta Features"

@author: Safaan Sadiq
"""

import pandas as pd
import numpy as np
import pvlib as pv


def preprocess(input):

    #df=pd.read_parquet('train_rows.parquet')

    df=pd.read_parquet(input)

    #df_original=pd.read_parquet('train_rows.parquet')

    c_cols=['site_id','timestamp']
    
    m_cols=['hrrr_temperature_c','hrrr_pressure_hpa','hrrr_humidity_pct'] # Meta/Additional Features
    r_cols=['goes_rain_rate_mm_hr','goes_rain_rate_max_50km','hrrr_rain_rate_mm_hr','hrrr_age_min'] # Rainfall Features
    cl_cols=['goes_cod','goes_cod_max_50km','goes_cloud_top_height_m',
        'goes_cod_gradient_per_km','goes_age_min','hrrr_cloud_liquid_water_kg_m2','hrrr_age_min'] # Cloud Features
    
    df['date']=df['timestamp'].dt.date


    # Transform timing into sin,cos format
    df['day_s']=np.sin(2*np.pi*df['day_of_year']/365)
    df['day_c']=np.cos(2*np.pi*df['day_of_year']/365)
    m_cols.extend(['day_s','day_c'])


    df['time_s']=np.sin(2*np.pi*(60*df['hour_utc']+df['minute_utc'])/1440)
    df['time_c']=np.cos(2*np.pi*(60*df['hour_utc']+df['minute_utc'])/1440)
    m_cols.extend(['time_s','time_c'])

    df=df.drop(columns=['day_of_year','hour_utc', 'minute_utc'])

    # Day/Night Cycle
    test=pv.solarposition.get_solarposition(time=df['timestamp'], latitude=df['lat'],longitude=df['lon'])
    df['is_day']=(test['apparent_elevation'].to_numpy()>0).astype(int)
    del test
    cl_cols.extend(['is_day'])

    # Transform latitude and longitude into sin,cos format
    df['lon_cos']=np.cos(np.radians(df['lon']))
    df['lon_sin']=np.sin(np.radians(df['lon']))
    df['lat_cos']=np.cos(np.radians(df['lat']))
    df['lat_sin']=np.sin(np.radians(df['lat']))
    m_cols.extend(['lon_cos','lon_sin','lat_cos','lat_sin'])

    df=df.drop(columns=['lat','lon'])


    # Forward filling data
    mask=df['goes_rain_rate_mm_hr'].notna()
    df.loc[mask,'goes_cod']=df['goes_cod'].shift(1)
    df.loc[mask,'goes_cod_max_50km']=df['goes_cod_max_50km'].shift(1)
    df.loc[mask,'goes_cloud_top_height_m']=df['goes_cloud_top_height_m'].shift(1)
    df.loc[mask,'goes_cod_gradient_per_km']=df['goes_cod_gradient_per_km'].shift(1)
    del mask
    
    df['goes_rain_rate_mm_hr']=df.groupby(['site_id','date'])['goes_rain_rate_mm_hr'].ffill()
    df['goes_rain_rate_max_50km']=df.groupby(['site_id','date'])['goes_rain_rate_max_50km'].ffill()


    # GOES Cloud Data
    df['goes_miss_cod']=df['goes_cod'].isna().astype(int)
    df['goes_miss_cod50km']=df['goes_cod_max_50km'].isna().astype(int)
    df['goes_miss_cth']=df['goes_cloud_top_height_m'].isna().astype(int)
    cl_cols.extend(['goes_miss_cod', 'goes_miss_cod50km', 'goes_miss_cth'])

    df['goes_cloud_top_height_m']=df['goes_cloud_top_height_m'].fillna(0.0)
    df['goes_cod']=df['goes_cod'].fillna(0.0)
    df['goes_cod_max_50km']=df['goes_cod_max_50km'].fillna(0.0)
    df['goes_cod_gradient_per_km']=df['goes_cod_gradient_per_km'].fillna(0.0)

    
    # Lagged Features
        # GOES Vars
    df['grrm_lag10']=df.groupby(['site_id','date'])['goes_rain_rate_mm_hr'].shift(10)
    df['grrm_lag20']=df.groupby(['site_id','date'])['goes_rain_rate_mm_hr'].shift(20)
    df['grrm_lag40']=df.groupby(['site_id','date'])['goes_rain_rate_mm_hr'].shift(40)
    r_cols.extend(['grrm_lag10','grrm_lag20','grrm_lag40'])

    df['grrmax_lag10']=df.groupby(['site_id','date'])['goes_rain_rate_max_50km'].shift(10)
    df['grrmax_lag20']=df.groupby(['site_id','date'])['goes_rain_rate_max_50km'].shift(20)
    df['grrmax_lag40']=df.groupby(['site_id','date'])['goes_rain_rate_max_50km'].shift(40)
    r_cols.extend(['grrmax_lag10','grrmax_lag20','grrmax_lag40'])

    df['gcod_lag5']=df.groupby(['site_id','date'])['goes_cod'].shift(5)
    df['gcod_lag10']=df.groupby(['site_id','date'])['goes_cod'].shift(10)
    df['gcod_lag25']=df.groupby(['site_id','date'])['goes_cod'].shift(25)
    cl_cols.extend(['gcod_lag5','gcod_lag10','gcod_lag25'])
    
    df['gcod50_lag5']=df.groupby(['site_id','date'])['goes_cod_max_50km'].shift(5)
    df['gcod50_lag10']=df.groupby(['site_id','date'])['goes_cod_max_50km'].shift(10)
    df['gcod50_lag25']=df.groupby(['site_id','date'])['goes_cod_max_50km'].shift(25)
    cl_cols.extend(['gcod50_lag5','gcod50_lag10','gcod50_lag25'])

    df['gcodg_lag5']=df.groupby(['site_id','date'])['goes_cod_gradient_per_km'].shift(5)
    df['gcodg_lag10']=df.groupby(['site_id','date'])['goes_cod_gradient_per_km'].shift(10)
    df['gcodg_lag25']=df.groupby(['site_id','date'])['goes_cod_gradient_per_km'].shift(25)
    cl_cols.extend(['gcodg_lag5','gcodg_lag10','gcodg_lag25'])

    df['gcth_lag5']=df.groupby(['site_id','date'])['goes_cloud_top_height_m'].shift(5)
    df['gcth_lag10']=df.groupby(['site_id','date'])['goes_cloud_top_height_m'].shift(10)
    df['gcth_lag25']=df.groupby(['site_id','date'])['goes_cloud_top_height_m'].shift(25)
    cl_cols.extend(['gcth_lag5','gcth_lag10','gcth_lag25'])

        # HRRR Vars
    df['hrrm_lag']=df.groupby(['site_id','date'])['hrrr_rain_rate_mm_hr'].shift(60)
    df['hh_lag']=df.groupby(['site_id','date'])['hrrr_humidity_pct'].shift(60)
    df['ht_lag']=df.groupby(['site_id','date'])['hrrr_temperature_c'].shift(60)
    df['hp_lag']=df.groupby(['site_id','date'])['hrrr_pressure_hpa'].shift(60)
    df['hclw_lag']=df.groupby(['site_id','date'])['hrrr_cloud_liquid_water_kg_m2'].shift(60)
    r_cols.extend(['hrrm_lag'])
    m_cols.extend(['hh_lag','ht_lag','hp_lag'])
    cl_cols.extend(['hclw_lag'])

    
    # Difference Features
        # GOES
    df['grrm_dif']=df['goes_rain_rate_mm_hr']-df['grrm_lag10'] # Rate of Change of rainfall
    df['grrm_acc']=df['grrm_dif']-df.groupby(['site_id','date'])['grrm_dif'].shift(10) # Acceleration of rainfall
    r_cols.extend(['grrm_dif','grrm_acc'])

    df['gcth_dif']=df['goes_cloud_top_height_m']-df['gcth_lag5']
    df['gcth_acc']=df['gcth_dif']-df.groupby(['site_id','date'])['gcth_dif'].shift(5)
    cl_cols.extend(['gcth_dif','gcth_acc'])


    # Clean Up
    df=df.drop(columns=['goes_source_timestamp', 'hrrr_source_timestamp', 'date'])
    df=df[df.is_forecast_origin == True]
    df=df.drop(columns=['is_forecast_origin'])

    return df, c_cols, r_cols, cl_cols, m_cols



if __name__=='__main__':
    
    df=preprocess('train_rows.parquet')



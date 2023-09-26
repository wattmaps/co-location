''' ============================
Import packages and set directory
============================ '''
import os, sys
from dotenv import load_dotenv
import time
import requests
import pandas as pd
import math
import numpy as np

thisDir = os.path.abspath(os.curdir)
print('Working directory:  ', thisDir)

SAM_Assumptions_SolarPV_Hourly = __import__('04_SAM_Assumptions_SolarPV_Hourly')

dotenv_path = os.path.join(os.path.dirname(thisDir), '.env') 
load_dotenv(dotenv_path)

''' ============================
Select whether to save NSRDB weather file
============================ ''' 

save = 'yes'

''' ============================
Select PV technology to simulate and column name prefix
============================ ''' 

tech = 'tracking_PVwatts' 

techAssumptions_dict = {'tracking_PVwatts' : SAM_Assumptions_SolarPV_Hourly.singleAxisTracking}

colNamePrefix = 'CF_status'

''' ============================
Select range of PIDs to run
============================ '''

PID_start = 1
PID_end = PID_start + 1335

''' ============================
Select NSRDB API request inputs
============================ ''' 

## ==============  Set user inputs ================
# Set solar radiation year
year = '2014'

# Set all variables as strings
# Spaces must be replaced with '+', i.e., change 'John Smith' to 'John+Smith'
your_name = 'FirstName+LastName'
reason_for_use = 'colocation+assessment'
your_affiliation = 'UCSB'
your_email = 'email@bren.ucsb.edu'

# Set NSRDB API key
# Request key and store in .env file
USER_API = os.getenv('user_api')
api_key = USER_API

mailing_list = 'true'

## ============== DO NOT CHANGE ANYTHING BELOW THIS =============
# Select attributes to extract (e.g., dhi, ghi, etc.), separated by commas
attributes = 'ghi,dhi,dni,wind_speed,air_temperature,solar_zenith_angle'

# Set leap year to true or false
leap_year = 'false'

# Set time interval in minutes, valid intervals are 30 (half hourly) & 60 (hourly)
interval = '60'

# Set Coordinated Universal Time (UTC) to true or false (false uses local time zone of data)
# To use NSRDB data in SAM, you must specify UTC as 'false'
utc = 'false'

''' ============================
Read csv of point geometries
============================ '''

loc_filename = os.path.join(thisDir, 'data', 'pid', 'us_PID_cords.csv')
df_loc = pd.read_csv(loc_filename, engine = 'python')

# Add columns for output capacity factors
if colNamePrefix not in df_loc.columns:
    print('Add new columns for status and hourly capacity factor')
    df_loc[colNamePrefix] = np.nan
    print('Add new columns for hourly capacity factors')
    new_cols = [f'cf_hour_{i}' for i in range(8760)]
    df_loc = df_loc.reindex(columns = df_loc.columns.tolist() + new_cols)
    
# Create a filepath to where you want to save simulation output for a given year
output_filename = os.path.join(thisDir, 'data', 'SAM', 'SAM_solar_' + year + '.csv')

''' ============================
Define function for single simulation given a PID
============================ '''

def runSimulation(PID):
    # Define df_loc as a global value so you can reassign it
    # global df_loc

    # Execute if there is no status value for given PID
    if math.isnan(df_loc.loc[df_loc['PID'] == PID]['cf_hour_0'].iloc[0]):
        
        print('PID: ', PID)
                
        lat = df_loc.loc[df_loc['PID'] == PID, 'lat'].iloc[0]
        lon = df_loc.loc[df_loc['PID'] == PID, 'lon'].iloc[0]
        p_cap = df_loc.loc[df_loc['PID'] == PID, 'p_cap'].iloc[0]

        ''' ============================
        Request data for a given year and location (PID) using NSRDB API
        ============================ '''

        # Set file name and filepath
        NSRDB_wf_folder = os.path.join(thisDir, 'data', 'NSRDB', 'solarTimeSeries' + year)
        NSRDB_wf = os.path.join(NSRDB_wf_folder, f'solar_{year}_{str(PID)}.csv')
        
        # If no file exists for this PID, then use API
        if not(os.path.exists(NSRDB_wf)):
            
            # Create directory if it does not exist
            if not os.path.exists(NSRDB_wf_folder):
                 os.makedirs(NSRDB_wf_folder)
                 print('Creating new folder ' + NSRDB_wf_folder)

            # Declare url string and store as csv
            url = 'https://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes)
            df = pd.read_csv(url, engine = 'python')

            # Save df to csv
            if save == 'yes':
                df.to_csv(NSRDB_wf)    
        
        # Otherwise, read weather file from csv
        else:
            print('Reading weather file from saved csv')
            df = pd.read_csv(NSRDB_wf, engine = 'python')
    
        timezone, elevation = float(df.loc[0, 'Local Time Zone']), float(df.loc[0, 'Elevation'])

        # Drop metadata row
        df.drop(0, inplace = True)

        # Select first row as header
        new_header = df.loc[1]
        # Subset data minus header
        df_data = df[1:]
        # Set first row as header
        df_data.columns = new_header

        # Convert to numeric
        df_data = df_data.apply(pd.to_numeric) 
    
        if leap_year == 'false':
            minInYear = 525600 # - minutes in a non-leap year
        else:
            minInYear = 527040 # + minutes in a leap year

        # Set time index in df
        df_data = df_data.set_index(pd.date_range('1/1/{yr}'.format(yr=year), freq=interval+'Min', periods=minInYear/int(interval)))

        ''' ============================
        Call SAM simulation function from module 
        ============================ '''

        simulationFunct = techAssumptions_dict[tech]
        capacity_factor = simulationFunct(homePath = thisDir,
                                          lat_in = lat,
                                          lon_in = lon,
                                          timezone_in = timezone,
                                          elevation_in = elevation,
                                          df_data_in = df_data)

        # Filter for PID in df_loc dataframe and select columns from cf_hour_0 forward
        selected_PID_row = df_loc.loc[df_loc['PID']== PID, 'cf_hour_0':]
        # Reshape capacity_factor series from 8760 rows and 1 column to 8760 columns and 1 row
        capacity_factor_reshape = np.transpose(capacity_factor)
        # Update df with reshaped df, where selected_PID_row has same dimensions as capacity_factor_reshape after taking transpose
        selected_PID_row[:] = capacity_factor_reshape
        # Update original df with values for hourly capacity factor
        df_loc.loc[df_loc['PID'] == PID, 'cf_hour_0':] = selected_PID_row
        # Add a successful run indicator in colNamePrefix column in df
        df_loc.loc[df_loc['PID'] == PID, colNamePrefix] = 1

''' ============================
Define function that runs simulation given a list of PIDs 
    -  If an error occurs, save to csv then pause before another attempt
    -  If it runs without error, save to csv
============================ '''

def runSimulationLoop(PID_list):
    i = 0 
    for PID in PID_list:
        while True:
            try: 
                i = i + 1
                # print(i)
                runSimulation(PID)
                if i == 300:
                    df_loc.to_csv(output_filename, index=False)
                    print('Saved to csv')
                    i = 0
            except Exception as exc:
                print(exc)
                # Save df to csv
                df_loc.to_csv(output_filename, index=False)
                # PAUSE
                time.sleep(5)
                continue
            break
             
    # Save df to csv
    df_loc.to_csv(output_filename, index = False)
        
''' ============================
Execute function
============================ '''

start_time = time.time()
PID_list_in = list(range(PID_start, PID_end, 1))
runSimulationLoop(PID_list_in)  
end_time = time.time()
print('Time taken:', end_time - start_time, 'seconds')
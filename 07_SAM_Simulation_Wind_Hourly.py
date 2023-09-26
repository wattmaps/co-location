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
from PySAM.ResourceTools import SRW_to_wind_data

thisDir = os.path.abspath(os.curdir)
print('Working directory:  ', thisDir)

SAM_Assumptions_Wind_Hourly = __import__('06_SAM_Assumptions_Wind_Hourly')

dotenv_path = os.path.join(os.path.dirname(thisDir), '.env')
load_dotenv(dotenv_path)

''' ============================
Select whether to save WIND Toolkit weather file
============================ ''' 

save = 'yes'

''' ============================
Read SAM Assumptions
============================ '''

techAssumptions_dict = {'onshore_wind' : SAM_Assumptions_Wind_Hourly.onshoreWind}
tech = 'onshore_wind'
colNamePrefix = 'CF_status' 

''' ============================
Select range of PIDs to run
============================ '''

PID_start = 1
PID_end = PID_start + 1335

''' ============================
Select WIND Toolkit API request inputs
============================ '''

## ==============  Set user inputs ================

# Set WIND Toolkit API key
# Request key and store in .env file
USER_API = os.getenv('user_api')
api_key = USER_API
your_email = 'email@bren.ucsb.edu'

# Set year
year = '2012'
# wtk_folder = str(wtk_folder_path) + str(year)

# Set Coordinated Universal Time (UTC) to true or false (false uses local time zone of data)
# To use WIND Toolkit data in SAM, you must specify UTC as 'false'
utc = 'false'

# Set leap year to true or false
leap_year = 'false' 

# Set wind hub height parameter
hubheight = 100

# Set time interval in minutes, valid intervals are 30 (half hourly) & 60 (hourly)
# interval = '60'

''' ============================
Read csv of point geometries
============================ ''' 

loc_filename = os.path.join(thisDir, 'data', 'pid', 'us_PID_cords.csv')
df_loc = pd.read_csv(loc_filename)

# Add columns for output CFs if it's not there
if colNamePrefix not in df_loc.columns:
    print('Add new columns for status and hourly capacity factor')
    df_loc[colNamePrefix] = np.nan
    print('Add new columns for hourly capacity factors')
    new_cols = [f'cf_hour_{i}' for i in range(8760)]
    df_loc = df_loc.reindex(columns = df_loc.columns.tolist() + new_cols)

# Create a filepath to where you want to save simulation output for a given year
output_filename = os.path.join(thisDir, 'data', 'SAM', 'SAM_wind_' + year + '.csv')

''' ============================
Define function for single simulation given a PID
============================ '''

def runSimulation(PID):

    # Execute if there is no status value for given PID
    if math.isnan(df_loc.loc[df_loc['PID'] == PID]['cf_hour_0'].iloc[0]):
        
        print('PID: ', PID)
                
        lat = df_loc.loc[df_loc['PID'] == PID, 'lat'].iloc[0]
        lon = df_loc.loc[df_loc['PID'] == PID, 'lon'].iloc[0]
        # p_cap = df_loc.loc[df_loc['PID'] == PID, 'p_cap'].iloc[0]
        # p_cap_kw = p_cap * 1000

        ''' ============================
        Request data for a given year and location (PID) using WIND Toolkit API
        ============================ '''
        
        # Set filename and path
        wtk_wf_folder = os.path.join(thisDir, 'data', 'Windtoolkit', 'windTimeSeries' + year)
        wtk_wf = os.path.join(wtk_wf_folder, f'wind_{year}_{str(PID)}.csv')
        
        # If no file exists for this PID, then use API
        if not(os.path.exists(wtk_wf)):
            
            # Create directory if it does not exist
            if not os.path.exists(wtk_wf_folder):
                 os.makedirs(wtk_wf_folder)
                 print('Creating new folder ' + wtk_wf_folder)

            # Declare url string and store as csv
            url = f'https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-srw-download?api_key={api_key}&lat={lat}&lon={lon}&year={year}&email={your_email}'
            response = requests.get(url)
            
            # Save data to srw
            if save == 'yes':
                with open(wtk_wf, 'w') as f:
                    f.write(response.text)
            wind_wf_byte = wtk_wf.encode()

        # Otherwise, read weather file from srw
        else:
            print('Reading weather file from saved srw')
            wind_wf_byte = wtk_wf.encode()

        ''' ============================
        Call SAM simulation function from module 
        ============================ '''

        simulationFunct = techAssumptions_dict[tech]
        capacity_factor = simulationFunct(homePath = thisDir,
                                           lat_in = lat,
                                           lon_in = lon,
                                           data_in = wind_wf_byte)
        print(f'Completed {PID}')
        # Filter for PID in df_loc dataframe and select columns from cf_hour_0 forward
        selected_PID_row = df_loc.loc[df_loc['PID'] == PID, 'cf_hour_0':]
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
    df_loc.to_csv(output_filename, index=False)
    
''' ============================
Execute function
============================ '''

start_time = time.time()
PID_list_in = list(range(PID_start, PID_end, 1))
runSimulationLoop(PID_list_in)  
end_time = time.time()
print('Time taken:', end_time - start_time, 'seconds')
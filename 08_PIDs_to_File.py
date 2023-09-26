''' ============================
Import packages and set directory
============================ '''
import os
import time as time
import pandas as pd

thisDir = os.path.abspath(os.curdir)
print('Working directory:  ', thisDir)

''' ============================
Read capacity factor csv files
============================ '''

# Change filepaths below to 'data/SAM/SAM_solar_2012.csv', 'data/SAM/SAM_solar_2013.csv', 'data/SAM/SAM_solar_2014.csv'
df_cf_2012 = pd.read_csv(os.path.join(thisDir, 'data', 'SAM', 'SAM_wind_2012.csv'))
df_cf_2013 = pd.read_csv(os.path.join(thisDir, 'data', 'SAM', 'SAM_wind_2013.csv'))
df_cf_2014 = pd.read_csv(os.path.join(thisDir, 'data', 'SAM', 'SAM_wind_2014.csv'))

# Define folder where output files will be saved
# Change filepath below to 'data/Solar_Capacity_Factors'
folder = os.path.join(thisDir, 'data', 'SAM', 'Wind_Capacity_Factors')

''' ============================
Reshape data
============================ '''

# Initialize counter variable
counter = 0

# Loop through each row of df and extract data for that PID
for i in range(df_cf_2012.shape[0]):
    pid_data_2012 = df_cf_2012.iloc[i, 5:].values.tolist()
    pid_data_2013 = df_cf_2013.iloc[i, 5:].values.tolist()
    pid_data_2014 = df_cf_2014.iloc[i, 5:].values.tolist()
    pid = df_cf_2012.iloc[i, 0]
    
    # Create df containing hourly energy data
    pid_df_2012 = pd.DataFrame({'energy_generation': pid_data_2012})
    pid_df_2013 = pd.DataFrame({'energy_generation': pid_data_2013})
    pid_df_2014 = pd.DataFrame({'energy_generation': pid_data_2014})

    # Add hour column
    pid_df_2012['hour'] = range(1, 8761)
    pid_df_2013['hour'] = range(1, 8761)
    pid_df_2014['hour'] = range(1, 8761)

    # Concatenate df time series
    pid_3yr_df = pd.concat([pid_df_2012, pid_df_2013, pid_df_2014])

    # Replicate df time series 8 times
    pid_24yr_df = pid_3yr_df.append([pid_3yr_df] * 7, ignore_index = True)

    # Replicate df time series and reset 'hour' and index
    pid_25y_df = pid_24yr_df.append(pid_df_2012, ignore_index = True)

    # Create year sequence 
    sequence = pd.Series(range(1, 26)).repeat(8760)
    year_df = pd.DataFrame({'year': sequence}).reset_index()
    year_df[['year']]
    
    # Add year column
    pid_df = pd.concat([pid_25y_df, year_df], axis = 1)
    pid_df[['hour', 'year', 'energy_generation']]

    # Reshape df to wide
    pid_df_wide = pd.pivot(pid_df, index = 'hour', columns = 'year', values = 'energy_generation')

    # Save new df as a CSV file with location name as file name
    file_path = os.path.join(folder, f'capacity_factor_PID{pid}.csv')
    pid_df_wide.to_csv(file_path)
    
    counter += 1
    # Increment counter and print an update statement every 200 files saved
    if counter % 200 == 0:
        print(f"Saved files for PID {counter}.")
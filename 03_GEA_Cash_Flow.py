''' ============================
Import packages and set directory
============================ '''
import os, sys
import pandas as pd

thisDir = os.path.abspath(os.curdir)
print('Working directory:  ', thisDir)

# Define input folder
inputFolder = os.path.join(thisDir, 'data')

# Define GEA regions by code
gea_region_codes = ['CAMXc', 'NWPPc', 'AZNMc', 'RMPAc', 'ERCTc', 'SPSOc', 'SPNOc', 
                    'MROWc', 'MROEc', 'SRMWc', 'SRMVc', 'SRTVc', 'SRSOc', 'FRCCc',
                    'RFCWc', 'RFCMc', 'SRVCc', 'RFCEc', 'NYSTc', 'NEWEc']

# Set year, phaseout, and scenario
# year = 2025
year = 2030
ptc_phaseout = True
# ptc_phaseout = False
# scenario = 'Mid-case'
scenario = 'Electrification'

''' ============================
For loop by GEA region
============================ '''

for gea in gea_region_codes:

    # Define production tax credit (in USD per MWh)
    PTC_wind = 26 

    # Define filepath for scenario folder and files
    folderScenario = f'Cambium22_{scenario}_2030' if year == 2030 else f'Cambium22_{scenario}'
    # Correct for ePrice_paths; update 'Mid-case' to 'MidCase' for input data
    scenario_filepath = f'Cambium22_{scenario}' if scenario != 'Mid-case' else 'Cambium22_MidCase'

    # Set filepath for folder
    ePrice_df_folder = os.path.join(inputFolder, folderScenario, 'input')

    # Define file paths for 2025
    if year == 2025:
        ePrice_paths = [
            os.path.join(ePrice_df_folder, f'{scenario_filepath}_hourly_{gea}_2024.csv'), # i = 0
            os.path.join(ePrice_df_folder, f'{scenario_filepath}_hourly_{gea}_2026.csv'), # i = 1
            os.path.join(ePrice_df_folder, f'{scenario_filepath}_hourly_{gea}_2028.csv'), # i = 2
            os.path.join(ePrice_df_folder, f'{scenario_filepath}_hourly_{gea}_2030.csv'), # i = 3
            os.path.join(ePrice_df_folder, f'{scenario_filepath}_hourly_{gea}_2035.csv'), # i = 4
            os.path.join(ePrice_df_folder, f'{scenario_filepath}_hourly_{gea}_2040.csv'), # i = 5
            os.path.join(ePrice_df_folder, f'{scenario_filepath}_hourly_{gea}_2045.csv'), # i = 6
            os.path.join(ePrice_df_folder, f'{scenario_filepath}_hourly_{gea}_2050.csv') # i = 7
        ]
    # Define file paths for 2030
    else:
        ePrice_paths = [
            os.path.join(ePrice_df_folder, f'{scenario_filepath}_hourly_{gea}_2024.csv'), # i = 0
            os.path.join(ePrice_df_folder, f'{scenario_filepath}_hourly_{gea}_2026.csv'), # i = 1
            os.path.join(ePrice_df_folder, f'{scenario_filepath}_hourly_{gea}_2028.csv'), # i = 2
            os.path.join(ePrice_df_folder, f'{scenario_filepath}_hourly_{gea}_2030.csv'), # i = 3
            os.path.join(ePrice_df_folder, f'{scenario_filepath}_hourly_{gea}_2035.csv'), # i = 4
            os.path.join(ePrice_df_folder, f'{scenario_filepath}_hourly_{gea}_2040.csv'), # i = 5
            os.path.join(ePrice_df_folder, f'{scenario_filepath}_hourly_{gea}_2045.csv'), # i = 6
            os.path.join(ePrice_df_folder, f'{scenario_filepath}_hourly_{gea}_2050.csv') # i = 7
        ]

    # Read csv and remove first 5 rows
    ePrice_dfs = [pd.read_csv(i, skiprows = 5) for i in ePrice_paths]

    # Apply production tax credit with phaseout where ptc_phaseout = True
    if ptc_phaseout:
        # enumerate() provides both index (i) and dataframe (df) to iterate over list of ePrice_dfs
        for i, df in enumerate(ePrice_dfs):
            # If year == 2025, then PTC is added to first four dfs/first 10 years
            # If year == 2030, then PTC is added to first five dfs/first 10 years
            if (year == 2025 and i < 4) or (year == 2030 and i < 5):
                df['total_cost_enduse'] += PTC_wind
    # Apply production tax credit with no phaseout where ptc_phaseout = False
    else:
        for df in ePrice_dfs:
            # Add PTC to all years
            df['total_cost_enduse'] += PTC_wind

    # Extract 
    ePrice_dfs_vec = [df['total_cost_enduse'] for df in ePrice_dfs]

    # Concatenate for 2025
    if year == 2025:
        ePrice_df = pd.concat(
            [
                ePrice_dfs_vec[0],              # ePrice_2025 
                *([ePrice_dfs_vec[1]] * 2),     # ePrice_2026_27
                *([ePrice_dfs_vec[2]] * 2),     # ePrice_2028_29
                *([ePrice_dfs_vec[3]] * 5),     # ePrice_2030_34
                *([ePrice_dfs_vec[4]] * 5),     # ePrice_2035_39
                *([ePrice_dfs_vec[5]] * 5),     # ePrice_2040_44
                *([ePrice_dfs_vec[6]] * 5)      # ePrice_2045_49
            ], axis = 1, ignore_index = True
        )
    else:  # year == 2030
        ePrice_df = pd.concat(
            [
                ePrice_dfs_vec[3],              # ePrice_2030_34
                ePrice_dfs_vec[3],              # ePrice_2030_34
                ePrice_dfs_vec[3],              # ePrice_2030_34
                ePrice_dfs_vec[3],              # ePrice_2030_34
                ePrice_dfs_vec[3],              # ePrice_2030_34
                *([ePrice_dfs_vec[4]] * 5),     # ePrice_2035_39
                *([ePrice_dfs_vec[5]] * 5),     # ePrice_2040_44
                *([ePrice_dfs_vec[6]] * 5),     # ePrice_2045_49
                *([ePrice_dfs_vec[7]] * 5)      # ePrice_2050_54
            ], axis = 1, ignore_index = True
        )
  
    # Add hour column
    ePrice_df['hour'] = range(1, 8761)
    ePrice_df.insert(0, 'hour', ePrice_df.pop('hour'))
    
    # Add years as column names
    new_column_names = ['hour'] + [str(i) for i in range(1, 26)]
    ePrice_df.columns = new_column_names

    # Reshape df to long
    ePrice_df_long = ePrice_df.melt(id_vars = ['hour'], var_name = 'year', value_name = 'total_cost_enduse')

    # Convert all columns to numeric
    ePrice_df_long = ePrice_df_long.apply(pd.to_numeric)

    # Set discount rate
    discount_rate = 0.04

    # Apply discounting
    ePrice_df_long['PV_total_cost_enduse'] =  ePrice_df_long['total_cost_enduse'] * (1 + discount_rate) ** (-ePrice_df_long['year'])

    # Reshape df to wide
    ePrice_df_long = ePrice_df_long[['year', 'hour', 'PV_total_cost_enduse']]
    ePrice_df_wide = pd.pivot(ePrice_df_long, index = 'hour', columns = 'year', values = 'PV_total_cost_enduse')

    # Save df to CSV file by GEA
    folder = os.path.join(inputFolder, folderScenario, 'Cash_Flow_PTC_No_Phaseout')
    # If PTC phaseout after 10 years
    if ptc_phaseout:
        folder = os.path.join(inputFolder, folderScenario, 'Cash_Flow')
    ePrice_df_wide.to_csv(f'{folder}/cambiumHourly_{gea}.csv')
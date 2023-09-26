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

''' ============================
For loop by GEA region
============================ '''

for i in gea_region_codes:

    gea = i

    # Define production tax credit (in USD per MWh)
    PTC_wind = 26 

    # Set filepath for folder
    ePrice_df_folder = os.path.join(inputFolder, 'Cambium22_Electrification', 'input')
    
    # Set filepath for csv
    ePrice_path_t1 = ePrice_df_folder + '/Cambium22_Electrification_hourly_' + gea + '_2024.csv'
    ePrice_path_t2 = ePrice_df_folder + '/Cambium22_Electrification_hourly_' + gea + '_2026.csv'
    ePrice_path_t3 = ePrice_df_folder + '/Cambium22_Electrification_hourly_' + gea + '_2028.csv'
    ePrice_path_t4 = ePrice_df_folder + '/Cambium22_Electrification_hourly_' + gea + '_2030.csv'
    ePrice_path_t5 = ePrice_df_folder + '/Cambium22_Electrification_hourly_' + gea + '_2035.csv'
    ePrice_path_t6 = ePrice_df_folder + '/Cambium22_Electrification_hourly_' + gea + '_2040.csv'
    ePrice_path_t7 = ePrice_df_folder + '/Cambium22_Electrification_hourly_' + gea + '_2045.csv'
    ePrice_path_t8 = ePrice_df_folder + '/Cambium22_Electrification_hourly_' + gea + '_2050.csv'

    # Read csv and remove first 5 rows
    ePrice_df_t1 = pd.read_csv(ePrice_path_t1, skiprows = 5)
    ePrice_df_t2 = pd.read_csv(ePrice_path_t2, skiprows = 5)
    ePrice_df_t3 = pd.read_csv(ePrice_path_t3, skiprows = 5)
    ePrice_df_t4 = pd.read_csv(ePrice_path_t4, skiprows = 5)
    ePrice_df_t5 = pd.read_csv(ePrice_path_t5, skiprows = 5)
    ePrice_df_t6 = pd.read_csv(ePrice_path_t6, skiprows = 5)
    ePrice_df_t7 = pd.read_csv(ePrice_path_t7, skiprows = 5)
    ePrice_df_t8 = pd.read_csv(ePrice_path_t8, skiprows = 5)
    # ePrice_df.columns.values

    # Extract 'total_cost_enduse' column
    ePrice_df_t1 = ePrice_df_t1['total_cost_enduse'] + PTC_wind 
    ePrice_df_t2 = ePrice_df_t2['total_cost_enduse'] + PTC_wind 
    ePrice_df_t3 = ePrice_df_t3['total_cost_enduse'] + PTC_wind 
    ePrice_df_t4 = ePrice_df_t4['total_cost_enduse'] + PTC_wind 
    ePrice_df_t5 = ePrice_df_t5['total_cost_enduse']
    ePrice_df_t6 = ePrice_df_t6['total_cost_enduse']
    ePrice_df_t7 = ePrice_df_t7['total_cost_enduse']
    ePrice_df_t8 = ePrice_df_t8['total_cost_enduse']

    # Concatenate 
    ePrice_2026_27_df = pd.concat([ePrice_df_t2, ePrice_df_t2], axis = 1, ignore_index = True)
    ePrice_2028_29_df = pd.concat([ePrice_df_t3, ePrice_df_t3], axis = 1, ignore_index = True)
    ePrice_2030_34_df = pd.concat([ePrice_df_t4, ePrice_df_t4, ePrice_df_t4, ePrice_df_t4, ePrice_df_t4], axis = 1, ignore_index = True)
    ePrice_2035_39_df = pd.concat([ePrice_df_t5, ePrice_df_t5, ePrice_df_t5, ePrice_df_t5, ePrice_df_t5], axis = 1, ignore_index = True)
    ePrice_2040_44_df = pd.concat([ePrice_df_t6, ePrice_df_t6, ePrice_df_t6, ePrice_df_t6, ePrice_df_t6], axis = 1, ignore_index = True)
    ePrice_2045_49_df = pd.concat([ePrice_df_t7, ePrice_df_t7, ePrice_df_t7, ePrice_df_t7, ePrice_df_t7], axis = 1, ignore_index = True)

    # Concatenate
    ePrice_df = pd.concat([ePrice_df_t1, ePrice_2026_27_df, ePrice_2028_29_df, ePrice_2030_34_df, ePrice_2035_39_df, ePrice_2040_44_df, ePrice_2045_49_df], axis = 1, ignore_index = True)
    
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
    folder = os.path.join(inputFolder, 'Cambium22_Electrification', 'Cash_Flow')
    ePrice_df_wide.to_csv(f'{folder}/cambiumHourly_{gea}.csv')

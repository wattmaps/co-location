def onshoreWind(homePath, lat_in, lon_in, data_in):

    import PySAM
    from PySAM.PySSC import PySSC
    import sys, os
    import pandas as pd
    import numpy as np

    ssc = PySSC()

    wfd = ssc.data_create()
    ssc.data_set_number(wfd, b'lat', lat_in)
    ssc.data_set_number(wfd, b'lon', lon_in)

    data = ssc.data_create()
    ssc.data_set_table(data, b'wind_resource_data', wfd)
    ssc.data_free(wfd)  
    

    ### ====> FROM THE SAM CODE GENERATOR
    ssc.module_exec_set_print(0)
    #data = ssc.data_create()
    ssc.data_set_number( data, b'wind_resource_model_choice', 0 )
    ssc.data_set_string( data, b'wind_resource_filename', data_in );
    wind_resource_distribution = [[ 3.1185,   45,   0.0344 ], [ 9.1355000000000004,   45,   0.0172 ], [ 15.1525,   45,   0.000457 ], [ 21.169499999999999,   45,   0 ], [ 3.1185,   135,   0.046800000000000001 ], [ 9.1355000000000004,   135,   0.0591 ], [ 15.1525,   135,   0.0032000000000000002 ], [ 21.169499999999999,   135,   0.00011400000000000001 ], [ 3.1185,   225,   0.063799999999999996 ], [ 9.1355000000000004,   225,   0.28199999999999997 ], [ 15.1525,   225,   0.26300000000000001 ], [ 21.169499999999999,   225,   0.0487 ], [ 3.1185,   315,   0.062399999999999997 ], [ 9.1355000000000004,   315,   0.0974 ], [ 15.1525,   315,   0.020500000000000001 ], [ 21.169499999999999,   315,   0.0010300000000000001 ]];
    ssc.data_set_matrix( data, b'wind_resource_distribution', wind_resource_distribution );
    ssc.data_set_number( data, b'weibull_reference_height', 50 )
    ssc.data_set_number( data, b'weibull_k_factor', 2 )
    ssc.data_set_number( data, b'weibull_wind_speed', 7.25 )
    ssc.data_set_number( data, b'wind_resource_shear', 0.14000000000000001 )
    ssc.data_set_number( data, b'wind_turbine_rotor_diameter', 90 )
    wind_turbine_powercurve_windspeeds =[ 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10, 10.25, 10.5, 10.75, 11, 11.25, 11.5, 11.75, 12, 12.25, 12.5, 12.75, 13, 13.25, 13.5, 13.75, 14, 14.25, 14.5, 14.75, 15, 15.25, 15.5, 15.75, 16, 16.25, 16.5, 16.75, 17, 17.25, 17.5, 17.75, 18, 18.25, 18.5, 18.75, 19, 19.25, 19.5, 19.75, 20, 20.25, 20.5, 20.75, 21, 21.25, 21.5, 21.75, 22, 22.25, 22.5, 22.75, 23, 23.25, 23.5, 23.75, 24, 24.25, 24.5, 24.75, 25, 25.25, 25.5, 25.75, 26, 26.25, 26.5, 26.75, 27, 27.25, 27.5, 27.75, 28, 28.25, 28.5, 28.75, 29, 29.25, 29.5, 29.75, 30, 30.25, 30.5, 30.75, 31, 31.25, 31.5, 31.75, 32, 32.25, 32.5, 32.75, 33, 33.25, 33.5, 33.75, 34, 34.25, 34.5, 34.75, 35, 35.25, 35.5, 35.75, 36, 36.25, 36.5, 36.75, 37, 37.25, 37.5, 37.75, 38, 38.25, 38.5, 38.75, 39, 39.25, 39.5, 39.75, 40 ];
    ssc.data_set_array( data, b'wind_turbine_powercurve_windspeeds',  wind_turbine_powercurve_windspeeds);
    wind_turbine_powercurve_powerout =[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18.559999999999999, 35.390000000000001, 52.210000000000001, 74.560000000000002, 96.879999999999995, 124.69, 157.99000000000001, 191.28999999999999, 241.12, 279.92000000000002, 335.19999999999999, 368.5, 423.77999999999997, 490.05000000000001, 556.37, 622.63999999999999, 683.40999999999997, 755.17999999999995, 832.42999999999995, 904.20000000000005, 992.49000000000001, 1080, 1140, 1220, 1310, 1410, 1480, 1570, 1640, 1700, 1750, 1770, 1790, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ];
    ssc.data_set_array( data, b'wind_turbine_powercurve_powerout',  wind_turbine_powercurve_powerout);
    ssc.data_set_number( data, b'wind_turbine_hub_ht', 100 )
    ssc.data_set_number( data, b'wind_turbine_max_cp', 0.45000000000000001 )
    ssc.data_set_number( data, b'wind_farm_wake_model', 0 )
    ssc.data_set_number( data, b'wind_resource_turbulence_coeff', 0.10000000000000001 )
    ssc.data_set_number( data, b'system_capacity', 90000 )
    wind_farm_xCoordinates =[ 0, 720, 1440, 2160, 2880, 3600, 4320, 5040, 5760, 6480, 0, 720, 1440, 2160, 2880, 3600, 4320, 5040, 5760, 6480, 0, 720, 1440, 2160, 2880, 3600, 4320, 5040, 5760, 6480, 0, 720, 1440, 2160, 2880, 3600, 4320, 5040, 5760, 6480, 0, 720, 1440, 2160, 2880, 3600, 4320, 5040, 5760, 6480 ];
    ssc.data_set_array( data, b'wind_farm_xCoordinates',  wind_farm_xCoordinates);
    wind_farm_yCoordinates =[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 720, 720, 720, 720, 720, 720, 720, 720, 720, 720, 1440, 1440, 1440, 1440, 1440, 1440, 1440, 1440, 1440, 1440, 2160, 2160, 2160, 2160, 2160, 2160, 2160, 2160, 2160, 2160, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880, 2880 ];
    ssc.data_set_array( data, b'wind_farm_yCoordinates',  wind_farm_yCoordinates);
    ssc.data_set_number( data, b'wake_int_loss', 0 )
    ssc.data_set_number( data, b'wake_ext_loss', 1.1000000000000001 )
    ssc.data_set_number( data, b'wake_future_loss', 0 )
    ssc.data_set_number( data, b'avail_bop_loss', 0.5 )
    ssc.data_set_number( data, b'avail_grid_loss', 0 )
    ssc.data_set_number( data, b'avail_turb_loss', 3.5800000000000001 )
    ssc.data_set_number( data, b'elec_eff_loss', 1.9099999999999999 )
    ssc.data_set_number( data, b'elec_parasitic_loss', 0.10000000000000001 )
    ssc.data_set_number( data, b'env_degrad_loss', 1.8 )
    ssc.data_set_number( data, b'env_exposure_loss', 0 )
    ssc.data_set_number( data, b'env_env_loss', 0.40000000000000002 )
    ssc.data_set_number( data, b'env_icing_loss', 0.20999999999999999 )
    ssc.data_set_number( data, b'ops_env_loss', 1 )
    ssc.data_set_number( data, b'ops_grid_loss', 0 )
    ssc.data_set_number( data, b'ops_load_loss', 0 )
    ssc.data_set_number( data, b'ops_strategies_loss', 0 )
    ssc.data_set_number( data, b'turb_generic_loss', 1.7 )
    ssc.data_set_number( data, b'turb_hysteresis_loss', 0.40000000000000002 )
    ssc.data_set_number( data, b'turb_perf_loss', 1.1000000000000001 )
    ssc.data_set_number( data, b'turb_specific_loss', 0.81000000000000005 )
    ssc.data_set_number( data, b'adjust:constant', 0 )
    ssc.data_set_number( data, b'total_uncert', 12.085000000000001 )
    # update the file path so it points to solar-wind-ratios/data/SAM/wind_grid_curtailment.csv on your local computer
    ssc.data_set_array_from_csv( data, b'grid_curtailment', fn = b'/Users/colleenmccamy/Documents/MEDS/Capstone/code/sam-generation/wind/grid_curtailment.csv');
    ssc.data_set_number( data, b'enable_interconnection_limit', 0 )
    ssc.data_set_number( data, b'grid_interconnection_limit_kwac', 100000 )
    module = ssc.module_create(b'windpower')	
    ssc.module_exec_set_print( 0 );

    if ssc.module_exec(module, data) == 0:
                print ('windpower simulation error')
                idx = 1
                msg = ssc.module_log(module, 0)
                while (msg != None):
                    print ('	: ' + msg.decode("utf - 8"))
                    msg = ssc.module_log(module, idx)
                    idx = idx + 1
                SystemExit( "Simulation Error" );
    ssc.module_free(module)
    module = ssc.module_create(b'grid')	
    ssc.module_exec_set_print( 0 );
    if ssc.module_exec(module, data) == 0:
            print ('grid simulation error')
            idx = 1
            msg = ssc.module_log(module, 0)
            while (msg != None):
                    print ('	: ' + msg.decode("utf - 8"))
                    msg = ssc.module_log(module, idx)
                    idx = idx + 1
            SystemExit( "Simulation Error" );
    ssc.module_free(module)
    #print ('Done')
    energy_generation_hourly = ssc.data_get_array(data, b'gen');

    capacity_factor_hourly = pd.Series(energy_generation_hourly) / 89760


    ssc.data_free(data);
    return capacity_factor_hourly
    print(f'done {FID}')
    # Calculate hourly capacity factor using turbine rating (max power curve)
    # capacity_factor_hourly = pd.Series(energy_generation_hourly) / (600000)
    ssc.data_free(data);
def singleAxisTracking(homePath, lat_in, lon_in, timezone_in, elevation_in, df_data_in):
    
	from PySAM.PySSC import PySSC
	import pandas as pd
    
	ssc = PySSC()
    
    # =====> ADDED TO READ DATA FROM NSRDB 
	wfd = ssc.data_create()
	ssc.data_set_number(wfd, b'lat', lat_in)
	ssc.data_set_number(wfd, b'lon', lon_in)
	ssc.data_set_number(wfd, b'tz', timezone_in)
	ssc.data_set_number(wfd, b'elev', elevation_in)
	ssc.data_set_array(wfd, b'year', df_data_in.index.year)
	ssc.data_set_array(wfd, b'month', df_data_in.index.month)
	ssc.data_set_array(wfd, b'day', df_data_in.index.day)
	ssc.data_set_array(wfd, b'hour', df_data_in.index.hour)
	ssc.data_set_array(wfd, b'minute', df_data_in.index.minute)
	ssc.data_set_array(wfd, b'dn', df_data_in['DNI'])
	ssc.data_set_array(wfd, b'df', df_data_in['DHI'])
	ssc.data_set_array(wfd, b'wspd', df_data_in['Wind Speed'])
	ssc.data_set_array(wfd, b'tdry', df_data_in['Temperature'])
    
	data = ssc.data_create()
	ssc.data_set_table(data, b'solar_resource_data', wfd)
	ssc.data_free(wfd)    
    
    # ====> FROM THE SAM CODE GENERATOR
	ssc.module_exec_set_print(0)
	# data = ssc.data_create()
	# ssc.data_set_string( data, b'solar_resource_file', b'C:/SAM/2022.11.21/solar_resource/phoenix_az_33.450495_-111.983688_psmv3_60_tmy.csv' );
	albedo =[ 0.20000000000000001 ];
	ssc.data_set_array( data, b'albedo',  albedo);
	ssc.data_set_number( data, b'use_wf_albedo', 1 )
	ssc.data_set_number( data, b'system_capacity', 50000 )
	ssc.data_set_number( data, b'module_type', 0 )
	ssc.data_set_number( data, b'dc_ac_ratio', 1.3400000000000001 )
	ssc.data_set_number( data, b'bifaciality', 0 )
	ssc.data_set_number( data, b'array_type', 2 )
	ssc.data_set_number( data, b'tilt', 0 )
	ssc.data_set_number( data, b'azimuth', 180 )
	ssc.data_set_number( data, b'gcr', 0.29999999999999999 )
	soiling =[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ];
	ssc.data_set_array( data, b'soiling',  soiling);
	ssc.data_set_number( data, b'losses', 12.304024826166826 )
	ssc.data_set_number( data, b'en_snowloss', 0 )
	ssc.data_set_number( data, b'inv_eff', 98 )
	ssc.data_set_number( data, b'batt_simple_enable', 0 )
	ssc.data_set_number( data, b'adjust:constant', 0 )
	# Update filepath so it points to solar-wind-ratios/data/SAM/solar_grid_curtailment.csv on your local computer
	ssc.data_set_array_from_csv( data, b'grid_curtailment', fn = b'C:\Code\sam-generation\solar\Python Scripts\grid_curtailment.csv');
	ssc.data_set_number( data, b'enable_interconnection_limit', 0 )
	ssc.data_set_number( data, b'grid_interconnection_limit_kwac', 100000 )
	module = ssc.module_create(b'pvwattsv8')	
	ssc.module_exec_set_print( 0 );
	if ssc.module_exec(module, data) == 0:
		print ('pvwattsv8 simulation error')
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
	annual_energy = ssc.data_get_number(data, b'annual_energy');
	print ('Annual AC energy in Year 1 = ', annual_energy)
	capacity_factor = ssc.data_get_number(data, b'capacity_factor');
	print ('DC capacity factor in Year 1 = ', capacity_factor)
	kwh_per_kw = ssc.data_get_number(data, b'kwh_per_kw');
	print ('Energy yield in Year 1 = ', kwh_per_kw)
	energy_generation_hourly = ssc.data_get_array(data, b'gen');
	capacity_factor_hourly = pd.Series(energy_generation_hourly) / 50000
	ssc.data_free(data);
	return capacity_factor_hourly
if __name__ == "__main__":
	singleAxisTracking()
""" ============================
Import packages and set directory
============================ """

import os, sys
import glob
import time
import pandas as pd
import numpy as np
import shutil
from pyomo.environ import *
from pyomo.opt import SolverFactory
from mpi4py import MPI

# Obtain time
start_time = time.time()

# Set filepath
current_dir = os.getcwd()
inputFolder = os.path.join(current_dir, "data")

# Read PIDS DataFrame
seq = pd.read_csv(os.path.join(inputFolder, "uswtdb", "us_PID_cords_15.csv"))

""" ============================
Set solver
============================ """
# Set solver
solver = "cplex"

if solver == "cplex":
    opt = SolverFactory("cplex", executable = "/sw/csc/CPLEX_Studio2211/cplex/bin/x86-64_linux/cplex") #"/Applications/CPLEX_Studio2211/cplex/bin/x86-64_osx/cplex")
    opt.options["mipgap"] = 0.005
    opt.options["optimalitytarget"] = 1 ## https://www.ibm.com/docs/en/icos/12.10.0?topic=parameters-optimality-target
    
""" ============================
Define functions to generate dictionary objects from dfs
============================ """

# FUNCTION 1
# Create dictionary from ix1 vector, 
# where i_indexName is name of index as a string, 
# and i_indexList is a list of those index names

def pyomoInput_dfVectorToDict(df, i_indexName, i_indexList):
    # Merge index list to df of vector data
    df_merged = pd.concat([pd.DataFrame({i_indexName: i_indexList}), df], axis = 1)
    
    # Set index
    df_merged.set_index(i_indexName, inplace = True)
    return df_merged.to_dict()

# FUNCTION 2
# Create dictionary from matrix 
def pyomoInput_matrixToDict(df, i_indexName, j_indexNames_list):

    # Melt by indicies
    df_melt = pd.melt(df, id_vars = i_indexName, value_vars = j_indexNames_list)
    
    # Melt all j_indexNames_list (columns) into single "variable" column
    # Single "variable" column is second index
    df_melt.set_index([i_indexName, "variable"], inplace = True)
    out_df = df_melt.to_dict()["value"]
    return out_df


""" ============================
Initialize df and filepath
============================ """

# Create a df with column names
output_df = pd.DataFrame(columns = ["PID", 
                                    "solar_capacity", 
                                    "wind_capacity", 
                                    "solar_wind_ratio", 
                                    "tx_capacity", 
                                    "revenue", 
                                    "cost", 
                                    "profit",
                                    "LCOE",
                                    "LVOE",
                                    "NVOE",
                                    "potentialGen_wind_lifetime",
                                    "potentialGen_solar_lifetime",
                                    "solar_wind_ratio_potentialGen", 
                                    "actualGen_lifetime",
                                    "potentialGen_lifetime",
                                    "export_lifetime",
                                    "curtailment"])

""" ============================
Retrieve system arguments
============================ """

cambium_scen = sys.argv[1] ## should be either "Cambium22Electrification" or "Cambium22Midcase"
PTC_scen = sys.argv[2] ## should be either "NoPhaseout" or "YesPhaseout"
ATBreleaseYr_scen = sys.argv[3] ## should be either 2022 or 2023
ATBcost_scen = sys.argv[4] ## should be advanced or moderate
ATBcapexYr_scen = sys.argv[5] ## should be either "2025" or "2030"
tx_scen = sys.argv[6] ## should be either "100" or "120"
scen_num = sys.argv[7]
mode = sys.argv[8] ## should be either "initial" or "backfill"
backfillNum = sys.argv[9]

""" ============================
Set filepath for combined results
============================ """

scenario_foldername_iter = "WindOnly_" + cambium_scen + "_" + PTC_scen + "_" + ATBreleaseYr_scen + "_" + ATBcost_scen + "_" + ATBcapexYr_scen + "_" + tx_scen + "_" + scen_num
scenario_filename_combined = "WindOnly_" + cambium_scen + "_" + PTC_scen + "_" + ATBreleaseYr_scen + "_" + ATBcost_scen + "_" + ATBcapexYr_scen + "_" + tx_scen + "_" + scen_num  + ".csv"
scenario_filename_combined_missingPIDs = "WindOnly_" + cambium_scen + "_" + PTC_scen + "_" + ATBreleaseYr_scen + "_" + ATBcost_scen + "_" + ATBcapexYr_scen + "_" + tx_scen + "_" + scen_num + "_"  + mode + "_missingPIDs" + ".csv"

output_df_path_iterationsFolder = os.path.join(current_dir, "results", "HPCscenarios", scenario_foldername_iter)
output_df_path = os.path.join(current_dir, "results", "HPCscenarios", scenario_filename_combined)
output_df_path_missingPIDs = os.path.join(current_dir, "results", "HPCscenarios", scenario_filename_combined_missingPIDs)

""" ============================
Get subset of PIDs that did not run
============================ """

print("Checking whether this file exists:", output_df_path) 

if mode == "backfill":
    if os.path.isfile(output_df_path):
        ## read the csv
        print("Reading compiled csv to get PIDs")
        combined_df = pd.read_csv(output_df_path)
    else: ## combine csvs into one
        print("Concatenating iterations into one csv to get PIDs")
        all_csvs_iter = glob.glob(os.path.join(output_df_path_iterationsFolder, "*.csv"))
        combined_df = pd.concat((pd.read_csv(f) for f in all_csvs_iter), axis = 0, ignore_index=True)

        
    PIDsList_finished = combined_df["PID"].tolist()
    PIDsList_needToRun = list(set(seq["PID"].tolist()) - set(PIDsList_finished))
    n_PIDs = len(PIDsList_needToRun)
    
else:
    PIDsList_needToRun = seq["PID"].tolist()
    n_PIDs = len(PIDsList_needToRun)
    
print("working on", n_PIDs, "PIDs. List of PIDs: ", PIDsList_needToRun)

""" ============================
Set up parallel processing 
============================ """
# Get MPI node information
def _get_node_info(verbose = False):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    if verbose:
        print(">> MPI: Name: {} Rank: {} Size: {}".format(name, rank, size) )
    return int(rank), int(size), comm

# MPI job variables
i_job, N_jobs, comm = _get_node_info()

# create list of PIDs to run in each node
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if N_jobs >1:
    iter_length = floor(n_PIDs/(N_jobs-1))
else:
    iter_length = n_PIDs

list_batch = list(chunks(PIDsList_needToRun, iter_length))

list_batch_iter = list_batch[i_job]

if mode == "backfill":
    ## add 10 to the job number to avoid overwriting the existing results
    i_job = (i_job + 1) * 100**(int(backfillNum))
    ## update the combined csv filename to include only the backfilled PIDs and avoid overwriting the initial csv output
    scenario_filename_combined = scenario_filename_combined[:-4] + "_backfill.csv"
    output_df_path = os.path.join(current_dir, "results", "HPCscenarios", scenario_filename_combined)

""" ============================
Set filepath for each iterative result
============================ """
# Set filepath
scenario_filename_iter = "WindOnly_" + cambium_scen + "_" + PTC_scen + "_" + ATBreleaseYr_scen + "_" + ATBcost_scen + "_" + ATBcapexYr_scen + "_" + tx_scen + "_" + scen_num + "_" + str(i_job)  + ".csv"

# Set file path for model results csv
output_df_path_iterations = os.path.join(current_dir, "results", "HPCscenarios", scenario_foldername_iter, scenario_filename_iter)

## Reassign variables to match 
if cambium_scen == "Cambium22Electrification":
    cambium_scen = "Cambium22_Electrification"
if cambium_scen == "Cambium22Midcase":
    cambium_scen = "Cambium22_Mid-case"

if PTC_scen == "NoPhaseout":
    PTC_scen = "Cash_Flow_PTC_No_Phaseout"
if PTC_scen == "YesPhaseout":
    PTC_scen = "Cash_Flow"
    
if tx_scen == "100":
    tx_scen = 1.0
if tx_scen == "120":
    tx_scen = 1.2
    
if ATBcapexYr_scen == "2030":
    cambium_scen_yr_append = "_2030"
if ATBcapexYr_scen == "2025":
    cambium_scen_yr_append = ""

""" ============================
Read data
============================ """

# Set file path and read csv for substation attributes
pid_substation_file = os.path.join(inputFolder, "PID_Attributes", "substation.csv")
pid_substation_df = pd.read_csv(pid_substation_file)

# Set file path and read csv for potential capacity for solar
cap_s_path = os.path.join(inputFolder, "Potential_Installed_Capacity", "solar_land_capacity_WindOnly.csv")
cap_s_df = pd.read_csv(cap_s_path)

# Set file path and read csv for potential capacity for wind
cap_w_path = os.path.join(inputFolder, "Potential_Installed_Capacity", "wind_land_capacity.csv")
cap_w_df = pd.read_csv(cap_w_path)

# Set file path and read csv for associated GEA
pid_gea_file = os.path.join(inputFolder, "PID_Attributes", "GEAs.csv")
pid_gea_df = pd.read_csv(pid_gea_file)

""" ============================
Define optimization function given a single value
============================ """

def runOptimization(PID, output_df_arg):

    print("PID: ", PID)

    """ ============================
    Set scalar parameters
    ============================ """

    # CAPITAL COSTS
    
    if ATBreleaseYr_scen == "2022" and ATBcapexYr_scen == "2025":
        # Define capital expenditure for wind and solar (in USD/MW) 
        # 2022 ATB advanced scenario
        capEx_w = 1081*1000 # class 5, 2025
        capEx_s = 922*1000 # class 5, 2025
        # OPERATION & MAINTENANCE COSTS
        om_w = 39*1000 # class 5, 2025
        om_s = 17*1000 # class 5, 2025
            
    if ATBreleaseYr_scen == "2022" and ATBcapexYr_scen == "2030":
        # Define capital expenditure for wind and solar (in USD/MW) 
        # 2022 ATB advanced scenario
        capEx_w = 704*1000 # class 5, 2030
        capEx_s = 620*1000 # class 5, 2030
        # OPERATION & MAINTENANCE COSTS
        om_w = 34*1000 # class 5, 2030
        om_s = 13*1000 # class 5, 2030
        
    
    if ATBreleaseYr_scen == "2023" and ATBcapexYr_scen == "2025":
        # 2023 ATB advanced scenario
        capEx_w = 1244*1000 # class 5, 2025
        capEx_s = 1202*1000 # class 5, 2025
        om_w = 27*1000 # class 5, 2025
        om_s = 20*1000 # class 5, 2025
    
    if ATBreleaseYr_scen == "2023" and ATBcapexYr_scen == "2030":
        # 2023 ATB advanced scenario
        capEx_w = 1096*1000 # class 5, 2030
        capEx_s = 917*1000 # class 5, 2030
        om_w = 24*1000 # class 5, 2030
        om_s = 16*1000 # class 5, 2030

    # CAPITAL RECOVERY FACTOR
    # Define discount rate for capital recovery factor
    d = 0.04 
    # Define number of years for capital recovery factor
    n = 25
    n_bat = 12.5 # life of battery
    # Define capital recovery factor (scalar)
    CRF = (d*(1+d)**n)/((1+d)**n - 1)
    # CRFbat = (d*(1+d)**n_bat)/((1+d)**n_bat - 1)
    # Define denominator of battery
    # denom_batt = (1 + d)**n_bat

    # TRANSMISSION AND SUBSTATION COSTS
    # Define USD2018 per km per MW for total transmission costs per MW
    # USD2018 per km for $500 MW of capacity; divide by 500 to obtain per MW value
    i = 572843/500
    # Define per MW for total transmission costs per MW
    # per 200 MW; divide by 200 to obtain per MW value
    sub = 7609776/200
    # Define kilometers for total transmission costs per MW
    # Corresponding distance to closest substation 115kV or higher by PID
    km = pid_substation_df.loc[pid_substation_df["PID"] == PID, "distance_km"].values[0]
    # Define total transmission costs per MW (USD per MW cost of tx + substation)
    totalTx_perMW = i*km+sub

    # SOLAR AND WIND POTENTIAL INSTALLED CAPACITY
    # Define potential installed capacity
    cap_s = cap_s_df.loc[cap_s_df["PID"] == PID, "solar_installed_cap_mw"].iloc[0]
    cap_w = cap_w_df.loc[cap_w_df["PID"] == PID, "p_cap_mw"].iloc[0]

    # TRANSMISSION CAPACITY
    # Define associated transmission substations capacity in MW
    ## size to wind capacity * certain percentage
    tx_MW = cap_w * tx_scen

    """ ============================
    Set vector parameters
    ============================ """

    ## WHOLESALE ELECTRICITY PRICES
    # Determine GEA associated with PID
    gea = pid_gea_df.loc[pid_gea_df["PID"] == PID, "gea"].values[0]
    
    # Set filepath where wholesale electricity prices are for each GEA
    ePrice_df_folder = os.path.join(inputFolder, cambium_scen + cambium_scen_yr_append, PTC_scen)
    ePrice_path = os.path.join(ePrice_df_folder, f"cambiumHourly_{gea}.csv")
    ePrice_df_wind = pd.read_csv(ePrice_path)

    # Read csv for discount by year
    discount_path = os.path.join(inputFolder, "discount_time_series.csv")
    discount_df = pd.read_csv(discount_path)

    ## SOLAR AND WIND CAPACITY FACTORS
    cf_s_path = os.path.join(inputFolder, "SAM", "Solar_Capacity_Factors_WindOnly", "capacity_factor_WindOnly.csv")
    cf_s_df = pd.read_csv(cf_s_path)
    
    cf_w_path = os.path.join(inputFolder, "SAM", "Wind_Capacity_Factors", f"capacity_factor_PID{PID}.csv")
    cf_w_df = pd.read_csv(cf_w_path)
    
    """ ============================
    Initialize model
    ============================ """

    model = AbstractModel()

    """ ============================
    Set hour and year indices
    ============================ """

    # Extract hour index
    hour = cf_s_df.loc[:,"hour"].tolist()
    # Generate an hour index with 0 
    hour0 = hour.copy()
    hour0.insert(0,0)
    hour0[:10]
    hour[:10]

    # Add hour list to model
    model.t = Set(initialize = hour)
    model.t0 = Set(initialize = hour0)

    # Set year index
    year = list(range(1, 26))
    year_char = [str(n) for n in year]
    
    # Add year list to model
    model.y = Set(initialize = year_char)
    
    # Add coupled index to model
    model.HOURYEAR = model.t * model.y
    model.HOUR0YEAR = model.t0 * model.y

    """ ============================
    Set vector parameters as dictionaries
    ============================ """

    ## ELECTRICITY PRICES ---
    # Electricity prices for wind and solar
    # ePrice_wind_hourly = next(iter(pyomoInput_dfVectorToDict(ePrice_df_wind, "hour", hour).values()))
    # ePrice_solar_hourly = next(iter(pyomoInput_dfVectorToDict(ePrice_df_wind, "hour", hour).values()))

    ## Adapted from wind_zones_v2 script
    ePrice_wind_hourly = pyomoInput_matrixToDict(ePrice_df_wind, "hour", year_char)
    #ePrice_solar_hourly = next(iter(pyomoInput_matrixToDict(ePrice_df_wind, "hour", year)))

    # Set parameter
    # model.eprice_wind = Param(model.t, default = ePrice_wind_hourly) # price of wind at each hour
    # model.eprice_solar = Param(model.t, default = ePrice_solar_hourly) # price of solar at each hour

    ## Adapted from wind_zones_v2 script
    model.eprice_wind = Param(model.HOURYEAR, default = ePrice_wind_hourly) # price of wind at each hour
    #model.eprice_solar = Param(model.HOURYEAR, default = ePrice_solar_hourly) # price of solar at each hour
    
    # Discount
    discount_df_hourly = pyomoInput_matrixToDict(discount_df, "hour", year_char)
    model.discount = Param(model.HOURYEAR, default = discount_df_hourly)

    ## Adapted from wind_zones_v2 script
    wind_cf_hourly = pyomoInput_matrixToDict(cf_w_df, "hour", year_char)
    solar_cf_hourly = pyomoInput_matrixToDict(cf_s_df, "hour", year_char)
    
    model.cf_wind = Param(model.HOURYEAR, default = wind_cf_hourly)
    model.cf_solar = Param(model.HOURYEAR, default = solar_cf_hourly)

    """ ============================
    Set scalar parameters
    ============================ """

    model.capEx_w = Param(default = capEx_w)
    model.capEx_s = Param(default = capEx_s)
    model.om_w = Param(default = om_w)
    model.om_s = Param(default = om_s)
    model.CRF = Param(default = CRF)
    # model.CRFbat = Param(default = CRFbat)
    model.capEx_tx = Param(default = totalTx_perMW)
    model.pot_w = Param(default = cap_w)
    model.pot_s = Param(default = cap_s)
    model.tx_capacity = Param(default = tx_MW)

    """ ============================
    Set decision, slack, and battery variables
    ============================ """

    # DECISION VARIABLES ---
    model.solar_capacity = Var(within = NonNegativeReals)
    # model.tx_capacity = Var(within = NonNegativeReals)

    # SLACK VARIABLES ---
    # Slack variable potential generation (without considering curtailment)
    model.potentialGen = Var(model.HOURYEAR)
    # Slack variable actual generation (considering curtailment)
    model.actualGen = Var(model.HOURYEAR)  
    # Slack variable for lifetime revenue
    model.revenue = Var()
    # Slack variable for lifetime costs
    model.cost = Var()
    # Slack variable for export lifetime discounted
    model.export_lifetime_discounted_var = Var()
    
    # Export of electricity to grid in time t
    model.Export_t = Var(model.HOURYEAR, within = NonNegativeReals)

    """ ============================
    Define objective function
    ============================ """
    
    # Maximize profit
    def obj_rule(model):
        return (model.revenue - model.cost)/model.export_lifetime_discounted_var
    model.obj = Objective(rule = obj_rule, sense = maximize)

    """ ============================
    Define constraints
    ============================ """

    ## Constraint (1) ---
    ## Define potential generation = CF*capacity for wind and solar
    # combines both forms of generation
    def potentialGeneration_rule(model, t, y):
        return model.potentialGen[t, y] == model.cf_solar[t, y]*model.solar_capacity + model.cf_wind[t, y] * model.pot_w
    model.potentialGeneration = Constraint(model.HOURYEAR, rule=potentialGeneration_rule)

    ## Constraint (2) ---
    ## Define actual generation is less than or equal to potential generation
    # how much is actually being delivered to the grid, less than the tx constraint
    def actualGenLTEpotentialGen_rule(model, t, y):
        return model.actualGen[t,y] <= model.potentialGen[t,y]
    model.actualGenLTEpotentialGen = Constraint(model.HOURYEAR, rule = actualGenLTEpotentialGen_rule)

    ## Constraint (3) --- ## UPDATE FOR BATTERY
    ## Define actual generation must be less than or equal to transmission capacity
    def actualGenLTEtxCapacity_rule(model, t, y):
        #return model.actualGen[t, y] <= model.tx_capacity
        return model.Export_t[t, y] <= model.tx_capacity
    model.actualGenLTEtxCapacity = Constraint(model.HOURYEAR, rule = actualGenLTEtxCapacity_rule)

    ## Constraint (4) --- ## UPDATE FOR BATTERY - RD VERSION ACCOUNTING FOR BATTERY LIFE THAT IS LESS THAN SOLAR AND WIND
    ## Define lifetime costs (equation #2) in net present value = overnight capital costs + NPV of fixed O&M (using annualPayments = CRF*NPV)
    def lifetimeCosts_rule(model):
        return model.cost == (model.solar_capacity*model.capEx_s) + ((model.solar_capacity*model.om_s)/model.CRF) + \
            (model.pot_w*model.capEx_w) + ((model.pot_w*model.om_w)/model.CRF) + \
                (model.tx_capacity*model.capEx_tx) # + \
                # (model.P_batt_max * model.batt_power_cost + model.E_batt_max * model.batt_energy_cost) +\
                #    (model.P_batt_max * model.batt_om / model.CRFbat) +\
                #  ((model.P_batt_max * model.batt_power_cost_future + model.E_batt_max * model.batt_energy_cost_future) +\
                #      (model.P_batt_max * model.batt_om_future / model.CRFbat))/ model.denom_batt # denominator needs to be reviewed
    model.lifetimeCosts = Constraint(rule = lifetimeCosts_rule)
    
    ## Constraint (5) ---
    ## Ensure that capacity is less than or equal to potential for solar (equation #5)
    def max_capacity_solar_rule(model):
        return model.solar_capacity <= model.pot_s
    model.maxCapacity_solar = Constraint(rule=max_capacity_solar_rule)

    ## Constraint (6) --- ## UPDATE FOR BATTERY
    ## Define lifetime revenue
    def lifetimeRevenue_rule(model):
        return model.revenue == sum(sum(model.Export_t[t, y] * model.eprice_wind[t, y] for t in model.t) for y in model.y)
    model.lifetimeRevenue = Constraint(rule = lifetimeRevenue_rule)

    # Define the constraint for lifetime discounted export
    def Export_t_discounted_rule(model):
        return model.export_lifetime_discounted_var == sum(sum(model.Export_t[t, y] / model.discount[t, y] for t in model.t) for y in model.y)
    model.export_lifetime_discounted = Constraint(rule = Export_t_discounted_rule) 

    # Constraint (7) ---
    # Check that transmission capacity is less than wind capacity 
    # will always size tx capacity to wind capacity, never undersizes so could change tx_capacity == wind_capacity
    # def tx_wind_rule(model):
    #     return model.tx_capacity <= model.wind_capacity #+ model.solar_capacity
    # model.tx_wind = Constraint(rule=tx_wind_rule)

    """ ============================
    Execute optimization
    ============================ """

    model_instance = model.create_instance()
    results = opt.solve(model_instance, tee = False)

    """ ============================
    Post-processing of opitmization solution
    ============================ """

    # FUNCTION 3
    # Parse time series variables from optimization
    def parseTimeSeries(df, val_colname):
        ## move index into columns
        df = df.reset_index()
        df["index"] = df["index"].astype("str")
        df_hour_yr = df["index"].str.split(",", expand = True)
        hour = df_hour_yr[0].str.extract(r"\((\d{1,})") 
        year = df_hour_yr[1].str.extract("(\d+)")
        df_parsed = hour.merge(year, left_index = True, right_index = True).merge(df[0], left_index = True, right_index = True)
        df_parsed.rename(columns = {"0_x": "hour", 
                                    "0_y": "year", 
                                    0: val_colname}, inplace = True)
        return df_parsed

    # Extract scalar variable values from optimization
    solar_capacity = model_instance.solar_capacity.value
    wind_capacity = model_instance.pot_w.value
    tx_capacity = model_instance.tx_capacity.value
    revenue = model_instance.revenue.value
    cost = model_instance.cost.value
    profit = model_instance.obj()
    solar_wind_ratio = solar_capacity/wind_capacity

    # Calculate actual generation
    actualGen = model_instance.actualGen.extract_values()
    actualGen_df = pd.DataFrame.from_dict(actualGen, orient = "index")
    actualGen_df_parsed = parseTimeSeries(actualGen_df, "actualGen")
    actualGen_df_annualSum = actualGen_df_parsed[["actualGen", "year"]].groupby("year").sum()
    actualGen_lifetime = actualGen_df_annualSum["actualGen"].sum()

    # Calculate potential generation
    potentialGen = model_instance.potentialGen.extract_values()
    potentialGen_df = pd.DataFrame.from_dict(potentialGen, orient = "index")
    potentialGen_df_parsed = parseTimeSeries(potentialGen_df, "potentialGen")
    potentialGen_df_annualSum = potentialGen_df_parsed[["potentialGen", "year"]].groupby("year").sum()
    potentialGen_lifetime = potentialGen_df_annualSum["potentialGen"].sum()

    # Calculate curtailment
    curtailment = (potentialGen_lifetime - actualGen_lifetime)/potentialGen_lifetime

    # Calculate export generation
    export = model_instance.Export_t.extract_values()
    export_df = pd.DataFrame.from_dict(export, orient = "index")
    export_df_parsed = parseTimeSeries(export_df, "export")
    # Calculate annual revenue
    export_df_annualSum = export_df_parsed[["export", "year"]].groupby("year").sum()
    # Discount annual revenue and calculate Net Present Value (NPV)
    export_df_annualSum = export_df_annualSum.reset_index()
    export_df_annualSum["year"] = export_df_annualSum["year"].astype("int")
    export_df_annualSum.sort_values(by = ["year"], inplace = True)
    export_df_annualSum["export_discounted"] = export_df_annualSum["export"] / (1+d)**export_df_annualSum["year"]
    export_lifetime_discounted = export_df_annualSum["export_discounted"].sum()
    export_lifetime = export_df_annualSum["export"].sum()
    
    # Calculate NPV of revenue (export generation * discounted price)
    eprice_wind = model_instance.eprice_wind.extract_values()
    eprice_wind_df = pd.DataFrame.from_dict(eprice_wind, orient = "index")
    revenue_df = export_df * eprice_wind_df
    revenue_df_parsed = parseTimeSeries(revenue_df, "revenue")
    revenue_df_annualSum = revenue_df_parsed[["revenue", "year"]].groupby("year").sum()
    
    # Calculate annual LVOE
    revenue_df_annualSum = revenue_df_annualSum.reset_index()
    revenue_df_annualSum["year"] = revenue_df_annualSum["year"].astype("int")
    LVOE_annual = revenue_df_annualSum.merge(export_df_annualSum, how = "left", on = "year")
    LVOE_annual["LVOE"] = LVOE_annual["revenue"]/LVOE_annual["export_discounted"]
    LVOE_annual["LVOE"].mean()
    
    # Calculate net value of electricity (NVOE)
    NVOE = (revenue - cost) / export_lifetime_discounted
    
    # Calculate levelized cost of energy (LCOE)
    LCOE = cost / export_lifetime_discounted
    
    # Calculate LVOE
    LVOE = revenue/export_lifetime_discounted
    
    # === CALCULATE POTENTIAL GENERATION FOR WIND
    ## potential generation  = capacity_wind * CF_hour
    potentialGen_wind = pd.melt(cf_w_df, value_vars = [str(item) for item in range(1,26)], id_vars = "hour", var_name = "year")
    potentialGen_wind["potentialGen"] = wind_capacity * potentialGen_wind["value"]
    potentialGen_wind_lifetime = potentialGen_wind["potentialGen"].sum()
    
    # === CALCULATE POTENTIAL GENERATION FOR SOLAR
    potentialGen_solar = pd.melt(cf_s_df, value_vars = [str(item) for item in range(1,26)], id_vars = "hour", var_name = "year")
    potentialGen_solar["potentialGen"] = solar_capacity * potentialGen_solar["value"]
    potentialGen_solar_lifetime = potentialGen_solar["potentialGen"].sum()
    
    # === CALCULATE POTENTIAL GENERATION RATIO FOR SOLAR:WIND
    solar_wind_ratio_potentialGen = potentialGen_solar_lifetime/potentialGen_wind_lifetime

    # append rows to output_df
    output_df = output_df_arg.append({"PID" : int(PID), 
                    "solar_capacity" : solar_capacity, 
                    "wind_capacity" : wind_capacity, 
                    "solar_wind_ratio" : solar_wind_ratio, 
                    "tx_capacity" : tx_capacity, 
                    "revenue": revenue, 
                    "cost": cost, 
                    "profit": profit,
                    "LCOE": LCOE,
                    "LVOE": LVOE,
                    "NVOE": NVOE,
                    "potentialGen_wind_lifetime": potentialGen_wind_lifetime,
                    "potentialGen_solar_lifetime": potentialGen_solar_lifetime,
                    "solar_wind_ratio_potentialGen": solar_wind_ratio_potentialGen, 
                    "actualGen_lifetime": actualGen_lifetime,
                    "potentialGen_lifetime": potentialGen_lifetime,
                    "curtailment": curtailment}, 
                    ignore_index = True)
    return output_df

""" ============================
Define optimization function given a list 
============================ """
# FUNCTION 4
def runOptimizationLoop(PID_list, output_df_arg):
    i = 0 
    for PID in PID_list:
        while True:
            try: 
                i = i + 1
                output_df_arg = runOptimization(PID, output_df_arg)
                output_df_arg.to_csv(output_df_path_iterations, index = False)
            except Exception as exc:
                print(exc)
                return output_df_arg
                output_df_arg.to_csv(output_df_path_iterations, index = False)
            break  
    return output_df_arg

""" ============================
Execute optimization loop
============================ """
# Obtain start time
start_time = time.time()

output_df_complete = runOptimizationLoop(list_batch_iter, output_df)  

print("** Completed iteration ", str(i_job), "with filename ", output_df_path_iterations)
print("**** Completed loop for PID starting with", str(list_batch_iter[0]), "and ending with", str(list_batch_iter[-1]))

# In HPC, wait for all jobs to be completed
comm.Barrier()
if i_job == 0:
    # Combine CSVs into a CSV
    print("Concatenating iterations into one csv and save to drive")
    all_csvs_iter = glob.glob(os.path.join(output_df_path_iterationsFolder, "*.csv"))
    # Check if the list is not empty before attempting to concatenate
    if all_csvs_iter:
        # Concatenate all CSV files into a single DataFrame
        combined_df = pd.concat((pd.read_csv(f) for f in all_csvs_iter), axis=0, ignore_index=True)
        # Save concatenated DataFrame to a CSV file
        combined_df.to_csv(output_df_path, index=False)
    else:
        print("No CSV files found for concatenation. Please check the output directory:", output_df_path_iterationsFolder)
    
    # Store list of PIDs that are still missing/yet to be run
    missingPIDs_list = list(set(seq["PID"].tolist()) - set(combined_df["PID"].tolist()))
    print("Missing PIDs:", missingPIDs_list)
    missingPIDs_df = pd.DataFrame(missingPIDs_list, columns=["missingPIDs"])
    missingPIDs_df.to_csv(output_df_path_missingPIDs)

else:
    comm.send(output_df_complete, dest = 0, tag = 11)
    
# Obtain end time
end_time = time.time()

# Print time of optimization
print("Time taken:", end_time - start_time, "seconds")

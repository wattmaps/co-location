""" ============================
Set up environment 
============================ """

# install mpi4py available on conda
# check cnsi website and check for mpi workshops

import os, sys
import glob
import time
import pandas as pd
import numpy as np
import shutil
from pyomo.environ import *
from pyomo.opt import SolverFactory
from mpi4py import MPI

start_time = time.time()

current_dir = os.getcwd()
print(current_dir)

inputFolder = os.path.join(current_dir, "data")

seq = pd.read_csv(os.path.join(inputFolder, "uswtdb", "us_PID_cords_15.csv"))
# seq = pd.DataFrame({'PID': [1]})

solver = "cplex"

if solver == "cplex":
    opt = SolverFactory("cplex", executable = "/sw/csc/CPLEX_Studio2211/cplex/bin/x86-64_linux/cplex") # "/Applications/CPLEX_Studio2211/cplex/bin/x86-64_osx/cplex")
    opt.options["mipgap"] = 0.005
    opt.options["optimalitytarget"] = 1 # https://www.ibm.com/docs/en/icos/12.10.0?topic=parameters-optimality-target

if solver == "gurobi":
    opt = SolverFactory("gurobi")
    
if solver == "cbc":
    opt = SolverFactory("cbc", solver_io = "python")

# Define pyomoInput_dfVectorToDict() function
# Create dictionary from ix1 vector, 
# where i_indexName is name of index as a string, 
# and i_indexList is a list of those index names
def pyomoInput_dfVectorToDict(df, i_indexName, i_indexList):
    # Merge index list to DataFrame of vector data
    df_merged = pd.concat([pd.DataFrame({i_indexName: i_indexList}), df], axis = 1)
    
    # Set index
    df_merged.set_index(i_indexName, inplace = True)
    return df_merged.to_dict()

# Define pyomoInput_matrixToDict() function
# Create dictionary from matrix 
def pyomoInput_matrixToDict(df, i_indexName, j_indexNames_list):
    # Melt by indicies
    df_melt = pd.melt(df, id_vars = i_indexName, value_vars = j_indexNames_list)
    
    # Melt all j_indexNames_list (columns) into single "variable" column
    # Single "variable" column is second index
    df_melt.set_index([i_indexName, "variable"], inplace = True)
    out_df = df_melt.to_dict()["value"]
    return out_df

# Initialize a DataFrame with column names
output_df = pd.DataFrame(columns = ["PID",\
                                    "solar_capacity", \
                                    "wind_capacity", \
                                    "solar_wind_ratio", \
                                    "tx_capacity", \
                                    "batteryCap", \
                                    "batteryEnergy", \
                                    "revenue", \
                                    "cost", \
                                    "profit",\
                                    "LCOE",\
                                    "LVOE",\
                                    "NVOE",\
                                    "potentialGen_wind_lifetime",\
                                    "potentialGen_solar_lifetime",\
                                    "solar_wind_ratio_potentialGen", \
                                    "actualGen_lifetime",\
                                    "potentialGen_lifetime",\
                                    "export_lifetime",\
                                    "curtailment"])


""" ============================
Set system arguments
============================ """

cambium_scen = sys.argv[1] # should be either "Cambium22Electrification" or "Cambium22Midcase"
PTC_scen = sys.argv[2] # should be either "NoPhaseout" or "YesPhaseout"
ATBreleaseYr_scen = sys.argv[3] # should be either 2022 or 2023
ATBcost_scen = sys.argv[4] # should be advanced or moderate
ATBcapexYr_scen = sys.argv[5] # should be either "2025" or "2030"
tx_scen = sys.argv[6] # should be either "100" or "120"
scen_num = sys.argv[7]
mode = sys.argv[8] # should be either "initial" or "backfill"
backfillNum = sys.argv[9]


""" ============================
Set file path for combined results
============================ """

scenario_foldername_iter = cambium_scen + "_" + PTC_scen + "_" + ATBreleaseYr_scen + "_" + ATBcost_scen + "_" + ATBcapexYr_scen + "_" + tx_scen + "_" + scen_num
scenario_filename_combined = cambium_scen + "_" + PTC_scen + "_" + ATBreleaseYr_scen + "_" + ATBcost_scen + "_" + ATBcapexYr_scen + "_" + tx_scen + "_" + scen_num  + ".csv"
scenario_filename_combined_missingPIDs = cambium_scen + "_" + PTC_scen + "_" + ATBreleaseYr_scen + "_" + ATBcost_scen + "_" + ATBcapexYr_scen + "_" + tx_scen + "_" + scen_num + "_"  + mode + "_missingPIDs" + ".csv"

output_df_path_iterationsFolder = os.path.join(current_dir, "results", "HPCscenarios", scenario_foldername_iter)
output_df_path = os.path.join(current_dir, "results", "HPCscenarios", scenario_filename_combined)
output_df_path_missingPIDs = os.path.join(current_dir, "results", "HPCscenarios", scenario_filename_combined_missingPIDs)

""" ============================
Get subset of PIDs that did not run
============================ """

print("Checking whether this file exists:", output_df_path) 

if mode == "backfill":
    if os.path.isfile(output_df_path):
        # read the csv
        print("Reading compiled csv to get PIDs")
        combined_df = pd.read_csv(output_df_path)
    else: # combine csvs into one
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

# Define _get_node_info() function
def get_node_info(verbose = False):
    # Initialize MPI communication
    comm = MPI.COMM_WORLD
    # Extract total number of processes
    size = comm.Get_size()
    # Extract rank of current process
    rank = comm.Get_rank()
    # Extract name of processor
    name = MPI.Get_processor_name()

    # Print MPI job variables
    if verbose:
        print(">> MPI: Name: {} Rank: {} Size: {}".format(name, rank, size))

    # Store rank, total number of processes, and MPI communicator object
    return int(rank), int(size), comm

# Gather MPI job variables
i_job, N_jobs, comm = get_node_info()

# Define chunks() function
def chunks(lst, n):
    # Split a list into chunks of size n
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Determine size of each chunk based on number of jobs (processes)
if N_jobs > 1:
    # If there are multiple jobs, divide total number of PIDs by the number of jobs
    iter_length = floor(n_PIDs / (N_jobs - 1))
else:
    # If there is only one job, the entire list will be one chunk
    iter_length = n_PIDs

# Split list of PIDs into chunks
list_batch = list(chunks(PIDsList_needToRun, iter_length))

# Select chunk corresponding to current job
list_batch_iter = list_batch[i_job]

if mode == "backfill":
    # Add 10 to the job number (avoids overwriting existing results)
    i_job = (i_job + 1) * 100**(int(backfillNum))
    # Update file name
    scenario_filename_combined = scenario_filename_combined[:-4] + "_backfill.csv"
    output_df_path = os.path.join(current_dir, "results", "HPCscenarios", scenario_filename_combined)

""" ============================
Set filepath for each iterative result
============================ """

scenario_filename_iter = cambium_scen + "_" + PTC_scen + "_" + ATBreleaseYr_scen + "_" + ATBcost_scen + "_" + ATBcapexYr_scen + "_" + tx_scen + "_" + scen_num + "_" + str(i_job)  + ".csv"

# Set file path for model results csv
output_df_path_iterations = os.path.join(current_dir, "results", "HPCscenarios", scenario_foldername_iter, scenario_filename_iter)

# Re-assign Cambium scenario variable to match 
if cambium_scen == "Cambium22Electrification":
    cambium_scen = "Cambium22_Electrification"
if cambium_scen == "Cambium22Midcase":
    cambium_scen = "Cambium22_Mid-case"

# Re-assign PTC scenario variable to match
if PTC_scen == "NoPhaseout":
    PTC_scen = "Cash_Flow_PTC_No_Phaseout"
if PTC_scen == "YesPhaseout":
    PTC_scen = "Cash_Flow"

# Re-assign TX scenario variable to match
if tx_scen == "100":
    tx_scen = 1.0
if tx_scen == "120":
    tx_scen = 1.2

# Re-assign ATB sceanrio variable to match
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
cap_s_path = os.path.join(inputFolder, "Potential_Installed_Capacity", "solar_land_capacity.csv")
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

    #### Capital Costs ####

    if ATBreleaseYr_scen == "2022" and ATBcapexYr_scen == "2025":
        # Define capital expenditure for wind and solar (in USD/MW) 
        # 2022 ATB advanced scenario
        capEx_w = 1081*1000 # class 5, 2025
        capEx_s = 922*1000 # class 5, 2025
        # OPERATION & MAINTENANCE COSTS
        om_w = 39*1000 # class 5, 2025
        om_s = 17*1000 # class 5, 2025
        
        ## Battery costs
        # BATTERY CAPITAL COSTS (2022 ATB)
        # Define capital cost for battery (in USD/MW), for advanced in 2025
        battPowerCost =  162*1000
        # Define energy cost (in USD/MWh), for advanced in 2025
        battEnergyCost = 211*1000
        # Define operations & maintenance cost for battery (in USD/MW-year), for 6 hr advanced in 2025
        battOMcost = 36*1000
        # Define capital future cost for battery (in USD/MW), for advanced in 2037
        battPowerCostFuture =  100*1000
        # Define energy future cost for battery (in USD/MWh), for advanced in 2037
        battEnergyCostFuture = 130*1000
        # Define operations & maintenance future cost for battery (in USD/MW-year), for 6 hr advanced in 2037
        battOMcostFuture = 22*1000
        # Define battery efficiency
        rtEfficiency_sqrt = sqrt(0.85)
            
    if ATBreleaseYr_scen == "2022" and ATBcapexYr_scen == "2030":
        # Define capital expenditure for wind and solar (in USD/MW) 
        # 2022 ATB advanced scenario
        capEx_w = 704*1000 # class 5, 2030
        capEx_s = 620*1000 # class 5, 2030
        # OPERATION & MAINTENANCE COSTS
        om_w = 34*1000 # class 5, 2030
        om_s = 13*1000 # class 5, 2030
        
        ## Battery costs
        # BATTERY CAPITAL COSTS (2022 ATB)
        # Define capital cost for battery (in USD/MW), for advanced in 2030
        battPowerCost =  110*1000
        # Define energy cost (in USD/MWh), for advanced in 2025
        battEnergyCost = 143*1000
        # Define operations & maintenance cost for battery (in USD/MW-year), for 6 hr advanced in 2030
        battOMcost = 24*1000
        # Define capital future cost for battery (in USD/MW), for advanced in 2042
        battPowerCostFuture =  93*1000
        # Define energy future cost for battery (in USD/MWh), for advanced in 2042
        battEnergyCostFuture = 121*1000
        # Define operations & maintenance future cost for battery (in USD/MW-year), for 6 hr advanced in 2042
        battOMcostFuture = 21*1000
        # Define battery efficiency
        rtEfficiency_sqrt = sqrt(0.85)
        
    if ATBreleaseYr_scen == "2023" and ATBcapexYr_scen == "2025":
        # 2023 ATB advanced scenario
        capEx_w = 1244*1000 # class 5, 2025
        capEx_s = 1202*1000 # class 5, 2025
        om_w = 27*1000 # class 5, 2025
        om_s = 20*1000 # class 5, 2025
        
        ## Battery costs
        # BATTERY CAPITAL COSTS (2023 ATB)
        # Define capital cost for battery (in USD/MW), for advanced in 2025
        battPowerCost =  216*1000
        # Define energy cost (in USD/MWh), for advanced in 2025
        battEnergyCost = 233*1000
        # Define operations & maintenance cost for battery (in USD/MW-year), for 6 hr advanced in 2025
        battOMcost = 40*1000
        # Define capital future cost for battery (in USD/MW), for advanced in 2037
        battPowerCostFuture =  150*1000
        # Define energy future cost for battery (in USD/MWh), for advanced in 2037
        battEnergyCostFuture = 161*1000
        # Define operations & maintenance future cost for battery (in USD/MW-year), for 6 hr advanced in 2037
        battOMcostFuture = 28*1000
        # Define battery efficiency
        rtEfficiency_sqrt = sqrt(0.85)
    
    if ATBreleaseYr_scen == "2023" and ATBcapexYr_scen == "2030":
        # 2023 ATB advanced scenario
        capEx_w = 1096*1000 # class 5, 2030
        capEx_s = 917*1000 # class 5, 2030
        om_w = 24*1000 # class 5, 2030
        om_s = 16*1000 # class 5, 2030
        
        ## Battery costs
        # BATTERY CAPITAL COSTS (2023 ATB)
        # Define capital cost for battery (in USD/MW), for advanced in 2030
        battPowerCost =  171*1000
        # Define energy cost (in USD/MWh), for advanced in 2030
        battEnergyCost = 184*1000
        # Define operations & maintenance cost for battery (in USD/MW-year), for 6 hr advanced in 2030
        battOMcost = 32*1000
        # Define capital future cost for battery (in USD/MW), for advanced in 2042
        battPowerCostFuture =  134*1000
        # Define energy future cost for battery (in USD/MWh), for advanced in 2042
        battEnergyCostFuture = 145*1000
        # Define operations & maintenance future cost for battery (in USD/MW-year), for 6 hr advanced in 2042
        battOMcostFuture = 25*1000
        # Define battery efficiency
        rtEfficiency_sqrt = sqrt(0.85)

    #### Capital Recovery Factor ####

    # Define discount rate
    d = 0.04 
    # Define project lifetime (number of years)
    n = 25
    # Define battery lifetime (number of years)
    n_bat = 12.5
    # Define capital recovery factor (scalar)
    CRF = (d*(1+d)**n)/((1+d)**n - 1)
    CRFbat = (d*(1+d)**n_bat)/((1+d)**n_bat - 1)
    # Define denominator of battery
    denom_batt = (1 + d)**n_bat

    #### Transmission and Substation Costs ####

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

    #### Solar and Wind Potential Installed Capacity ####

    cap_s = cap_s_df.loc[cap_s_df["PID"] == PID, "solar_installed_cap_mw"].iloc[0] # in MW
    cap_w = cap_w_df.loc[cap_w_df["PID"] == PID, "p_cap_mw"].iloc[0] # in MW

    #### Transmission Capacity ####

    # Size transmission substation capacity by multiplying wind capacity and TX percentage
    tx_MW = cap_w * tx_scen

    #### Wholesale Electricity Prices ####
    # Determine GEA associated with PID
    gea = pid_gea_df.loc[pid_gea_df["PID"] == PID, "gea"].values[0]
    # Set filepath where wholesale electricity prices are for each GEA
    ePrice_df_folder = os.path.join(inputFolder, cambium_scen + cambium_scen_yr_append, PTC_scen)
    ePrice_path = os.path.join(ePrice_df_folder, f"cambiumHourly_{gea}.csv")
    ePrice_df_wind = pd.read_csv(ePrice_path)

    #### Solar and Capacity Factors ####
    cf_s_path = os.path.join(inputFolder, "SAM", "Solar_Capacity_Factors", f"capacity_factor_PID{PID}.csv")
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

    ePrice_wind_hourly = pyomoInput_matrixToDict(ePrice_df_wind, "hour", year_char)
    model.eprice_wind = Param(model.HOURYEAR, default = ePrice_wind_hourly) # price of wind at each hour
    #model.eprice_solar = Param(model.HOURYEAR, default = ePrice_solar_hourly) # price of solar at each hour
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
    model.CRFbat = Param(default = CRFbat)
    model.capEx_tx = Param(default = totalTx_perMW)
    model.pot_w = Param(default = cap_w)
    model.pot_s = Param(default = cap_s)
    model.tx_capacity = Param(default = tx_MW)
    model.batt_rtEff_sqrt = Param(default = rtEfficiency_sqrt)
    model.batt_power_cost = Param(default = battPowerCost)
    model.batt_energy_cost = Param(default = battEnergyCost)
    model.batt_om = Param(default = battOMcost)
    model.batt_power_cost_future = Param(default = battPowerCostFuture)
    model.batt_energy_cost_future = Param(default = battEnergyCostFuture)
    model.batt_om_future = Param(default = battOMcostFuture)
    model.denom_batt = Param(default = denom_batt)

    """ ============================
    Set decision, slack, and battery variables
    ============================ """

    # DECISION VARIABLES 
    model.solar_capacity = Var(within = NonNegativeReals)
    # model.tx_capacity = Var(within = NonNegativeReals)

    # SLACK VARIABLES 
    # Slack variable potential generation (without considering curtailment)
    model.potentialGen = Var(model.HOURYEAR)
    # Slack variable actual generation (considering curtailment)
    model.actualGen = Var(model.HOURYEAR)  
    # Slack variable for lifetime revenue
    model.revenue = Var()
    # Slack variable for lifetime costs
    model.cost = Var()
    
    # BATTERY VARIABLES 
    # Maximum power of battery
    model.P_batt_max = Var(within = NonNegativeReals)
    # Maximum energy of battery
    model.E_batt_max = Var(within = NonNegativeReals)
    # charging power in time t
    model.P_char_t = Var(model.HOURYEAR, within = NonNegativeReals)
    # discharging power in time t
    model.P_dischar_t = Var(model.HOURYEAR, within = NonNegativeReals)
    # Energy of battery in time t
    model.E_batt_t = Var(model.HOUR0YEAR, within = NonNegativeReals)
    # Losses while charging in time t
    model.L_char_t = Var(model.HOURYEAR, within = NonNegativeReals)
    # Losses while discharging in time t
    model.L_dischar_t = Var(model.HOURYEAR, within = NonNegativeReals)
    # Export of electricity to grid in time t
    model.Export_t = Var(model.HOURYEAR, within = NonNegativeReals)

    """ ============================
    Define objective function
    ============================ """
    
    # Maximize profit
    def obj_rule(model):
        return model.revenue - model.cost
    model.obj = Objective(rule = obj_rule, sense = maximize)

    """ ============================
    Define constraints
    ============================ """

    ## Constraint (1) 
    ## Define potential generation = CF*capacity for wind and solar
    # combines both forms of generation
    def potentialGeneration_rule(model, t, y):
        return model.potentialGen[t, y] == model.cf_solar[t, y]*model.solar_capacity + model.cf_wind[t, y] * model.pot_w
    model.potentialGeneration = Constraint(model.HOURYEAR, rule=potentialGeneration_rule)

    ## Constraint (2) 
    ## Define actual generation is less than or equal to potential generation
    # how much is actually being delivered to the grid, less than the tx constraint
    def actualGenLTEpotentialGen_rule(model, t, y):
        return model.actualGen[t,y] <= model.potentialGen[t,y]
    model.actualGenLTEpotentialGen = Constraint(model.HOURYEAR, rule = actualGenLTEpotentialGen_rule)

    ## Constraint (3)  ## UPDATE FOR BATTERY
    ## Define actual generation must be less than or equal to transmission capacity
    def actualGenLTEtxCapacity_rule(model, t, y):
        #return model.actualGen[t, y] <= model.tx_capacity
        return model.Export_t[t, y] <= model.tx_capacity
    model.actualGenLTEtxCapacity = Constraint(model.HOURYEAR, rule = actualGenLTEtxCapacity_rule)

    ## Constraint (4)  ## UPDATE FOR BATTERY - RD VERSION ACCOUNTING FOR BATTERY LIFE THAT IS LESS THAN SOLAR AND WIND
    ## Define lifetime costs (equation #2) in net present value = overnight capital costs + NPV of fixed O&M (using annualPayments = CRF*NPV)
    def lifetimeCosts_rule(model):
        return model.cost == (model.solar_capacity*model.capEx_s) + ((model.solar_capacity*model.om_s)/model.CRF) + \
            (model.pot_w*model.capEx_w) + ((model.pot_w*model.om_w)/model.CRF) + \
                (model.tx_capacity*model.capEx_tx) + \
                (model.P_batt_max * model.batt_power_cost + model.E_batt_max * model.batt_energy_cost) +\
                    (model.P_batt_max * model.batt_om / model.CRFbat) +\
                  ((model.P_batt_max * model.batt_power_cost_future + model.E_batt_max * model.batt_energy_cost_future) +\
                      (model.P_batt_max * model.batt_om_future / model.CRFbat))/ model.denom_batt # denominator needs to be reviewed
    model.lifetimeCosts = Constraint(rule = lifetimeCosts_rule)
    
    ## Constraint (5) 
    ## Ensure that capacity is less than or equal to potential for solar (equation #5)
    def max_capacity_solar_rule(model):
        return model.solar_capacity <= model.pot_s
    model.maxCapacity_solar = Constraint(rule=max_capacity_solar_rule)

    ## Constraint (6)  ## UPDATE FOR BATTERY
    ## Define lifetime revenue
    def lifetimeRevenue_rule(model):
        #return model.revenue == sum(sum(model.actualGen[t, y] * model.eprice_wind[t, y] for t in model.t) for y in model.y)
        return model.revenue == sum(sum(model.Export_t[t, y] * model.eprice_wind[t, y] for t in model.t) for y in model.y)
    model.lifetimeRevenue = Constraint(rule = lifetimeRevenue_rule)

    # Constraint (7) 
    # Check that transmission capacity is less than wind capacity 
    # will always size tx capacity to wind capacity, never undersizes so could change tx_capacity == wind_capacity
    # def tx_wind_rule(model):
    #     return model.tx_capacity <= model.wind_capacity #+ model.solar_capacity
    # model.tx_wind = Constraint(rule=tx_wind_rule)

    # CONSTRAINT (1 - BATTERY) 
    # Battery cannot simultaneously charge and discharge
    #def batt_noSimultaneousChargingDischarging_rule(model, t, y):
    #    return model.P_char_t[t,y] * model.P_dischar_t[t,y] == 0
    #model.batt_noSimultaneousChargingDischarging = Constraint(model.HOURYEAR, rule = batt_noSimultaneousChargingDischarging_rule)
    
    # CONSTRAINT (2 - BATTERY) 
    # No charging in hour 1
    def batt_nochargingAtTime1_rule(model, t, y):
        return model.P_char_t[1,y] - model.P_dischar_t[1,y] == 0
    model.batt_nochargingAtTime1 = Constraint(model.HOURYEAR, rule = batt_nochargingAtTime1_rule)
    
    # CONSTRAINT (3a - BATTERY)  
    # initiate the battery charge at time  = 0 at 50% of maximum energy storage 
    def batt_startAt50percent_rule(model, t, y):
        return model.E_batt_t[0, y] == 0.5 * model.E_batt_max
    model.batt_startAt50percent = Constraint(model.HOUR0YEAR, rule = batt_startAt50percent_rule)
    
    # CONSTRAINT (3b - BATTERY)  
    # end the battery charge at time  = 8760 at 50% of maximum energy storage 
    def batt_endAt50percent_rule(model, t, y):
        return model.E_batt_t[8760, y] == 0.5 * model.E_batt_max
    model.batt_endAt50percent = Constraint(model.HOUR0YEAR, rule = batt_endAt50percent_rule)
    
    # CONSTRAINT (4 - BATTERY)  
    # the losses while charging in hour t is equal to the charging power times 1 - the square root of the round trip efficiency
    def batt_loss_charging_rule(model, t, y):
        return model.L_char_t[t, y] == model.P_char_t[t, y] * (1 - model.batt_rtEff_sqrt)
    model.batt_loss_charging = Constraint(model.HOURYEAR, rule = batt_loss_charging_rule)
    
    # CONSTRAINT (5 - BATTERY)  
    # the losses while discharging in hour t is equal to the discharging power plus the losses while discharging times 1 - the square root of the round trip efficiency
    def batt_loss_discharging_rule(model, t, y):
        return model.L_dischar_t[t, y] == (model.P_dischar_t[t, y] + model.L_dischar_t[t, y]) * (1 - model.batt_rtEff_sqrt)
    model.batt_loss_discharging = Constraint(model.HOURYEAR, rule = batt_loss_discharging_rule)

    # CONSTRAINT (6 - BATTERY)  
    # energy balance of the battery is equal to the energy in the previous hour plus the charging power in hour t minus discharging power minus losses
    def batt_energyBalance_rule(model, t, y):
        return model.E_batt_t[t, y] == model.E_batt_t[t-1, y] + model.P_char_t[t, y] - model.P_dischar_t[t, y] - model.L_char_t[t, y] - model.L_dischar_t[t, y]
    model.batt_energyBalance = Constraint(model.HOURYEAR, rule = batt_energyBalance_rule)
    
    # CONSTRAINT (7 - BATTERY)  CHECK THIS -- may need to change to actualGen
    # Charge in hour t must be less than or equal to the amt of potential generation 
    def batt_chargeLessThanGeneration_rule(model, t, y):
        return model.P_char_t[t, y] <= model.potentialGen[t, y]
    model.batt_chargeLessThanGeneration = Constraint(model.HOURYEAR, rule = batt_chargeLessThanGeneration_rule)
    
    # CONSTRAINT (8a - BATTERY)   
    # Discharge in hour t must be less than or equal amount of energy in the battery in time t-1
    def batt_dischargeLessThanEnergyAvailable_rule(model, t, y):
        return model.P_dischar_t[t, y] <= model.E_batt_t[t-1,y]
    model.batt_dischargeLessThanEnergyAvailable = Constraint(model.HOURYEAR, rule = batt_dischargeLessThanEnergyAvailable_rule)
    
    # CONSTRAINT (8b - BATTERY)   
    # Charge in hour t must be less than or equal to the maximum rated power of the battery 
    def batt_chargeLessThanPowerCapacity_rule(model, t, y):
        return model.P_char_t[t, y] <= model.P_batt_max
    model.batt_chargeLessThanPowerCapacity = Constraint(model.HOURYEAR, rule = batt_chargeLessThanPowerCapacity_rule)

    # CONSTRAINT (8c - BATTERY)   
    # Discharge in hour t must be less than or equal to the maximum rated power of the battery 
    def batt_dischargeLessThanPowerCapacity_rule(model, t, y):
        return model.P_dischar_t[t, y] <= model.P_batt_max
    model.batt_dischargeLessThanPowerCapacity = Constraint(model.HOURYEAR, rule = batt_dischargeLessThanPowerCapacity_rule)
    
    # CONSTRAINT (9 - BATTERY)  
    # Electricity exported to the grid is equal to actual generation plus the battery dicharge minus the battery charging power
    def batt_export_rule(model, t, y):
        return model.Export_t[t, y] == model.actualGen[t, y] + model.P_dischar_t[t, y] - model.P_char_t[t, y]
    model.batt_export = Constraint(model.HOURYEAR, rule = batt_export_rule)
    
    # CONSTRAINT (10 - BATTERY)  
    # Energy in battery at time t must be less than or equal to maximum rated energy capacity of the battery
    def batt_energyLessThanMaxRated_rule(model, t, y):
        return model.E_batt_t[t,y] <= model.E_batt_max
    model.batt_chargeLessThanRated = Constraint(model.HOURYEAR, rule = batt_energyLessThanMaxRated_rule)
    
    # # CONSTRAINT (11 - BATTERY)  
    # # Maximum battery energy is equal to the battery duration times the maximum power of the battery
    # def batt_maxEnergy_rule(model):
    #     return model.E_batt_max == model.P_batt_max * model.duration_batt
    # model.batt_maxEnergy = Constraint(rule = batt_maxEnergy_rule)

    """ ============================
    Execute optimization
    ============================ """
    model_instance = model.create_instance()
    results = opt.solve(model_instance, tee = False)

    """ ============================
    Post-processing of opitmization solution
    ============================ """
    
    # Extract optimization solutions from model instance
    solar_capacity = model_instance.solar_capacity.value
    wind_capacity = model_instance.pot_w.value
    tx_capacity = model_instance.tx_capacity.value
    revenue = model_instance.revenue.value
    cost = model_instance.cost.value
    profit = model_instance.obj()
    solar_wind_ratio = solar_capacity/wind_capacity
    batteryCap = model_instance.P_batt_max.value
    batteryEnergy = model_instance.E_batt_max.value
    
    # Define parseTimeSeries() function
    def parseTimeSeries(df, val_colname):
        # Move the index into columns
        df = df.reset_index()
        df["index"] = df["index"].astype("str")
        # Split index into hour and year columns
        df_hour_yr = df["index"].str.split(",", expand = True)
        hour = df_hour_yr[0].str.extract(r"\((\d{1,})") 
        year = df_hour_yr[1].str.extract("(\d+)")
        # Merge hour and year columns back into DataTrame
        df_parsed = hour.merge(year, left_index = True, right_index = True).merge(df[0], left_index = True, right_index = True)
        df_parsed.rename(columns = {"0_x": "hour", 
                                    "0_y": "year", 
                                    0: val_colname}, inplace = True)
        return df_parsed

    # Actual Generation
    # Extract actual generation time series from model instance
    actualGen = model_instance.actualGen.extract_values()
    # Convert dictionary to a DataFrame
    actualGen_df = pd.DataFrame.from_dict(actualGen, orient = "index")
    # Parse DataFrame to include hour and year columns
    actualGen_df_parsed = parseTimeSeries(actualGen_df, "actualGen")
    # Group parsed DataFrame by "year" column and sum actual generation values for each year
    actualGen_df_annualSum = actualGen_df_parsed[["actualGen", "year"]].groupby("year").sum()
    # Calculate lifetime actual generation
    actualGen_lifetime = actualGen_df_annualSum["actualGen"].sum()

    # Potential Generation
    # Extract potential generation time series from model instance
    potentialGen = model_instance.potentialGen.extract_values()
    # Convert dictionary to a DataFrame
    potentialGen_df = pd.DataFrame.from_dict(potentialGen, orient = "index")
    # Parse DataFrame to include hour and year columns
    potentialGen_df_parsed = parseTimeSeries(potentialGen_df, "potentialGen")
    # Group parsed DataFrame by "year" column and sum potential generation values for each year
    potentialGen_df_annualSum = potentialGen_df_parsed[["potentialGen", "year"]].groupby("year").sum()
    # Calculate lifetime potential generation
    potentialGen_lifetime = potentialGen_df_annualSum["potentialGen"].sum()

    # Curtailment
    # Calculate lifetime curtailment
    curtailment = (potentialGen_lifetime - actualGen_lifetime) / potentialGen_lifetime

    # Export
    # Extract export time series from model instance
    export = model_instance.Export_t.extract_values()
    # Convert dictionary to a DataFrame
    export_df = pd.DataFrame.from_dict(export, orient = "index")
    # Parse DataFrame to include hour and year columns
    export_df_parsed = parseTimeSeries(export_df, "export")
    # Group parsed DataFrame by "year" column and sum export values for each year
    export_df_annualSum = export_df_parsed[["export", "year"]].groupby("year").sum()
    # Reset index
    export_df_annualSum = export_df_annualSum.reset_index()
    # Convert "year" column to an integer type to ensure proper sorting/future
    export_df_annualSum['year'] = export_df_annualSum["year"].astype('int')
    # Sort DataFrame by year (chronological order)
    export_df_annualSum.sort_values(by = ['year'], inplace = True)
    # Discount annual export, add as a new column
    export_df_annualSum['export_discounted'] = export_df_annualSum['export'] / (1+d)**export_df_annualSum['year']
    # Calculate discounted lifetime revenue
    export_lifetime_discounted = export_df_annualSum['export_discounted'].sum()
    # Calculate undiscounted lifetime revenue
    export_lifetime = export_df_annualSum['export'].sum()

    # export_df_path = os.path.join(current_dir, 'results', 'HPCscenarios', 'test', 'export_df_parsed.csv')
    # export_df_annualSum_path = os.path.join(current_dir, 'results', 'HPCscenarios', 'test', 'export_df_annualSum.csv')
    # export_df_parsed.to_csv(export_df_path, index = True)
    # export_df_annualSum.to_csv(export_df_annualSum_path, index = True)
    # print(f'export_lifetime_discounted: {export_lifetime_discounted}')
    # print(f'export_lifetime: {export_lifetime}')
    
    # Revenue
    # Extract electricity price time series from model instance
    eprice_wind = model_instance.eprice_wind.extract_values()
    # Convert dictionary to a DataFrame
    eprice_wind_df = pd.DataFrame.from_dict(eprice_wind, orient = "index")
    revenue_df = export_df * eprice_wind_df
    # Parse DataFrame to include hour and year columns
    revenue_df_parsed = parseTimeSeries(revenue_df, "revenue")
    # Group parsed DataFrame by "year" column and sum revenue values for each year
    revenue_df_annualSum = revenue_df_parsed[["revenue", "year"]].groupby("year").sum()
    
    # Levelized Value of Electricity (LVOE)
    # Reset index for merge()
    revenue_df_annualSum = revenue_df_annualSum.reset_index()
    # Convert year column to integer type
    revenue_df_annualSum["year"] = revenue_df_annualSum["year"].astype("int")
    # Merge revenue and export to calculate LVOE
    LVOE_annual = revenue_df_annualSum.merge(export_df_annualSum, how = "left", on = "year")
    # Calculate LVOE for each year
    LVOE_annual["LVOE"] = LVOE_annual["revenue"] / LVOE_annual["export_discounted"]
    # Calculate mean lifetime LVOE
    LVOE_annual["LVOE"].mean()
    # Calculate LVOE
    LVOE = revenue / export_lifetime_discounted
    
    # Net Value of Electricity (NVOE)
    NVOE = (revenue - cost) / export_lifetime_discounted
    
    # Levelized Cost of Electricity
    LCOE = cost / export_lifetime_discounted
    
    # Battery Discharge
    discharge = model_instance.P_dischar_t.extract_values()
    # Convert dictionary to a DataFrame
    discharge_df = pd.DataFrame.from_dict(discharge, orient = "index")
    # Parse DataFrame to include hour and year columns
    discharge_df_parsed = parseTimeSeries(discharge_df, "discharge")
    # Group parsed DataFrame by "year" column and sum discharge values for each year
    discharge_df_annualSum = discharge_df_parsed[["discharge", "year"]].groupby("year").sum()

    # Battery Charge
    charge = model_instance.P_char_t.extract_values()
    # Convert dictionary to a DataFrame
    charge_df = pd.DataFrame.from_dict(charge, orient = "index")
    # Parse DataFrame to include hour and year columns
    charge_df_parsed = parseTimeSeries(charge_df, "charge")
    # Group parsed DataFrame by "year" column and sum charge values for each year
    charge_df_annualSum = charge_df_parsed[["charge", "year"]].groupby("year").sum()
    
    # Potential Wind Generation
    potentialGen_wind = pd.melt(cf_w_df, value_vars = [str(item) for item in range(1,26)], id_vars = "hour", var_name = "year")
    potentialGen_wind["potentialGen"] = wind_capacity * potentialGen_wind["value"]
    potentialGen_wind_lifetime = potentialGen_wind["potentialGen"].sum()
    
    # Potential Solar Generation
    # Calculate potential solar generation based on capacity and capacity factor for each hour
    potentialGen_solar = pd.melt(cf_s_df, value_vars = [str(item) for item in range(1,26)], id_vars = "hour", var_name = "year")
    potentialGen_solar["potentialGen"] = solar_capacity * potentialGen_solar["value"]
    potentialGen_solar_lifetime = potentialGen_solar["potentialGen"].sum()
    
    # Calculate ratio of potential solar generation to potential wind generation
    solar_wind_ratio_potentialGen = potentialGen_solar_lifetime / potentialGen_wind_lifetime

    # Append to DataFrame
    output_df = output_df_arg.append({"PID" : int(PID), 
                    "solar_capacity" : solar_capacity, 
                    "wind_capacity" : wind_capacity, 
                    "solar_wind_ratio" : solar_wind_ratio, 
                    "tx_capacity" : tx_capacity, 
                    "batteryCap" : batteryCap, 
                    "batteryEnergy": batteryEnergy, 
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
                    "exportGen_lifetime": export_lifetime,
                    "curtailment": curtailment}, 
                    ignore_index = True)
    return output_df


""" ============================
Define optimization function given a list 
============================ """

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

start_time = time.time()

output_df_complete = runOptimizationLoop(list_batch_iter, output_df)  

print("** Completed iteration ", str(i_job), "with filename ", output_df_path_iterations)
print("**** Completed loop for PID starting with", str(list_batch_iter[0]), "and ending with", str(list_batch_iter[-1]))

# Wait for all jobs to be completed
comm.Barrier()
if i_job == 0:
    print("Concatenating iterations into one csv and save to drive")
    all_csvs_iter = glob.glob(os.path.join(output_df_path_iterationsFolder, "*.csv"))
    combined_df = pd.concat((pd.read_csv(f) for f in all_csvs_iter), axis = 0, ignore_index=True)
    combined_df.to_csv(output_df_path)
    
    missingPIDs_list = list(set(seq["PID"].tolist()) - set(combined_df["PID"].tolist()))
    print("Missing PIDs:", missingPIDs_list)
    missingPIDs_df = pd.DataFrame(missingPIDs_list, columns=["missingPIDs"])
    missingPIDs_df.to_csv(output_df_path_missingPIDs)

else:
    comm.send(output_df_complete, dest = 0, tag = 11)
    
end_time = time.time()

print("Time taken:", end_time - start_time, "seconds")
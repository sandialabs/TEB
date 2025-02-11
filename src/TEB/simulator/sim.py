# -*- coding: utf-8 -*-
"""

License Statement:

Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government 
retains certain rights in this software.

BSD 2-Clause License

Copyright (c) 2021, Sandia National Laboratories
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
===========================END OF LICENSE STATEMENT ==========================


Created on Thu Oct  8 15:46:20 2020

This is a python file that uses the RC_BuildingSimulator to produce a predicted
total demand load based on several inputs.


The RC_BuildingSimulator is not available on pypi.org using conda or pip. You
must download it from its GIT repository:
    
    https://github.com/architecture-building-systems/RC_BuildingSimulator
    
The command 'git install RC_BuildingSimulator' will install download the repository
which can then be placed in the 'site-packages' folder of you python 3.8 install.

See the following documentation for performing Hourly Simulation using this 
model:
    
    https://github.com/architecture-building-systems/RC_BuildingSimulator/wiki/Hourly-Simulation
    
    
references 

[1] Chapter 11 of Expanded Shale, Clay & Slate Institute (ESCSI) "Properties of Walls Using 
    Lightweight Concrete and Lightweight Concrete Masonry Units" April 2007 Accessed 10/8/2020
    https://www.escsi.org/wp-content/themes/escsi/assets/images/11%20Chapter%2011%20Properties%20of%20Walls%20Using%20LWC%20and%20LWCMU.pdf
    I have locally stored this file in:
        C:\\Users\\dlvilla\\Documents\\BuildingEnergyModeling\\Resilience\\ResilientCommunities_Jeffers\\ElCano\\References\\2007-ExpandedShaleClayAndSlateInstitute-PropertiesOfWalls-Chapter11.pdf
        
[2] Gao, Bingtuan, Xiaofeng Liu, and Zhenyu Zhu. 2018. "A bottom-up model for household 
    load profile based on the consumption behavior of residents" energies 11: 2112 www.mdpi.com/journal/energies
    
[3] Stull, Roland. 2011. "Web-Bulb Temperature from Relative Humidity and Air Temperature." American Meteorological Society
    
TODO list:
    
    Complex appliances
    
    1. Refrigerators/freezers for industrial purposes must have the option to 
       reject heat to ambient rather than to intneral conditions
       
    2. You need to add Natural Gas and Propane consumption technologies for
       heating and cooking.
       
    3. You need to add other forms of construction to calculate the overall R and C 
       factors than concrete.
       
    4. Add a dryer complex appliance that can be called out as either electric or gas or propaine
       The effectiveness of venthing heat is important here.
    
    5. Major changes - create agents for occupancy of a community that have jobs,
       go places in the community, leave the community, and drive electricity use 
       through making decisions (i.e. cook food, dry cloths, etc..). You may 
       be able to pick up other behavioral models and leverage engines similar to
       


@author: dlvilla@sandia.gov 505-321-1269
"""

import os
import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from matplotlib import rc
from rcbsim.radiation import Window
from rcbsim.radiation import Location
from rcbsim.building_physics import Zone
from rcbsim import supply_system
from rcbsim import emission_system
from TEB.simulator.complex_appliances import Refrigerator, Wall_AC_Unit, Fan, Light
from TEB.simulator.thermodynamics import thermodynamic_properties
from copy import deepcopy
from matplotlib import pyplot as plt
from numbers import Number

class Unit_Convert():
    
    hours_to_seconds = 3600.0
    kW_to_Watts = 1000.0
    days_to_hours = 24
    kWh_per_day_to_AvgWatts_per_hour = kW_to_Watts

def parallel_building_run(building,master_building_name,tier,building_name,stop_time,troubleshoot):
    # inputtup = inputdict["main"]
    # building = inputtup[0]
    # master_building_name = inputtup[1]
    # tier = inputtup[2]
    # building_name = inputtup[3]
    # stop_time = inputtup[4]
    # troubleshoot = inputtup[5]
    
    building.run_model(0,stop_time,troubleshoot,master_building_name)
    
    return_tup = (building.Results,master_building_name,tier,building_name)
    
    return return_tup
    
    
    

class TieredAnalysis(object):
    
    # this tier map provides the number of tiers that are turned on given
    # a specific tier
    noncrit_tier = "Non-critical"
    tier_map = {noncrit_tier:[noncrit_tier,"Tier 1","Tier 2","Tier 3"],
                "Tier 3":["Tier 1","Tier 2","Tier 3"],
                "Tier 2":["Tier 1","Tier 2"],
                "Tier 1":["Tier 1"]}
    months = np.array(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    
    def __init__(self,tiered_load_spreadsheet_path,troubleshoot=False,
                 stop_time=8760,results_path="Results",run_parallel=False,run_name=""):
        self.run_name = run_name
        self.result_path = results_path
        if run_parallel:
            import multiprocessing as mp
            pool = mp.Pool(mp.cpu_count()-1)
            
        dat = ReadInputSpreadSheet(tiered_load_spreadsheet_path)
        # remove invalid columns with "Unamed:" in the string
        for key,val in dat.inputs.items():
            for col in val.columns:
                if "Unnamed: " in col:
                    val.drop([col],axis=1,inplace=True)
        
        run_name = os.path.basename(tiered_load_spreadsheet_path).split(".")[0]
        # reassure the input is correct (spreadsheets! Ugghh)
        # TODO get this working
        #TieredDataFormat(tier_loads)
        tiers = self.tier_map.keys()
        self._num_tier = len(tiers) # TODO - generalize this!
        self._num_building = len(dat.buildings.columns) # TODO generalize this
        
        # Add repeat building configurations from sheet "RepeatBuildingConfigs"
        repeat_names = self._repeat_names(dat)
            
        results = {}
        buildings = {}
        building_objects = {}
        
        run_list = {}
        # loop over buildings to create the needed data structure
        for building_name, building_data in dat.buildings.items():
            #loop over tiers
            tier_results = {}
            building_objects[building_name] = {}
            for tier in tiers:
                # Establishes all of the complex appliances but does NOT 
                # assign numbers of complex appliances to buildings. 
                self._create_complex_appliances(dat,building_name,tier)
                
                # Here, all data preparation for creating an RC model is
                #       accomplished and then the RC Building simulator Zone 
                #       class is invoked.
                #       1) building inputs are altered by tier
                #       2) static electric loads per tier are aggregated
                #       3) complex appliances are added to the building.
                buildings = self._create_RC_building_model(building_name, 
                                                          building_data,
                                                          dat,
                                                          tier,
                                                          repeat_names)
                sub_building_results = {}
                for bname,building in buildings.items():
                    #TODO - generalize run time window for the buildings
                    run_list[building_name + "_" + tier + "_" + bname] = {"main":(
                        building,building_name,tier,bname,stop_time,troubleshoot)}
                    
                    if not run_parallel:
                        building.run_model(0,stop_time,troubleshoot,building_name)
                        sub_building_results[bname] = building.Results
                        
                building_objects[building_name][tier] = deepcopy(buildings)
                if not run_parallel:
                    tier_results[tier] = sub_building_results
            if not run_parallel:    
                results[building_name] = tier_results
        # execute all runs in parallel
        #
        if run_parallel:
            
            async_results = []
            
            for key, run_list_dict in run_list.items():
                inputtup = run_list_dict["main"]
                building = inputtup[0]
                master_building_name = inputtup[1]
                tier = inputtup[2]
                building_name = inputtup[3]
                stop_time = inputtup[4]
                troubleshoot = inputtup[5]
                
                async_results.append(pool.apply_async(parallel_building_run,args=(building,master_building_name,tier,building_name,stop_time,troubleshoot)))
            
            # get waits till each process is done. Get results into the
            # tri level dictionary form the non-parallel loop creates.
            for ares in async_results:
                lres = ares.get()
                if not lres[1] in results:
                    results[lres[1]] = {}
                if not lres[2] in results[lres[1]]:
                    results[lres[1]][lres[2]] = {}
                results[lres[1]][lres[2]][lres[3]] = lres[0]
            
            # ready for _write_results             
            pool.close()
            
        df_results = self._write_results(results,run_name,results_path,
                                         stop_time,dat)
        self.df_results = df_results
        self.buildings = building_objects
        self.input_data = dat
        
        # build all of the results into a single dataframe
        return
    
    def post_process(self,building_mixture_law_dict,combined_name,plot_results=False,title="",
                     save_file_name=None):
        """
        This function takes the existing data-frame and creates a composite 
        total electricity signal and 1) Normalizes the results to a specific 
        total Area in meters 2 2) Does a weighted average sum of all "Buildings"
        for each "MasterBuilding" Entry. Buildings are the same kind of MasterBuilding
        with the different occupant and load behavior inside. For example, a town home
        can be occupied by healthy individuals or by individuals that have home health
        care requirements. The calculation allows the user to say 20% of the townhomes
        have home health care needs and 80% are normal. This can be fused with other
        categories such as With AC / without AC etc..
        
        Inputs:
            building_mixture_law_dict : dict :
                A dictionary of dictionaries. The key is the set of MasterBuildings
                for the tiered analysis. Each key value is a dictionary with two
                keys:
                    1) "AllBuildingsArea" which should be in meters squared
                and include total floor space area of all buildings 
                    2) "BuildingMixture" which must contain a dictionary with keys = The repeat_name buildings
                names and the MasterBuilding name with weights for each building.
                
        """
        conserved_columns = ["SolarGains","PlugInFans","StaticElectricLoads",
                             "Refrigerators","Wall_ACs","Lights",
                             "TotalElectricity","UnmetCooling","Occupants",
                             "UnmetHeating","HeatLoadToMeetThermostat","Central_AC"]
        
        df = self.df_results
        ind = 0
        list_df = []
        for tier,not_used in self.tier_map.items():
            df_tr = df[df["Tier"]==tier]
            df_total = None
            total_area = 0.0
            for master_building_name, mixlaw_dict in building_mixture_law_dict.items():
                if not "AllBuildingsArea" in mixlaw_dict or not "BuildingMixture" in mixlaw_dict:
                    raise KeyError("The input structure for input" +
                                   " 'building_mixture_law_dict' must be a" +
                                   " dictionary with keys 'AllBuildingArea' " +
                                   "and 'BuildingMixture'!")
                else:
                    
                    df_mr = df_tr[df_tr["MasterBuilding"]==master_building_name]
                    normfact = mixlaw_dict["AllBuildingsArea"]
                    total_area += normfact 
                    
                    df_sum = None
                    sum_weights = 0.0
                    
                    for building_name, weight in mixlaw_dict["BuildingMixture"].items():
                        if not isinstance(weight, Number):
                            raise TypeError("The entries of the 'BuildingMixture' "+
                                            "subdictionary must be numeric values!")
                        else:
                            df_bg = df_mr[df_mr["Building"]==building_name]
                            
                            if not len(df_bg) == 0:
                                sum_weights += weight
                            
                                if df_sum is None:
                                    df_sum = weight * df_bg[conserved_columns]/df_bg["BuildingArea"].iloc[0]
                                else:
                                    df_bg.index = df_sum.index
                                    df_sum = df_sum + weight * df_bg[conserved_columns]/df_bg["BuildingArea"].iloc[0]
                    
                    # normalize by the total area desired.    
                    if sum_weights == 0:
                        # do nothing
                        pass
                        #import pdb; pdb.set_trace()
                        #raise ValueError("None of the buildings for " + master_building_name + " have a simulation. Please correct the input!")
                    else:
                        # normfact renormalizes to the total square footage desired.
                        if df_total is None:
                            df_total = normfact * df_sum / sum_weights
                        else:
                            df_sum.index = df_total.index
                            df_total = df_total + normfact * df_sum / sum_weights
            if df_total is None:
                raise ValueError("None of the buildings for " + master_building_name + " have a simulation. Please correct the input!")
            # add columns so this can be appended to the final result.
            df_total["MasterBuilding"] = combined_name
            # WARNING! we 
            df_total["BuildingArea"] = total_area
            df_bg.index = df_total.index
            df_total["Date"] = df_bg["Date"]
            df_total["Tier"] = tier
            df_total.index = range(ind,ind + len(df_total))
            ind += len(df_total)
            list_df.append(df_total)
            
        df_final = pd.concat(list_df)
        
        total_area = np.array([val["AllBuildingsArea"] for key,val in building_mixture_law_dict.items()]).sum()
        
        
        if plot_results:
            total_use = self._plot_results(df_final,title)
            self._plot_daily_profiles(df_final,combined_name)
                
        eui = np.array(total_use)/total_area/Unit_Convert.kW_to_Watts
        
        if not save_file_name is None:            
            df_final.to_csv(save_file_name)
            
            
        
        return df_final, eui
    
    def _plot_daily_profiles(self,df_final,combined_name):
        min_day_ = None
        max_day_ = None
        maxval = -1e6
        minval = 1e6
        fig,axl = plt.subplots(1,2,figsize=(10,5))
        for tier in self.tier_map.keys():
            df = deepcopy(df_final[df_final['Tier']==tier])
            df.index = df['Date']
            df_daily = df.resample('1D').sum()['TotalElectricity']
            min_day = df_daily.idxmin()
            max_day = df_daily.idxmax()
            
            # assure the peaks happen on the same day as Non-critical!
            if not min_day_ is None:
                if min_day != min_day_:
                    print("Warning!:" + tier + "minimum day happened on " + str(min_day) + " but the non-critical minimum day is " + str(min_day_))
                    min_day = min_day_
            else:
                min_day_ = min_day
            if not max_day_ is None:
                if max_day != max_day_:
                    print("Warning!:" + tier + "maximum day happened on " + str(max_day) + " but the non-critical maximum day is " + str(max_day_))
                    max_day = max_day_
            else:
                max_day_ = max_day
                
            df_max_day = df[(df.index.month == max_day.month) & (df.index.day == max_day.day) & (df.index.year == max_day.year)]
            df_min_day = df[(df.index.month == min_day.month) & (df.index.day == min_day.day) & (df.index.year == min_day.year)]
            (df_min_day['TotalElectricity']/Unit_Convert.kW_to_Watts).plot(ax=axl[0],rot=60,fontsize=12,grid=True)
            (df_max_day['TotalElectricity']/Unit_Convert.kW_to_Watts).plot(ax=axl[1],label=tier,rot=60,fontsize=12,grid=True)
            
            cur_minval = df_min_day['TotalElectricity'].min() 
            cur_maxval = df_max_day['TotalElectricity'].max()
            
            if cur_minval < minval:
                minval = cur_minval
            if cur_maxval > maxval:
                maxval = cur_maxval
        
        for ax in axl:
            ax.set_ylim([minval*0.99/Unit_Convert.kW_to_Watts,maxval*1.01/Unit_Convert.kW_to_Watts])
            
        axl[0].set_xlabel("Minimum Day")
        axl[1].set_xlabel("Maximum Day")
        axl[1].legend(bbox_to_anchor=(1.00,0.7))
        for ax in axl:
            ylim =  ax.get_ylim()
            ax.set_ylim([0,ylim[1]])
        fig.suptitle(combined_name+ " Min/Max Load Profiles")
        axl[0].set_ylabel("Hourly Average Power (kW)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_path,"HourlyMaxMinDayProfiles_" + combined_name + ".png"))
        
    def _plot_results(self,df_final,title):
        """
        plot total electricity per tier and monthly contribution plots for different
        load types per building per tier.
        
        """
        font = {'family':'normal','weight':'normal','size':14}
        rc('font', **font)
        
        ElectricityUse = ["PlugInFans","StaticElectricLoads","Refrigerators","Wall_ACs",
                          "Lights","Central_AC"]
        
        total_yearly_use = []
        fig,ax = plt.subplots(1,1,figsize=(10,5))
        figb, axb = plt.subplots(len(self.tier_map),1,figsize=(10,15))
        num = 0
        monthly_title = title + " monthly sub-use breakdown"
        
        for tier, not_used in self.tier_map.items():
            df_sub = df_final[df_final["Tier"]==tier]
            df_sub.index = df_sub["Date"]
            
            df_monthly = df_sub.resample("ME").sum()[ElectricityUse]
            
            
            self._monthly_bar_plot(df_monthly.T,
                                   0.001,
                                   monthly_title + " " + tier,
                                   "Monthly Energy Use (kWh)",axb[num])
            
            
            df_sub["TotalElectricity"].plot(ax=ax,label=tier)
            total_yearly_use.append(df_sub["TotalElectricity"].sum())
            num += 1

        #Save the monthly plot of energy use break-down
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_path,monthly_title + self.run_name + ".png"),fig=figb)
        
        
        ax.set_ylabel("Total Electricity (W)")
        ax.legend(bbox_to_anchor=(1.04,0.5))
        ax.set_title(title)
        plt.sca(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_path,title + self.run_name + ".png"),bbox_inches="tight",fig=fig)

        return total_yearly_use
        
    def _monthly_bar_plot(self,
                          df_monthly,
                          conv,
                          plot_title,
                          plot_ylabel,
                          ax):
        """
        Creates a stacked bar chart of df_monthly
        
        Inputs:
            df_monthly : DataFrame :
                A dataframe with 12 columns with datetime's for each month of the
                relevant year for the 12 months of the year. Each row label should
                reflect a unique building/tier/electric load signal.
            conv : unit conversion factor for data in df_monthly
            
        
        
        
        """
        plt.sca(ax)
        legend_names = []
        axp = []
        First = True
        
        for row in df_monthly.iterrows():
            legend_names.append(str(row[0]).split(":")[0])
            if First:
                axp.append(plt.bar(self.months,row[1].values*conv,linewidth=0))
                First = False
                bot = row[1].values*conv
            else:
                if not (row[1].values < 0.0).any():
                    axp.append(plt.bar(self.months,row[1].values*conv,linewidth=0, bottom=bot))
                    bot = list(np.add(bot,row[1].values*conv))
        
        # see https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot/43439132#43439132 for legend settings
        plt.legend(axp,legend_names,bbox_to_anchor=(1.04,0.5),loc="center left",borderaxespad=0)
        plt.title(plot_title)
        plt.ylabel(plot_ylabel)
        plt.grid("on")
        
        no_fac = df_monthly
        no_fac[df_monthly < 0] = 0.0
        
        ax.set_ylim([0,abs(no_fac).sum().max()*conv*1.05])
        
        
        
        
    
    def _repeat_names(self,dat):
        bdat = dat.buildings
        bnames = bdat.columns
        repeat_bdat = dat.inputs["RepeatBuildingConfigs"]
        repeat_names = {}
        for building_name, building_data in bdat.items():
            new_names = repeat_bdat[repeat_bdat["Building"] == building_name]["New Name"].values
            for name in new_names:
                if name in bnames:
                    raise ValueError("New Name column in 'RepeatBuildingConfigs'" +
                                     " must be different than all building names in 'Buildings' sheet!")
            repeat_names[building_name] = new_names
        return repeat_names

    def _write_results(self,results,run_name,results_path,stop_time,dat):
        df_long = None
        num = 0
        for master_building_name, building_data in dat.buildings.items():
            for tier_name,tier_dict in results[master_building_name].items():
                for building_name,building_dict in tier_dict.items():
                    
                    df = pd.DataFrame(building_dict)
                    df["Tier"] = tier_name
                    df["Building"] = building_name
                    new_index = range(num*stop_time+1, (num+1)*stop_time+1)
                    df["Hour"] = df.index
                    df.index = new_index
                    if df_long is None:
                        df_long = df
                    else:
                        df_long = pd.concat([df_long,df])
                    num += 1
        if not os.path.isdir(results_path):
            try:
                os.mkdir(results_path)
            except:
                results_path = "."
                print("Could not find or create '" + results_path + 
                      "' results have been written to the current directory instead.")
        df_long.to_csv(os.path.join(results_path,run_name+".csv"))
        return df_long
    
    def _calibration_comparison(self):
        pass 
    
    def _create_complex_appliances(self,dat,bname,tier):
        # populate refrigerator object
        dfr = dat.inputs['Refrigerator'] 
        dfr.index = dfr['Description']
        fridges = {}
        for i in range(len(dfr.columns)-1):
            fridge_name = dfr.columns[i+1]
            fridges[fridge_name] = Refrigerator(surf_area = dfr.loc['Fridge surface area m2',fridge_name],
                                   aspect_ratio= dfr.loc['Fridge aspect ratio',fridge_name],
                                   R_total = dfr.loc['Fridge universal (convection and conduction) resistance (m2*K/W)',fridge_name],
                                   T_diff = dfr.loc['Temperature difference between ambient and condenser and fridge and evaporator',fridge_name],
                                   T_inside = dfr.loc['Constant average temperature of fridge compartment (K) Tf',fridge_name],
                                   ach_mix=dfr.loc['Peak average air mixing rate from door opening (ACH)',fridge_name],
                                   ach_schedule=dat.schedules[dfr.loc['Name of air mixing rate from door opening',fridge_name]],
                                   frac_Carnot = dfr.loc['Fraction of Reverse Carnot Efficiency',fridge_name],
                                   fan_power = dfr.loc['Average Fan Power (W)',fridge_name],
                                   compressor_efficiency=dfr.loc['Compressor efficiency',fridge_name],
                                   compressor_power=dfr.loc['Compressor Power Rating (W)',fridge_name],
                                   name=fridge_name)
        
        # populate the AC unit objects - tables use rows instead of columns 
        # for each unit. Sorry for the inconsistency!
        df_ac = DF_ops.df_col_index_and_drop(dat.inputs,"AC","Description")
        
        df_therm = dat.inputs['ACThermostat']
        df_coef = dat.inputs['ACcoef']
        ac = {}
        thermostat_heat = None
        thermostat_cool = None
        ac_names = np.array(df_ac.columns[1:])
        for colname, ac_col in df_ac.items():
            ac_name = colname
            
            df_therm_temp = df_therm.iloc[:,1:][df_therm["AC Unit"] == ac_name]
            if len(df_therm_temp) != 0:
                # set thermostats - TODO - these really need to be moved to the
                # building level as a schedule that would allow some measure of 
                # efficiency.
                if thermostat_heat is None:
                    thermostat_heat = df_therm_temp[df_therm_temp["Type"]=="Heating (⁰C)"]
                else:
                    if (thermostat_heat.values != df_therm_temp[df_therm_temp["Type"]=="Heating (⁰C)"]).values.any():
                        raise ValueError("All Heating must have the same thermostat values (only chang accross Tiers!)")
                        
                if thermostat_cool is None:
                    thermostat_cool = df_therm_temp[df_therm_temp["Type"]=="Cooling (⁰C)"]
                else:
                    if (thermostat_cool.values != df_therm_temp[df_therm_temp["Type"]=="Cooling (⁰C)"]).values.any():
                        raise ValueError("All AC must have the same cooling thermostat values (only change accross Tiers!)")
                
                ac[ac_name] = Wall_AC_Unit(TC=ac_col['Name Plate Total Cooling (TC) (W)'],
                                           SCfrac=ac_col['Sensible Cooling Fraction'],
                                           derate_frac=ac_col['Name Plate Derate Fraction'],
                                           npPower=ac_col['Power input (W)'],
                                           flow_rated=ac_col['Air flow rate (m3/h)'],
                                           df_coef=df_coef.iloc[:,1:][df_coef["AC Unit"] == ac_name],
                                           df_thermostat=df_therm_temp,
                                           tier=tier,
                                           Name=ac_name)
            else:
                print("An empty column must exist in the input spreadsheet for the AC sheet!")
        self.thermostat_heat = thermostat_heat
        self.thermostat_cool = thermostat_cool
        
        # fans
        df_fan = DF_ops.df_col_index_and_drop(dat.inputs,"Fan","Description")
        df_fan_curve = DF_ops.df_col_index_and_drop(dat.inputs,'FanCurves',"Speed")
        df_fan_temp = DF_ops.df_col_index_and_drop(dat.inputs,'FanTemperatures','Speed')
        
        fans = {}
        for colname, fancol in df_fan.items():
            fan_name = colname
            
            fans[fan_name] = Fan(fan_name=fan_name, 
                                 power=fancol["Power Consumption (W)"], 
                                 heat_energy_ratio=fancol["Heat Energy Ratio"], 
                                 ExtACH=fancol["External Air Changes"], 
                                 SpeedEfficiencies=df_fan_curve[fancol["Efficiency Curve"]], 
                                 Curve_Temperatures=df_fan_temp[fancol["Efficiency Curve"]])
        
        lights = {}
        df_light = DF_ops.df_col_index_and_drop(dat.inputs,"Lighting","Description")
        for colname, lightcol in df_light.items():
            light_name = colname
            lights[light_name] = Light(name=light_name,
                                       light_type=lightcol["Type"],
                                       lux_threshold=lightcol["Lighting control lux threshold (lux)"],
                                       power=lightcol["Power (W)"],
                                       fraction_heat=lightcol["Fraction Heat"])
            
        self.lights = lights 
        self.wall_acs = ac
        self.fridges = fridges
        self.fans = fans
        
    
    def _find_schedule(self,dat,SchName):
        if not type(SchName) is str:
            print("The input for a Schedule is not correct. Please investigate your input spreadsheet!")
            return None
        else:
            try:
                schedule = dat["Schedules"][SchName]
            except KeyError:
                raise KeyError("The schedule " + SchName + " does not exist!")
            except:
                raise ValueError("Unknown Error!")
            return schedule
            
    def _derive_static_loads_schedule(self,dat,building_name,tier, building_area, repeat_names):
        """
        This function derives a total static electricity load dissipated in a building
        The building can be represented by 'repeat_names' where only a small aspect
        of the building is being changed as well.
                
        Inputs -
           dat - dict - dictionary for all inputs of the input spreadsheet
           building_name - str - a unique identifier of one of the buildings in
                           the 'Buildings' sheet of the input 
           tier - str - a unique identifier indicating what Tier simulation
                        is being run
           building_area - float - total floor space of the building being 
                                   simulated.
           repeat_names - additional building names where the base building is 
                          building_name but that additional changes are applied
                          to form a new simulation.
                          
        # repeat names must belong to a building_name and either:
        #  1. Replaces electronics schedule if the Load Name is a repeat
        #  2. Adds a new electronics schedule in addition to the original building
        #     name
        
        """
        def _load_dict(bstat,dat,building_area):
            load_dict = {}
            # bstat has already been filtered for building and tier_list
            for bname, row in bstat.iterrows():
                app_sch = self._find_schedule(dat, row["Schedule"])
                if not app_sch is None:
                    if row["Per Area (W = False or W/m2 = True?)"]:
                        app_peak = row["Multiplier"] * Unit_Convert.kW_to_Watts * row["Energy (kWh/day or kWh/day/m2)"] # already in W/m2
                    else: # is on a W basis and needs to be on a W/m2 basis!
                        app_peak = row["Multiplier"] * Unit_Convert.kW_to_Watts * row["Energy (kWh/day or kWh/day/m2)"] / building_area
                # verify that the tier and load name are unique
                unique_key = row["Load Name"] + "_" + row["Tier"]
                if unique_key in load_dict.keys():
                    raise ValueError("The unique key '" + unique_key + 
                                     "' is already in use for building '" + bname + "' Please correct the" +
                                     " input spreadsheet to not include repeat" +
                                     " Tier and Load Names for a given building!")
                else:
                    load_dict[unique_key] = {"Wh_per_day_per_m2":app_peak,
                                             "Sch":app_sch,
                                             "FracHeat":row["Fraction Heat Added"]}
            return load_dict
                
        def _sum_schedules(schdict,name,building_area,tier,noncrit_tier,result_path):
            
            sch_sum = None
            heat_sch_sum = None
            for key,sch in schdict.items():
                if sch_sum is None:
                    sch_sum = sch["Wh_per_day_per_m2"] * sch["Sch"] / sch["Sch"].sum()
                    heat_sch_sum = sch_sum * sch["FracHeat"]
                else:
                    Elec = sch["Wh_per_day_per_m2"] * sch["Sch"] / sch["Sch"].sum() # should be normalized but this kind of schedule must sum to 1. and be of length = 24
                    sch_sum += Elec
                    heat_sch_sum += Elec * sch["FracHeat"] 
            if tier == noncrit_tier:
                fig,ax = plt.subplots(1,1,figsize=(10,10))
                (pd.DataFrame(schdict).loc["Wh_per_day_per_m2"]/Unit_Convert.kW_to_Watts*building_area).plot.bar(ax=ax,title=name,grid=True,fontsize=12)
                ax.set_ylabel("Static Electric Load Type (kWh/day)")
                plt.tight_layout()
                plt.savefig(os.path.join(result_path,"StaticElectricLoads_" + name + self.run_name + ".png"))
            return sch_sum, heat_sch_sum
                
        
        # reduce the static loads to only the building and tier list of interest for 
        # the current run.
        result_path = self.result_path
        static = dat["StaticElectricLoads"]
        rnames = repeat_names[building_name]
        tier_list = self.tier_map[tier]
        bstat = {}
        
        bstat_org = DF_ops.column_multiselect(static[static["Building"]==building_name], "Tier", tier_list)
        for rname in rnames:
            bstat[rname] = DF_ops.column_multiselect(static[static["Building"]==rname], "Tier", tier_list)

        org_load_dict = _load_dict(bstat_org, dat, building_area[building_name])
        tot_sch, tot_heat_sch = _sum_schedules(org_load_dict,building_name,building_area[building_name],tier,self.noncrit_tier,self.result_path)
        # rename sum to a meaningful name.
        if tot_sch is None:
            self.warning_messages.append("There are no static electricity loads for building " + building_name + " is this intentional??")
            tot_sch = pd.Series(np.zeros(24),name="no static loads")
            tot_heat_sch = tot_sch
        
        tot_sch.rename("Total Appliance Electric Loads",inplace=True)
        tot_heat_sch.rename("Total Appliance Heat Loads",inplace=True)
        
        repeat_names_tot_schs = {}
        repeat_names_tot_heat_schs = {}
        for key,val in bstat.items():
            # build off of the base-building for the repeat name.
            repeat_dict = deepcopy(org_load_dict)

            new_entries =  _load_dict(val,dat, building_area[key])

            
            for unique_key, schdict in new_entries.items():
                # this either replaces or adds to the existing dictionary of schedules
                repeat_dict[unique_key] = schdict
            repeat_names_tot_schs[key], repeat_names_tot_heat_schs[key] = _sum_schedules(repeat_dict,key,building_area[key],tier,self.noncrit_tier,self.result_path)
            # gives these correct names
            repeat_names_tot_schs[key].rename("Total Appliance Electric Loads",inplace=True)
            repeat_names_tot_heat_schs[key].rename("Total Appliance Heat Loads",inplace=True)
            
        if tot_sch.isna().sum() > 0 or tot_heat_sch.isna().sum() > 0:
            raise ValueError("The input spreadsheet must have an incorrect"+
                             "value or blank value where it needs an entry."+
                             "A NaN has been detected!")
            
        return tot_sch, tot_heat_sch, repeat_names_tot_schs, repeat_names_tot_heat_schs
            
    def _derive_complex_appliances(self,name,dat,tier,repeat_names,
                                   org_complex_appliances=None,
                                   org_df_tier_dict=None):
        """
        

        Parameters
        ----------
        name : str
            building name (or repeat building name being applied)
        dat : ReadInputSpreadsheet object
            Contains all of the input data from the input excel spreadsheet
        tier : str
            current tier being evaluated
        repeat_names : dict
            each key contains the name of a building from the "Buildings" sheet
            which accesses a list of names that have all of the same attributes
            as the key name except where an input exists with the new "repeat" name
        org_complex_appliances : None or dict, optional
            When None, the function is being called for the first time otherwise
            this is the complex appliances derived for the original building name
            and which needs to be altered for each repeat_name. The default is None.
        org_df_tier_dict : None or dict, optional
            Must be None when org_complex_appliances is None. Contains a dataframe
            for each complex appliance type that provides the exact counts for each
            tier and complex appliance type so that repeat names can replace the
            number of that appliance if an overwrite is needed.

        Raises
        ------
        ValueError
            Some complex data structures are required by this function it will
            fail with a ValueError if something goes wrong during try/except
            attempts. 

        Returns
        -------
        complex_appliances : dict of dict
            Contains specification of each complex appliance type in the
            manually maintained "sheet_names" with count and objects for each
            appliance type so that the run_model method in the RCBuilding class
            can determine how many appliances of each type are operational.

        """
        # This function is used recursively
        # all complex appliance objects have already been added to self in _create_complex_appliances
        
        # Every complex appliance has to have two excel spreadsheets.
        # "tab" is the sheet name that contains the actual parameters for a given technology
        # "numtab" is the sheet that contains how many of each technology are in a given building.
        
        # ALL of this eventually needs to be changed to a relational database or
        #     a database structure similar to WNTR's
        # Manual input is required here if you are adding a new complex appliance.
        
        # verify optional inputs are configured properly.
        not_repeat_name = org_complex_appliances is None
        
        if (not_repeat_name) and not (org_df_tier_dict is None):
            raise ValueError("org_complex_appliances is None but df_tier_dict is not None! Invalid input combination for this function!" +
                             " Both must be None or both must not be None!")
        elif not (not_repeat_name) and (org_df_tier_dict is None):
            raise ValueError("org_complex_appliances is not None but df_tier_dict is None! Invalid input combination for this function!" +
                             " Both must be None or both must not be None!")
        
        sheet_names = {"fridges":{"numtab":"BuildingRefrigerators","tab":"Refrigerator"},
                       "fans":{"numtab":"BuildingFans","tab":"Fan"},
                       "wall_acs":{"numtab":"BuildingWallAC","tab":"AC"},
                       "lights":{"numtab":"BuildingLighting","tab":"Lighting"}}
        
        
        
        if not_repeat_name:
            complex_appliances = {}  
        else:
            complex_appliances = deepcopy(org_complex_appliances)
            
        df_tier_dict = {}
        for key,tdict in sheet_names.items():
            # grab the current appliance's sheet dataframe
            df = dat.inputs[tdict["numtab"]]
            
            # valid names are all columns after the "Description" column.
            valid_names_desc = np.array(dat.inputs[tdict["tab"]].columns)
            valid_names = np.delete(valid_names_desc,np.where(valid_names_desc=="Description"))
            
            df_building = df[df["Building"] == name]
            
            tier_list = []
            for inc_tier in self.tier_map[tier]:
                tier_list.append(df_building[df_building["Tier"] == inc_tier])
            df_tier = pd.concat(tier_list)
            df_tier.index = df_tier["Tier"]
            df_tier_dict[key] = df_tier
            
            if not_repeat_name:
                complex_appliances[key] = {}
                            
            for vname in valid_names:
                # initialize
                if not_repeat_name:
                    num = 0
                else:
                    num = org_df_tier_dict[key][vname].sum()
                    
                # iterate over tiers
                for rtier,row in df_tier.iterrows():
                    try:
                        num_add = row[vname]
                        # here we get ready to replace the original ("org") 
                        # number of "vname" appliances with the repeat name 
                        # number. If "rtier" doesn't exist then we just add
                        if not not_repeat_name:
                            s1 = org_df_tier_dict[key][vname]
                            
                        if not_repeat_name:
                            num_sub = 0
                        elif rtier in s1:
                            num_sub = s1[rtier]
                        
                        num += (num_add - num_sub)
                        
                    except:
                        raise ValueError("the appliance name " + vname + 
                                         " is not included. You need to update the "
                                         + tdict['numtab'] + " sheet to include it.")
                if num > 0:
                    try:
                        complex_appliances[key][vname] = (num,getattr(self,key)[vname])
                    except:
                        raise ValueError("Data structure error 2!")
                        
        # transfer these over to this object so that it can be assigned at 
        # the building level
        complex_appliances["ac_thermostat_cool"] = self.thermostat_cool
        complex_appliances["ac_thermostat_heat"] = self.thermostat_heat
        
        if not_repeat_name:
            repeat_complex_appliances = {}
            for rname in repeat_names[name]:
                (rcomplex_appliances, 
                 IsNone) = self._derive_complex_appliances(
                     rname,
                     dat,
                     tier,
                     repeat_names,
                     complex_appliances,
                     df_tier_dict)
                repeat_complex_appliances[rname] = deepcopy(rcomplex_appliances)
        else:
            repeat_complex_appliances = None
        
        return complex_appliances, repeat_complex_appliances
                    
    def _alter_building_data_by_tier(self,name,building_data,dat,tier, repeat_names):
        # this only alters the building if something is input. Otherwise
        # the building input is not changed
        
        def make_changes_for_name(building_tiers,name,tier,building_data):
            
            btiers_downselect = building_tiers[building_tiers["Building Name"] == name].loc[:,["Description",tier]]
            for ind,row in btiers_downselect.iterrows():
                if row["Description"] in building_data:
                    building_data[row["Description"]] = row[tier]
                else:
                    raise ValueError("The item: " + row["Description"] + " is not a valid name for the 'Buildings' sheet" + 
                                     ". Please correct the input!")
            return building_data
        
        
        
        building_tiers = dat.inputs["BuildingsTiers"]

        building_data = make_changes_for_name(building_tiers,name,tier,building_data)
        repeat_building_data = {}
        for rname in repeat_names[name]:
            rdata = building_data.copy()
            rdata = make_changes_for_name(building_tiers,rname,tier,rdata)  
            repeat_building_data[rname] = rdata

        return building_data, repeat_building_data
            
        
            
    
    def _create_RC_building_model(self,name,building_data,dat,tier,repeat_names):
        
        # Change the building based on the "BuildingTiers" sheet which allows 
        # redefinition of values in the "Buildings" sheet based on tiers.
        (org_building_data, 
         repeat_building_data) = self._alter_building_data_by_tier(name, 
                                                                   building_data, 
                                                                   dat, 
                                                                   tier,
                                                                   repeat_names)
        # floor area and occupant schedules
        org_total_floor_area = building_data["Floor Area (m2)"] * building_data["Number of Floors"]
        org_occupant_schedule = self._find_schedule(dat.inputs, 
                                                building_data["Occupant Schedule"])
        repeat_total_floor_area = {}
        repeat_occupant_schedule = {}
        for rname,bdata in repeat_building_data.items():
            repeat_total_floor_area[rname] = bdata["Floor Area (m2)"] * bdata["Number of Floors"]
            repeat_occupant_schedule[rname] = self._find_schedule(dat.inputs, 
                                                bdata["Occupant Schedule"])
        # For floor area the original name has to be assigned earlier because 
        # it is used by _derive_static_loads_schedule
        repeat_total_floor_area[name] = org_total_floor_area    

        # static electric loads        
        (org_appliance_schedule,
         org_appliance_heat_schedule,
         repeat_name_appliance_schs,
         repeat_name_appliance_heat_schs) = self._derive_static_loads_schedule(dat.inputs,
                                                                       name, 
                                                                       tier, 
                                                                       repeat_total_floor_area, 
                                                                       repeat_names)

        # complex appliances applied to each building.
        (org_complex_appliances,
         repeat_complex_appliances) = self._derive_complex_appliances(name,
                                                             dat,
                                                             tier,
                                                             repeat_names)
        # now add the original building to the repeat list so that we
        # can loop through everything
        repeat_building_data[name] = org_building_data
        repeat_name_appliance_schs[name] = org_appliance_schedule
        repeat_occupant_schedule[name] = org_occupant_schedule
        repeat_complex_appliances[name] = org_complex_appliances
        repeat_name_appliance_heat_schs[name] = org_appliance_heat_schedule
        
        buildings = {}
        for rname,building_data in repeat_building_data.items():
            occupant_schedule = repeat_occupant_schedule[rname]
            complex_appliances = repeat_complex_appliances[rname]
            appliance_schedule = repeat_name_appliance_schs[rname]
            appliance_heat_schedule = repeat_name_appliance_heat_schs[rname]    
            buildings[rname] = RCBuilding(building_name = rname, 
                            floor_area = building_data["Floor Area (m2)"], #m2
                            floor_height = building_data["Floor Height (m)"],
                            num_floors = building_data["Number of Floors"],
                            person_per_floor = building_data["Person per Floor"],
                            building_aspect_ratio = building_data["Building Aspect Ratio"],
                            fraction_internal_walls = building_data["Fraction Internal Walls"],
                            window_to_wall_fraction_north = building_data["Window/Wall Ratio North"],
                            window_to_wall_fraction_east = building_data["Window/Wall Ratio East"],
                            window_to_wall_fraction_south = building_data["Window/Wall Ratio South"],
                            window_to_wall_fraction_west = building_data["Window/Wall Ratio West"],
                            external_wall_and_roof_avg_thickness = building_data["External Wall/Roof Average Thickness (m)"],
                            concrete_density = building_data["Concrete Density (kg/m3)"],
                            concrete_aggregate_vol_frac = building_data["Concrete Aggregate Volume Fraction (for > 1920 kg/m3)"],
                            concrete_aggregate_conductivity = building_data["Concrete Aggregate Phase Thermal Conductivity (for > 1920 kg/m3)"],
                            concrete_paste_conductivity = building_data["Concrete Paste Thermal Conductivity (for > 1920 kg/m3)"],
                            fraction_air_in_CMS_units = building_data["Fraction Air in CMS units"],
                            windows_U_factor = building_data["Windows Average U Factor"],
                            infiltration_air_changes_per_hour = building_data["Infiltration Air Changes Per Hour"],
                            weather_file = building_data["Weather File"],
                            occupant_schedule = occupant_schedule,
                            building_latitude = building_data["Building Latitude"],
                            building_longitude = building_data["Building Longitude"],
                            simulation_year = building_data["Simulation Year"],
                            start_hour_of_year = building_data["Start Hour of Year"],
                            additional_R_insulation = building_data["Additional Insulation RSI Value (m2*K/W)"],
                            additional_heat_capacity_besides_walls = building_data["Additional Non-wall Heat Capacity (J/(m2*K))"],
                            glass_solar_transmittance = building_data["Glass Solar Transmittance"],
                            glass_light_transmittance = building_data["Glass Light Transmittance"],
                            heat_gain_per_person = building_data["Heat Gain per Person (W)"], 
                            complex_appliances = complex_appliances,
                            appliance_schedule = appliance_schedule,
                            appliance_heat_schedule = appliance_heat_schedule,
                            R_per_area_ground = building_data["Resistance to Ground (m2*K/W)"],
                            elevation = building_data["Elevation (m)"],
                            ground_temp_elev_sens= building_data["Ground temperature to elevation sensitivity (K/m)"],
                            tier=tier,
                            use_central_ac=building_data["Include Central A/C"],
                            central_ac_avg_cop=building_data["Central A/C average COP (only used if Include Central A/C = True)"])
        return buildings





class DF_ops():
    """
    Data frame operations
    """
        
    @staticmethod
    def df_col_index_and_drop(dat,key,indexcolname):
        df = dat[key]
        if indexcolname in df.columns:
            df.index = df[indexcolname]
            df.drop(columns=indexcolname,inplace=True)
        return df
    
    @staticmethod
    def column_multiselect(df, col, colval):
        list_df = []
        for val in colval:
            list_df.append(df[df[col]==val])
        df_out = pd.concat(list_df)
        return df_out
        

        
        
class humidity_model(object):
    
    def __init__(self,log_warnings=True):
        self.tp = thermodynamic_properties()
        self.log_warnings = log_warnings
        if log_warnings:
            self.warnings = []
        
    
    def _humidity_model(self,mdot_water,ACH,air_pressure,internal_rh,
                   external_rh, internal_T, external_T, time_step_hr, volume):
        """
        This entire process is assumed to be isothermal The building energy model
        accounts for the air changes occuring. We just want to account for moisture
        differences driven by internal condensation via a wall-unit air-conditioner.
        
        This routine has been run through with a single run but needs unit testing
        to verify it works robustly.
        """
        # Convert ACH to m3/s
        Vdot_ACH = volume * ACH / Unit_Convert.hours_to_seconds
        
        # calculate saturated vapor pressures of water
        Psat_h2o_int = self.tp.h2o_saturated_vapor_pressure(air_pressure,
                                                                         internal_T)
        Psat_h2o_ext = self.tp.h2o_saturated_vapor_pressure(air_pressure,
                                                                         external_T)
        # water partial pressures                                                                 
        Ph2o_int = Psat_h2o_int * internal_rh
        Ph2o_ext = Psat_h2o_ext * external_rh
        # mole fractions
        yh2o_int = Ph2o_int / air_pressure 
        yh2o_ext = Ph2o_ext / air_pressure  # we assume building pressurization is minimal.
        
        # molar masses
        Mh2o = self.tp.h2o_M
        Mair = self.tp.air_M
        rho_air = self.tp.rho_air
        Rgas = self.tp.R_gas
        
        # mixed molar mass
        M_mix_int = yh2o_int * Mh2o + (1 - yh2o_int) * Mair
        M_mix_ext = yh2o_ext * Mh2o + (1 - yh2o_ext) * Mair
        
        # specific gas constants
        R_mix_int = self.tp.R_gas / M_mix_int
        R_mix_ext = self.tp.R_gas / M_mix_ext
        
        rho_mix_int = air_pressure / (R_mix_int * (internal_T + self.tp.CelciusToKelvin))
        rho_mix_ext = air_pressure / (R_mix_ext * (external_T + self.tp.CelciusToKelvin))
        
        # moles of moist air inside the building
        n_int = air_pressure * volume / (Rgas * (internal_T + self.tp.CelciusToKelvin))
        
        vapor_int = n_int * yh2o_int * Mh2o
        air_int = n_int * (1.0 - yh2o_int) * Mair
        
        moist_air_mass_flow_int = Vdot_ACH * rho_mix_int
        # assume equivalent mass of dry air is sucked in because of mdot_water leaving
        moist_air_mass_flow_ext = Vdot_ACH * rho_mix_ext + (mdot_water * rho_mix_ext / rho_air)
        
        # the building energy simulation already does a balance
        # on thermal energy but does not include cold condensed water
        # leaving by a wall unit condenser.
        
        vapor_leaving = moist_air_mass_flow_int * (yh2o_int * Mh2o /M_mix_int)
        air_leaving = moist_air_mass_flow_int * ((1-yh2o_int) * Mair /M_mix_int)
        vapor_entering = moist_air_mass_flow_ext * (yh2o_ext * Mh2o /M_mix_ext)
        air_entering = moist_air_mass_flow_ext * ((1-yh2o_ext) * Mair /M_mix_ext)
        
        # total mass flow rates
        mdot_vapor = (vapor_entering - vapor_leaving - mdot_water) 
        mdot_air = (air_entering - air_leaving) 
        
        # new mass balances
        air_int_new = air_int + mdot_air * time_step_hr * Unit_Convert.hours_to_seconds
        vapor_int_new = vapor_int + mdot_vapor * time_step_hr * Unit_Convert.hours_to_seconds
        
        if vapor_int_new < 0.0:
            if self.log_warnings:
                self.warnings.append("humidity_model: All humidity condensed by wall units,"
                 + " the mixture law assumptions of this model are invalid!" 
                 + " A smaller time step is needed!")
            vapor_int_new = 0.0
        
        # calculate new mole balances
        n_vapor_new = vapor_int_new / Mh2o 
        
        P_h2o_new = n_vapor_new * Rgas * (internal_T + self.tp.CelciusToKelvin) / volume
        
        internal_rh_new = P_h2o_new / Psat_h2o_int
        
        return internal_rh_new

class concrete_wall(object):
    """provides bulk thermal properties of a concrete wall consisting of concrete
       paste and aggregate as seen in Figure 11.1.1 of """
    Rf = 0.149 #surface/air film resistance m2K/W TODO - double check that this is not incorporated into the RC_BuildingSimulator.
    k_air = 0.026 # thermal conductivity of air (this is temperature dependent) 
                  # see https://www.engineeringtoolbox.com/air-properties-viscosity-conductivity-heat-capacity-d_1509.html
      
    def __init__(self,dc,Vair,L,added_R,Va=None,ka=None,kp=None,calc_HC=True,checkbounds=True):
        """concrete_wall(self,dc,Vair,L,Va=None,ka=None,kp=None)
        Inputs:
            dc = density of concrete (kg/m3)
            
            Vair = volume fraction of air
            
            L = wall thickness (m)
            
            added_R = additional insulation R factor (m2*K/W) R = L/k for conduction
            
            Va (optional) = aggregate volume (for 2-phase concrete model)
            
            ka (optional) = aggregate phase thermal conductivity (W/(m*K))
            
            kp (optional) = paste phase thermal conductivity (W/(m*K))
            
            calc_HC (optional) = (bool) = True - calculate heat capacity
            
            checkbounds (optional) = (bool) = Check bounds on heat capacity calculation
        
        """
        self.U_value = self._calc_U_value(dc, Vair,L, added_R, Va, ka, kp)
        if calc_HC:
            self.HC_value = self._CMS_heat_capacity_regression(100*(1-Vair),L,dc,checkbounds)
        
        
    def _calc_U_value(self,dc,Vair,L,added_R,Va=None,ka=None,kp=None):
        """
        U_value - calculate U_value of a single layer concrete wall:
            All units must be SI becaues unit dependent empirical relationships
            are encoded in this routine.
            
            Two models for thermal conductivity can be used. The first uses 
            the concrete density (dc) to provide 
            an empirical estimate of the concrete thermal conductivity. and only
            requires the first three inputs.
            
            The second assumes that a cube of aggregate material (ka) is 
            subsumed in a cubic shape of finer concrete paste (kp) and that
            the aggregate material has volume fraction Va. A mixture law
            is then applied to get the equivalent thermal conductivity.
            
            This second model must be used for dc > 1920 kg/m3
            
            A second mixture law is then applied for air void fractions assuming
            a concrete masonry unit geometry. The surface air film resistance
            and conductivity of air are assumed to be the constant values
            set within this class.
            
            Inputs:
                dc : float > 0 : density of concrete (kg/m3)
                
                Vair : 1 >= float >= 0 : Volume fraction of air (for CMS blocks)
                
                L : float > 0 : thickness of concrete wall (m)
                
                Va : 1 >= float >= 0 : (optional)
                       Volume fraction of aggregate filler material.
                       Only include this input if a two-phase model is desired
                
                ka : float >= 0 : (optional)
                       thermal conductivity of aggregate material (W/(m*K))
                
                kp : float >= 0 : (optional)
                       thermal conductivity of finer concrete paste (W/(m*K))
        
        All of the equations in this function come from [1] 
                      
        """
        # input checks
        if (dc <= 0) or (L <= 0):
            raise ValueError("Inputs dc and L must be > 0")
        elif (Vair > 1) or (Vair < 0):
            raise ValueError("Vair must be in the range [0,1]")
        elif ((Va is None or ka is None or kp is None) and 
             not (Va is None and ka is None and kp is None)):
            raise TypeError("Optional input values Va, ka, and kp must all be" +
                            " input together.")
        use_volume_model = (not Va is None) and (not ka is None) and (not kp is None)
        if use_volume_model:
            if (Va > 1) or (Va < 0):
                raise ValueError("Va must be in the range [0,1]")
            elif (ka < 0) or (kp < 0):
                raise ValueError("ka and kp must be greater than zero")
        
        if dc > 1920 and (Va is None or ka is None or kp is None):
            raise ValueError("The three phase inputs Va, ka, and kp are needed for" +
                             " concrete that is more dense than 1920 kg/m3")
        elif use_volume_model:
            kc = concrete_wall.cubic_model_conductivity(ka,Va,kp)
        else:
            kc_dry,kc_wet = concrete_wall.concrete_density_conductivity_model(dc)
            kc = kc_wet
        
        # area fraction of air
        Af_air = np.sqrt(Vair)
        Af_c = 1 - Af_air
        R_air = L*(1 - Af_air)/self.k_air
        
        Rc = L/kc
        
        RT = self.Rf + 1/(Af_air/R_air + Af_c/Rc) + added_R
        
        U_concrete_wall = 1/RT
        
        self.k_equivalent = L/(RT-self.Rf)
        
        return U_concrete_wall
    
    
        
    @staticmethod
    def cubic_model_conductivity(ka,Va,kp):
        
        """
        Inputs 
        
        dc : float > 0 - oven dry density of concrete paste
        ka : 
        
        """
        Va_2_3 = Va**(2/3)
        kc = kp * (Va_2_3/(Va_2_3 - Va + (Va/(ka*Va_2_3/kp + 1 - Va_2_3))))
        return kc
        
        
        
    @staticmethod
    def concrete_density_conductivity_model(dc):
        """
        Inputs 
        
        dc : float > 0 - oven-dry density of concrete paste in kg/m3
        
        Returns:
        kc_dry - oven dry density of concrete equation 11-1 of [1]
        kc_typical_moisture - typical moisture retained by concrete 11-3 of [1] (Valore, 1980 assumptions)
        
        
        """
        return 0.072 * np.exp(0.00125 * dc), 0.0865 * np.exp(0.00125 * dc)
    
    def _CMS_heat_capacity_regression(self,percent_solid,thickness,concrete_density,apply_data_bounds=True):
        """
        CMS_heat_capacity_regression(percent_solid,thickness,concrete_density,apply_data_bounds=True)
        
        This function neglects the heat capacity of air trapped in Concrete Masonry Units (CMS)
        Inputs:
            percent_solid : 48 > float > 100 : The amount of volume that is concrete (1-percent_solid) = % air
            
            thickness : 0.3048 > float > 0.1016  : The thickness of the CMS unit in meters (i.e. wall thickness)
            
            density : 1280 > float > 2240 : Density of concrete being used (not combined with air space) 
    
        Outputs:
            heat capacity in J/(m2*K) where m2 refers to the vertical/lateral surface area of the wall.
            
        poly 3 error max: 2.006 min: -1.714 average magnitude: 0.731
        """
        # data bounds check
        if apply_data_bounds:
            if concrete_density < 1280 or concrete_density > 2240:
                raise ValueError("Concrete density outside of valid empirical"+
                                 " range! 1280 < concrete_density < 2240.")
            elif thickness < 0.1016 or thickness > 0.3048:
                raise ValueError("Wall thickness outside of the valid empirical" +
                                 " range! 0.1016 < thickness < 0.3048.")
            else:
                if thickness >= 0.1016 and thickness < 0.1524:
                    if percent_solid < 65 or percent_solid > 100:
                        raise ValueError("percent_solid outside of the valid empirical"+
                                         " range! When thicknesS is between 0.1016 and 0.1524," +
                                         " 65 <= percent_solid <= 100!")
                elif thickness >= 0.1524 and thickness < 0.2032:
                    if percent_solid < 55 or percent_solid > 78:
                        raise ValueError("percent_solid outside of the valid empirical"+
                                         " range! When thickness is between 0.1524 and 0.2032," +
                                         " 55 <= percent_solid <= 78!")
                elif thickness >= 0.2032 and thickness < 0.2540:
                    if percent_solid < 52 or percent_solid > 78:
                        raise ValueError("percent_solid outside of the valid empirical"+
                                         " range! When thickness is between 0.2032 and 0.2540," +
                                         " 52 <= percent_solid <= 78!")
                elif thickness >= 0.2540 and thickness < 0.3048:
                    if percent_solid < 48 or percent_solid > 78:
                        raise ValueError("percent_solid outside of the valid empirical"+
                                         " range! When thickness is between 0.2540 and 0.3048," +
                                         " 48 <= percent_solid <= 78!")
                        
        else:
            print("concrete_wall.CMS_heat_capacity_regression: Warning!" + 
                  " data bounds are not being checked. The underlying" +
                  " relationship is an empirical fit!")
        # empirical coefficients from fit to Table 11.1.5 data in [1].
        #  fit in test_ElCano_BuildingEnergy_Demand_Load_model.py test_heat_capacity
        x = np.zeros(20)
        x[0]     = -3.780755e+01
        x[1]     =  5.047281e-01
        x[2]     = -1.806909e+01
        x[3]     =  5.251188e-02
        x[4]     = -3.032686e-03
        x[5]     =  2.966671e+00
        x[6]     = -4.836498e-04
        x[7]     =  4.239181e+02
        x[8]     = -1.762487e-01
        x[9]     = -1.823540e-05
        x[10]    =  2.490824e-05
        x[11]    = -2.437617e-02
        x[12]    = -3.208220e-07
        x[13]    = -1.085491e+00
        x[14]    =  2.530963e-03
        x[15]    =  1.449476e-07
        x[16]    = -6.137019e+02
        x[17]    =  6.847028e-03
        x[18]    =  4.918505e-05
        x[19]    =  1.064036e-09
        
        # correct the units. The wrong conversion factor was used in the original fit.
        Btu_per_ft2F_to_J_per_m2K = 6309.206291
        Btu_per_hft2F_to_W_per_m2K = 5.68
        
        x = x * Btu_per_ft2F_to_J_per_m2K/Btu_per_hft2F_to_W_per_m2K
        x0 = percent_solid
        x1 = thickness
        x2 = concrete_density
        features = np.array([1, x0, x1, x2, x0**2, x0*x1, x0*x2, x1**2, x1*x2, x2**2, x0**3, x0**2*x1, x0**2*x2, x0*x1**2, x0*x1*x2, x0*x2**2, x1**3, x1**2*x2, x1*x2**2, x2**3])
        return np.dot(x,features)
    

class RCBuilding(object):
    """ 
    This class models as simple conceptual apartment unit of known floor area
    and number of floors. 
    
    Units = SI
    
    THESE ARE OUT OF ORDER NOW! TODO FIX THE INPUT!
    
    initialization variables
    
    building_name               : str - a name that is associated with the demand
                                    profile
    
    floor_area              : float > 0 building footprint area in m2
    
    floor_height            : float > 0 building floor height in m
    
    num_floors              : int > 0 : number of building floors
    
    person_per_floor        : int > 0 : number of persons per floor 
    
    building_aspect_ratio   : east west length of building / north-south length
    
    fraction_internal_walls : float > 0 : multiplier on outside wall thermal 
                              heat capcitance due
                              to internal walls (ussually < 1.0)
                              
    window_to_wall_fraction_north : 0.0 < float < 1.0: determines size of 
                                    windows on north wall as a relative fraction
                                    of the total north wall area
    window_to_wall_fraction_east  : 0.0 < float < 1.0: determines size of 
                                    windows on east wall as a relative fraction
                                    of the total east wall area
    window_to_wall_fraction_south : 0.0 < float < 1.0: determines size of 
                                    windows on south wall as a relative fraction
                                    of the total south wall area
    window_to_wall_fraction_west  : 0.0 < float < 1.0: determines size of 
                                    windows on west wall as a relative fraction
                                    of the total west wall area
                                    
    external_wall_and_roof_avg_thickness : float > 0.0
                                           Thickness of external walls and roof
                                           in m
    concrete_density : 2200.0 > float > 0.0 : density of concrete used in
                                              construction of Concrete masonry
                                              units (CMS) (concrete without air 
                                                           voids in CMS)
                                              
    concrete_aggregate_vol_frac = fraction of aggregate material. Only required if 
                                  concrete density > 1920 kg/m3. If a value is 
                                  supplied, then the concrete cubic mixture model
                                  is used
    concrete_aggregate_conductivity = aggregate material (e.g sand, stones, etc..),
                                 thermal conductivity in W/(m*K). This is only needed
                                 if density > 1920 kg/m3.
    concrete_paste_conductivity = Concrete paste thermal conductivity. Paste is the
                                 binding material between aggregate filler 
                                 ( only needed for > 1920 kg/m3)"],
    fraction_air_in_CMS_units : 1.0 >= float >= 0.0 : Amount of bulk air voids
                                in CMS units
    windows_U_factor : float > 0.0 : U-factor for windows in W/(m2*K)
    
    lighting_load_Wpm2 : float > 0.0 : Amount of lighting energy use per m2
                                       consumed when lights are on. (W/m2)
    
    fan_air_changes_per_hour : float >= 0.0 : Air changes per hour produced
                                              by plug load fans
    
    infiltration_air_changes_per_hour : float >= 0 : leakiness of the building
                                                     to external air when sealed
                                                     (ach)
                                                     
    weather_file : string : path and file name to the energy plus weather file
                            that will be applied to this model. Right now
                            only dry bulb temperature is used but other values
                            may eventually be used to do things like vary
                            infiltration with wind speed etc...
        
    occupant_schedule : np.array 1-D : a list of fractions indicating how many
                        of the occupants per floor are present. This is used
                        to calculate occupant heat gain. This schedule will be
                        applied cyclically if it is a 24 hour schedule, then
                        it will be applied every day and then repeated. Weekly
                        will be repeated weekly etc... Each entry is a time 
                        step number
                        
    appliance_schedule : np.array 1-D : same as occupant schedule but scaling
                       "heat_gain_appliances_Wpm2"
                      
    glass_solar_transmittance : (optional) 1.0 > float > 0.0 : Fraction of solar 
                                radiation that enters the building unimpeded
                                
    glass_light_transmittance : (optional) 1.0 > float > 0.0 : Fraction of 
                                internal room light that escapes from the 
                                room when it is incident on a window.
    
    additional_heat_capacity_besides_walls : (optional) float > 0.0 [J/m2]
                                added heat capacity from insulation, flooring
                                furniture etc... This model only accounts for
                                the heat capacity of the cement CMS walls
                                
    additional_R_insulation : (optional) : add insulation R factor (m2*K/W) in 
                            addition to the CMS walls R factor (calculated)
                            internally based on concrete density and thickness
                            and air void fraction.
                            
    lighting_control_lux_threshold : (optional) default = 300. Numbers of Lux 
                            (lumen/m2) below which lights are turned on in the
                            units.
                                
    heat_gain_per_person : (optional) default = 110.0 W/person (moderate activity)
    
    heat_gain_appliances : (optional) default = 10 W/m2 heat gain from appliances
                            in the building. (not including fan loads)
                                
    heat_gain_per_person : float > 0 : sum of sensible and latent heat per person:
                           moderate activeity is 110.0 W/person https://www.engineeringtoolbox.com/metabolic-heat-persons-d_706.html
    
    complex_appliances : dict :
        Dictionary with entries "fridges", "fans", "ac", "lights". Other complex
        appliances may be added in the future. Each dictionary key provides
        a list of types and numbers of the given complex appliances within the
        building.
    
    appliance_schedule : pandas.Series :
        An aggregate schedule of static electricity consumption in the building.
        Many different loads are added together for this.
        
    appliance_heat_schedule : pandas.Series :
        An aggregate schedule of heat created by appliances. Some appliances
        create mostly heat energy while others may be outside (such as an EV 
        charger) so that none of the heat is inside the building.                                                       
        
    R_per_area_ground : float > 0 :
        Resistance to ground temperature (m2*K/W) through the foundation of the
        building.
        
    elevation : float > 0:
        Elevation in meters above sea level of the building
        
    ground_temp_elev_sens : float :
        Sensitivity of ground temperature to elevation ground temperature reduces
        as elevation increases
        
    tier : str = {Non-critical, Tier 1, Tier 2, Tier 3}:
        Designates what tier the building is being simulated at.
        
    use_central_ac : bool
        Designates if central AC is used in the building. If it is, then
        wall ac units are not used.
        
    central_ac_avg_cop : float > 0
        Central AC average coefficient of performance (COP). There are no
        performance curves for the efficiency of the central ac. Values of 2-3 
        are typical with 3-5 indicative of highly efficient systems such as
        a ground source heat pump.
        
    
    The basic parameters of this analysis can be changed mid-step to simulate
    more complex scenarios. For example, window blinds can be simulated by
    changing the window to wall fraction based on feedback from the output
    of the last hourly time step.
    
    
    """
    air_heat_capacity = 1006 #J/(kg*K)
    air_density = 1.276 #kg/m
    hours_in_day = 24
    non_leap_num_day_in_year = 365
    
    
    def __init__(self, building_name,
                       floor_area,
                       floor_height,
                       num_floors,
                       person_per_floor,
                       building_aspect_ratio,
                       fraction_internal_walls,
                       window_to_wall_fraction_north,
                       window_to_wall_fraction_east,
                       window_to_wall_fraction_south,
                       window_to_wall_fraction_west,
                       external_wall_and_roof_avg_thickness,
                       concrete_density,
                       concrete_aggregate_vol_frac,
                       concrete_aggregate_conductivity,
                       concrete_paste_conductivity,
                       fraction_air_in_CMS_units,
                       windows_U_factor,
                       infiltration_air_changes_per_hour,
                       weather_file,
                       occupant_schedule,
                       building_latitude,
                       building_longitude,
                       simulation_year,
                       start_hour_of_year,
                       additional_R_insulation,
                       additional_heat_capacity_besides_walls,
                       glass_solar_transmittance,
                       glass_light_transmittance,
                       heat_gain_per_person, 
                       complex_appliances,
                       appliance_schedule,
                       appliance_heat_schedule,
                       R_per_area_ground,
                       elevation,
                       ground_temp_elev_sens,
                       tier,
                       use_central_ac,
                       central_ac_avg_cop):
        
        self.name = building_name
        # interpret these inputs into the inputs to the 5RC1 model.
        self.use_central_ac = use_central_ac
        self.central_ac_avg_cop = central_ac_avg_cop
        self.total_area = num_floors * floor_area
        self.floor_area = floor_area
        self.num_floors = num_floors
        short_length = np.sqrt(floor_area/building_aspect_ratio)
        long_length = floor_area/short_length
        
        small_wall_area = short_length * num_floors * floor_height
        large_wall_area = long_length * num_floors * floor_height
        
        # calculate window areas
        if building_aspect_ratio > 1:
            south_window_area = small_wall_area * window_to_wall_fraction_south
            north_window_area = small_wall_area * window_to_wall_fraction_north
            east_window_area = large_wall_area * window_to_wall_fraction_east
            west_window_area = large_wall_area * window_to_wall_fraction_west
        else:
            south_window_area = large_wall_area * window_to_wall_fraction_south
            north_window_area = large_wall_area * window_to_wall_fraction_north
            east_window_area = small_wall_area * window_to_wall_fraction_east
            west_window_area = small_wall_area * window_to_wall_fraction_west
        
        # sum total areas and volume
        total_window_area = (south_window_area + north_window_area + 
                             east_window_area + west_window_area)
        total_wall_area = 2*small_wall_area + 2*large_wall_area + floor_area  # assume roof is same as walls
        total_volume = floor_area * num_floors * floor_height
        
        # create windows objects to calculate solar gains into each wall
        # For azimuth angle 0 degrees = South < 0 is east and > 0 is west
        windows = {}
        windows["S"] = Window(azimuth_tilt=0, 
                              alititude_tilt=90, 
                              glass_solar_transmittance=glass_solar_transmittance,
                              glass_light_transmittance=glass_light_transmittance, 
                              area=south_window_area)  
        windows["W"] = Window(azimuth_tilt=90, 
                              alititude_tilt=90, 
                              glass_solar_transmittance=glass_solar_transmittance,
                              glass_light_transmittance=glass_light_transmittance, 
                              area=west_window_area) 
        windows["N"] = Window(azimuth_tilt=180, 
                              alititude_tilt=90, 
                              glass_solar_transmittance=glass_solar_transmittance,
                              glass_light_transmittance=glass_light_transmittance, 
                              area=north_window_area)
        windows["E"] = Window(azimuth_tilt=-90, 
                              alititude_tilt=90, 
                              glass_solar_transmittance=glass_solar_transmittance,
                              glass_light_transmittance=glass_light_transmittance, 
                              area=east_window_area)
        
        # quantify the heat capacity of the structure
        if (np.isnan(concrete_aggregate_vol_frac) or
            np.isnan(concrete_aggregate_conductivity) or
            np.isnan(concrete_paste_conductivity)):
            
            wall1 = concrete_wall(concrete_density,
                                  fraction_air_in_CMS_units, 
                                  external_wall_and_roof_avg_thickness,
                                  additional_R_insulation)
        
        else:
            wall1 = concrete_wall(concrete_density,
                                  fraction_air_in_CMS_units, 
                                  external_wall_and_roof_avg_thickness,
                                  additional_R_insulation,
                                  Va=concrete_aggregate_vol_frac,
                                  ka=concrete_aggregate_conductivity,
                                  kp=concrete_paste_conductivity)
            
        
        total_heat_capacity_per_floor_area = (total_volume * self.air_density * self.air_heat_capacity +
                                      (1 + fraction_internal_walls) * total_wall_area * wall1.HC_value + 
                                      additional_heat_capacity_besides_walls)/floor_area
        
        self.windows = windows
        self.ach_infiltration = infiltration_air_changes_per_hour
        self.building = Zone(window_area=total_window_area,
              walls_area=total_wall_area,
              floor_area=self.total_area,
              room_vol=total_volume,
              total_internal_area=total_wall_area * fraction_internal_walls,
              lighting_load=1.0,  # This changes based on the lights class
              lighting_control=300.0, # this changes based on the Lights class
              lighting_utilisation_factor=0.45, # TODO add to lights class
              lighting_maintenance_factor=0.9, # TODO add to lights class
              u_walls=wall1.U_value,
              u_windows=windows_U_factor,
              ach_vent=0.0,  # no central HVAC system. Opening a window is the best that can be done
              ach_infl=infiltration_air_changes_per_hour,
              ventilation_efficiency=0.0, # no heat recovery
              thermal_capacitance_per_floor_area=total_heat_capacity_per_floor_area,
              t_set_heating=complex_appliances["ac_thermostat_heat"][tier].values[0], # these are intended to keep the air-conditioning off.
              t_set_cooling=complex_appliances["ac_thermostat_cool"][tier].values[0], # these are intended to keep the air-conditioning off.
              max_cooling_energy_per_floor_area=-1e20, # unlimited central cooling if use_central_ac = True.
              #Cooling is negative heat flow! We use the building simulator only for its crank-nicolson equations. We set the 
                                                      # energy demand based on our own calculations
              max_heating_energy_per_floor_area=1e20, # unlimited central heating if use_central_ac = True.
              heating_supply_system=supply_system.DirectHeater,
              cooling_supply_system=supply_system.DirectCooler,
              heating_emission_system=emission_system.NewRadiators,
              cooling_emission_system=emission_system.AirConditioning)
        
        # get the weather data needed and add dates to everything based on the year.
        self.year = simulation_year
        self.Location = Location(epwfile_path=os.path.join(weather_file))
        if self._is_leap_year(self.year):
            num_day = self.non_leap_num_day_in_year + 1
        else:
            num_day = self.non_leap_num_day_in_year
        num_hour = num_day * self.hours_in_day
        wd = self.Location.weather_data["drybulb_C"]
        new_index = pd.date_range("1-1-"+str(self.year),"1-1-"+str(self.year+1),num_hour+1)[:-1]
        
        wd.index = new_index[0:len(wd)] # just in case this is TMY3 data with no leap day.
        
        self.Location.weather_data["actual date"] = new_index[0:len(wd)]
        
        self.building.t_air = self.Location.weather_data['drybulb_C'].iloc[0]
        self.building.energy_demand_unrestricted = 0.0 # using the RC building simulator to calculate 
                                                       # cooling and heating loads.
        self.latitude = building_latitude
        self.longitude = building_longitude
        self.current_hour = start_hour_of_year
        self.max_occupants = person_per_floor * num_floors
        self.occupant_schedule = occupant_schedule     # 
        self.len_occupant_sch = len(occupant_schedule)
        self.appliance_schedule = appliance_schedule   # 
        self.len_appliance_sch = len(appliance_schedule)
        self.appliance_heat_schedule = appliance_heat_schedule
        # TODO find a model of sensible and latent heat gain as a function of 
        #      environment. We add 0 sensible heat past 97F and transition to
        #      only latent heat.
        self.heat_gain_per_person = heat_gain_per_person
        
        # temperature initial condition
        self.T_prev = self.Location.weather_data['drybulb_C'].iloc[0]
        # assume the room is initially as humid as the outside
        self.internal_rh = self.Location.weather_data['relhum_percent'].iloc[0]/100
        
        self._initialize_results()
            
        self.humidity_model = humidity_model()
        
        # complex appliances
        self.complex_appliances = complex_appliances
        
        # ground temperatures
        self.Rground = R_per_area_ground / floor_area
        self.elevation = elevation
        self.ground_temp_elev_sens = ground_temp_elev_sens
        self._calculate_monthly_ground_temperatures(elevation,wd)
        self.warning_messages = []
        
    def _calculate_monthly_ground_temperatures(self,elevation,wd):
        # see Lugo-Camacho et. al., 2009 "Soil temperature studyin Puerto Rico"
        avg_air_temperatures = wd.resample("ME").mean()
        self.ground_temperature = avg_air_temperatures - self.ground_temp_elev_sens * elevation
        self.month_at_hour = wd.index.month

    
    def _is_leap_year(self,year):
        return np.mod(year,4) == 0 and (np.mod(year,100) != 0 or np.mod(year,400) == 0)

    def _initialize_results(self):
        # initialize results lists.
        self.ForTroubleshooting = {}
        self.ForTroubleshooting["HeatingDemand"] = []
        self.ForTroubleshooting["HeatingEnergy"] = []
        self.ForTroubleshooting["CoolingDemand"] = []
        self.ForTroubleshooting["CoolingEnergy"] = []
        self.ForTroubleshooting["COP"] = []
        self.Results = {}
        self.Results["IndoorAirTemp"] = []
        self.Results["SolarGains"] = []
        self.Results["StructureTemp"] = []
        self.Results["IndoorSurfaceTemp"] = []
        self.Results["OutsideAirTemp"] = []
        self.Results["PlugInFans"] = []
        self.Results["StaticElectricLoads"] = []
        self.Results["Refrigerators"] = []
        self.Results["Wall_ACs"] = []
        self.Results["Lights"] = []
        self.Results["TotalElectricity"] = []
        self.Results["IndoorAirRelativeHumidity"] = []
        self.Results["TotalElectricity"] = []
        self.Results["TotalElectricity"] = []
        self.Results["UnmetCooling"] = []
        self.Results["OutdoorAirRelativeHumidity"] = []
        self.Results["Occupants"] = []
        self.Results["UnmetHeating"] = []
        self.Results["HeatLoadToMeetThermostat"] = []
        self.Results["Central_AC"] = []
        self.Results["Month"] = []
        self.Results["DayOfWeek"] = []
        self.Results["DayOfMonth"] = []
        self.Results["HourOfDay"] = []
        self.Results["MasterBuilding"] = []
        self.Results["BuildingArea"] = []
        self.Results["Date"] = []
        
    def run_model(self,start_hour,stop_hour,troubleshoot,master_building_name):

        # reset the time marker from previous runs
        self.master_building_name = master_building_name
        self.current_hour = start_hour
        self.T_prev = self.Location.weather_data['drybulb_C'].iloc[start_hour]
        self._preliminary_model_checks()
        for hr in np.arange(start_hour,stop_hour):
            self._time_step(troubleshoot)       
        
    #def _human_comfort_index():
    #    Tc = self.Location.weather_data['drybulb_C'].iloc[ts]
    #    # see [2] TODO - replace this with ASHRAE 55 thermal comfort envelope.
    #    return 1.8 * Tc - 0.55 * (1.8 * Tc - 26.0) * (1 - RH) + 9.2 *(9+10.9*np.sqrt(vw) - vw) + 32.0
      
    def _preliminary_model_checks(self):
        if self.use_central_ac and self.complex_appliances['wall_acs']:
            self.warning_messages.append("Input Warning: The building has central AC but" +
                                         " also has wall AC's specified." +
                                         " The wall AC's will not be used" +
                                         " because central AC always meets" +
                                         " the cooling load!")
        # add other model checks and associated warnings here!
    
    def _time_step(self,troubleshoot):
        # Occupancy for the time step
        # Gains from occupancy and appliances
        
        # set local time step for occupant and appliance schedules
        ts = self.current_hour
        hour_of_day = np.mod(ts,self.hours_in_day)
        ts_occ = np.mod(ts,self.len_occupant_sch)
        ts_app = np.mod(ts,self.len_appliance_sch)
        
        # static heat gains inside the structure
        occupancy = self.max_occupants * self.occupant_schedule[ts_occ]
        internal_gains = (occupancy * self.heat_gain_per_person + 
            self.total_area * self.appliance_heat_schedule[ts_app])
    
        # Extract the outdoor temperature in for that hour

        T_out = self.Location.weather_data['drybulb_C'].iloc[ts]
        P_out = self.Location.weather_data['atmos_Pa'].iloc[ts]

        T_prev = self.T_prev
    
        ## TODO - there are all kinds of occupant behaviors that may be driven
        #         by T_prev and T_out for non-conditioned spaces. You eventually
        #         want to make this into a model that allows the formation of
        #         rules and code concerning occupant behavior under normal 
        #         and abnormal conditions.
    
        Altitude, Azimuth = self.Location.calc_sun_position(
            latitude_deg=self.latitude, longitude_deg=self.longitude, 
            year=self.year, hoy=ts)
        
        solar_gains = 0.0
        transmitted_illuminance = 0.0
        for key,window in self.windows.items():
            window.calc_solar_gains(sun_altitude=Altitude, sun_azimuth=Azimuth,
                                     normal_direct_radiation=self.Location.weather_data['dirnorrad_Whm2'].iloc[ts],
                                     horizontal_diffuse_radiation=self.Location.weather_data['difhorrad_Whm2'].iloc[ts])
            window.calc_illuminance(sun_altitude=Altitude, sun_azimuth=Azimuth,
                                     normal_direct_illuminance=self.Location.weather_data[
                                         'dirnorillum_lux'].iloc[ts],
                                     horizontal_diffuse_illuminance=self.Location.weather_data['difhorillum_lux'].iloc[ts])
            solar_gains += window.solar_gains
            transmitted_illuminance += window.transmitted_illuminance
        
        # 2 x provides convergence on changing conditions w/r to predicting what cooling loads should be for
        #     wall AC units.
        if self.use_central_ac:
            repeat_num = 1
        else:
            repeat_num = 2
            
        for i in range(repeat_num):
            
            # ac units, refrigerators, fans and lights
            (fridge_Qnet, fridge_power, wall_ac_power, 
             wall_ac_sensible_heat, wall_ac_mdot_condensed, 
             fan_power, fan_heat, fan_ACH,
             light_heat, light_power,unmet_cooling,unmet_heating) = self._complex_appliances(self.building.t_air,
                                                                      hour_of_day,
                                                                      self.current_hour,
                                                                      P_out,
                                                                      internal_gains,
                                                                      transmitted_illuminance,
                                                                      occupancy,1,T_out,self.Location.weather_data[
                                         'dirnorillum_lux'].iloc[ts])
            
            if troubleshoot:
                self._print_troubleshooting(fridge_Qnet, fridge_power, wall_ac_power, 
                                            wall_ac_sensible_heat, wall_ac_mdot_condensed, 
                                            fan_power, fan_heat, fan_ACH,
                                            light_heat, light_power)
    
            #TODO add latent heat rate for human's inside the house which adds moisture        
            external_rh = self.Location.weather_data['relhum_percent'].iloc[ts]/100
            internal_rh_new = self.humidity_model._humidity_model(wall_ac_mdot_condensed, 
                                             self.ach_infiltration,
                                             P_out,
                                             self.internal_rh,
                                             external_rh,
                                             self.building.t_air,
                                             T_out,
                                             1,
                                             self.building.room_vol)  # TODO is the time step 1 hour the correct unit?
            
            self.internal_rh = internal_rh_new
            
            
            # Air changes per hour (ACH) effects
            #THESE LINES ARE FROM building_physics:Zone:__init__. You must adjust
            # self.h_ve_adj because our fans can alter the air changes per hour.
            ach_tot = fan_ACH + self.ach_infiltration#ach_infl + ach_vent  # Total Air Changes Per Hour
            # temperature adjustment factor taking ventilation and infiltration
            # [ISO: E -27]
            ach_vent = 0.0
            ventilation_efficiency = 0.0
            
            b_ek = (1 - (ach_vent / (ach_tot)) * ventilation_efficiency)
            self.building.h_ve_adj = 1200 * b_ek * self.building.room_vol * \
                (ach_tot / 3600)  # Conductance through ventilation [W/M]
            
            
            # ground heat transfer model.
            current_month = self.month_at_hour[self.current_hour]
            ground_temp = self.ground_temperature.iloc[current_month-1]

            Qground = (T_prev - ground_temp)/self.Rground
            
            # This really doesn't do anyting w/r to energy use. It does create 
            # the thermal balances though. TODO - verify that each thermal load is 
            # correctly oriented.
            all_internal_heat = (internal_gains + 
                                 fridge_Qnet + light_heat + fan_heat - Qground)

            if self.use_central_ac:
                # NO WALL UNITS SIMULATED IF CENTRAL AC BEING USED!
                self.building.solve_energy(internal_gains=all_internal_heat,
                                    solar_gains=solar_gains,
                                    t_out=T_out,
                                    t_m_prev=T_prev)
                
                # We assume that energy demand is always met!
                central_ac_power =  (np.abs(self.building.energy_demand_unrestricted) /
                                     self.central_ac_avg_cop)
                
            else:
                self.building.has_cooling_demand = True
                self.building.calc_energy_demand(all_internal_heat,solar_gains,T_out,T_prev)
            
                if unmet_cooling > 0:
                    print("What is going on?")
                
                t_m,t_air,t_operative = self.building.calc_temperatures_crank_nicolson(-wall_ac_sensible_heat,
                                                               all_internal_heat,
                                                               solar_gains,T_out,T_prev)
                central_ac_power = 0.0

    
            # Set the previous temperature for the next time step
            # TODO - check that this is the air (or structure?? ) temperature.
            self.T_prev = self.building.t_m_next
        # increment the time step only 1 hour time steps supported for now.
        self._append_results(fan_power,fridge_power,wall_ac_power,T_out,
                             solar_gains,ts_app,light_power,unmet_cooling,
                             external_rh,unmet_heating,occupancy,central_ac_power)
        self.current_hour += 1
    
    def _print_troubleshooting(self,fridge_Qnet, fridge_power, wall_ac_power, 
                                        wall_ac_sensible_heat, wall_ac_mdot_condensed, 
                                        fan_power, fan_heat, fan_ACH,
                                        light_heat, light_power):
        print("\n\n")
        print("Hour: {0:5d}".format(self.current_hour))
        print("\n\n")
        print("fridge_Qnet:{0:5.3e}".format(fridge_Qnet))
        print("fridge_power:{0:5.3e}".format(fridge_power)) 
        print("wall_ac_power:{0:5.3e}".format(wall_ac_power)) 
        print("wall_ac_sensible_heat:{0:5.3e}".format(wall_ac_sensible_heat)) 
        print("wall_ac_mdot_condensed:{0:5.3e}".format(wall_ac_mdot_condensed)) 
        print("fan_power:{0:5.3e}".format(fan_power))
        print("fan_heat:{0:5.3e}".format(fan_heat)) 
        print("fan_ACH:{0:5.3e}".format(fan_ACH))
        print("light_power:{0:5.3e}".format(light_power))
        if len(self.ForTroubleshooting["HeatingDemand"])>1:
            print("Heating Demand:building simulator: {0:5.3e}".format(self.ForTroubleshooting["HeatingDemand"][-1]))
            print("Cooling Demand:building simulator: {0:5.3e}".format(self.ForTroubleshooting["CoolingDemand"][-1]))
            print("Cooling Energy:building simulator: {0:5.3e}".format(self.ForTroubleshooting["CoolingEnergy"][-1]))
            
            print("IndoorAirTemp:{0:5.3e}".format(self.Results["IndoorAirTemp"][-1]))
            print("StructureTemp:{0:5.3e}".format(self.Results["StructureTemp"][-1]))
            print("IndoorSurfaceTemp:{0:5.3e}".format(self.Results["IndoorSurfaceTemp"][-1]))       
            print("IndoorAirRelativeHumidity:{0:5.3e}".format(self.Results["IndoorAirRelativeHumidity"][-1]))
            print("OutsideAirTemp:{0:5.3e}".format(self.Results["OutsideAirTemp"][-1]))
            print("SolarGains:{0:5.3e}".format(self.Results["SolarGains"][-1]))
        
        

        
    def _append_results(self, fan_power,fridge_power,wall_ac_power,T_out,
                             solar_gains,ts_app,light_power,unmet_cooling,
                             external_rh,unmet_heating,occupancy,
                             central_ac_power):
        # These should all be zero for a non-air conditioned space.
        #self.ForTroubleshooting["HeatingDemand"].append(self.building.heating_demand)
        #self.ForTroubleshooting["HeatingEnergy"].append(self.building.heating_energy)
#        self.ForTroubleshooting["CoolingDemand"].append(self.building.cooling_demand)
#        self.ForTroubleshooting["CoolingEnergy"].append(self.building.cooling_energy)
#        self.ForTroubleshooting["COP"].append(self.building.cop)
        # Results
        static_loads_power = self.total_area * self.appliance_schedule[ts_app]
        
        self.Results["PlugInFans"].append(fan_power)
        self.Results["StaticElectricLoads"].append(static_loads_power)
        self.Results["Refrigerators"].append(fridge_power)
        self.Results["Wall_ACs"].append(wall_ac_power)
        self.Results["Central_AC"].append(central_ac_power)
        self.Results["Lights"].append(light_power)
        # TODO - ALL Of this depends on a 1 hour time step so that these go from W to W*hr
        self.Results["TotalElectricity"].append(static_loads_power + 
                                              fan_power + fridge_power + wall_ac_power + light_power + central_ac_power)
        self.Results["IndoorAirTemp"].append(self.building.t_air)
        self.Results["StructureTemp"].append(self.building.t_m_next)
        self.Results["IndoorSurfaceTemp"].append(self.building.t_s)
        self.Results["IndoorAirRelativeHumidity"].append(self.internal_rh)
        self.Results["OutdoorAirRelativeHumidity"].append(external_rh)
        self.Results["OutsideAirTemp"].append(T_out)
        self.Results["SolarGains"].append(solar_gains)
        self.Results["UnmetCooling"].append(unmet_cooling)
        self.Results["UnmetHeating"].append(unmet_heating)
        self.Results["Occupants"].append(occupancy)
        self.Results["HeatLoadToMeetThermostat"].append(self.building.energy_demand_unrestricted)

        month = self.Location.weather_data["actual date"][self.current_hour].month
        day = self.Location.weather_data["actual date"][self.current_hour].day
        year = self.Location.weather_data["actual date"][self.current_hour].year
        
        date = datetime.datetime(year, month, day, np.mod(self.current_hour, 24))
        self.Results["Date"].append(date)
        weekday = date.weekday()
        self.Results["Month"].append(month)
        self.Results["DayOfWeek"].append(weekday)
        self.Results["DayOfMonth"].append(day)
        # assumes a simulation starts 
        self.Results["HourOfDay"].append(np.mod(self.current_hour, 24)+1)
        self.Results["MasterBuilding"].append(self.master_building_name) # This establishes what buildings are different use cases of the same
                                                                   # underlying structure.
        self.Results["BuildingArea"].append(self.total_area)
        
    def _complex_appliances(self,T_air_in,hour_of_day,hour_of_year, Pressure, static_heat_loads,
                            transmitted_illuminance,occupancy,ts,T_out,normal_direct_illuminance):
        """
        
        Any appliances with more complicated thermal models with thermodynamic
        loops that have feedback between room temperature and thermal response
        
        inputs:
            T_air_in : float : previous time step's air temperature in the 
                               building
            
            hour_of_day : int : 0 - 23 
            
            ts : int : time step 0 - 8759 (depends on weather history length) 
            
            Pressure : float : Atmospheric pressure in (Pa)
        
        """
        # TODO - all of this can be made more abstract and less cumbersome!
        cap = self.complex_appliances

        # first all appliances that add/subtract heat to the space need to be
        # analyzed before AC.
        
        fan_power = 0
        fan_heat = 0
        fan_ACH = 0
        if len(cap['fans']) != 0:
            for key,fantup in cap['fans'].items():
                fanobj = fantup[1]
                numfan = fantup[0]
                if type(key) is str:
                    Power, ACH, Heat = fanobj.fan_energy_model(T_air_in, T_out)
                    fan_power += numfan * Power
                    fan_heat += numfan * Heat
                    fan_ACH += numfan * ACH
        
        fridge_power = 0
        fridge_Qnet = 0
        
        if len(cap['fridges']) != 0:
            for key,fridgetup in cap['fridges'].items():
                fridge = fridgetup[1]
                numfridge = fridgetup[0]
                #TODO - you need to verify the thermal direction of the heat loads
                if type(key) is str:
                    Qnet, Pnet = fridge.avg_thermal_loads(T_air_in,hour_of_day,ts)
                    fridge_Qnet += numfridge * Qnet
                    fridge_power += numfridge * Pnet
        
        # lights - right now only the interior lights control the lighting threshold. 
        # Only one threshold is analyzed for now even though every lighting type
        # is given a threshold
        total_light_demand = 0
        light_heat = 0
        if len(cap['lights']) != 0:
            for key, lighttup in cap['lights'].items():
                lights = lighttup[1]
                numlight = lighttup[0]
                
                if type(key) is str:
                    light_demand = numlight * lights.power
                    self.building.lighting_load = light_demand / self.total_area 
                    
                    if lights.light_type == "Exterior":
                        illuminance = normal_direct_illuminance
                    else:
                        illuminance = transmitted_illuminance
                    self.building.lighting_control = lights.lux_threshold
                    self.building.solve_lighting(illuminance=illuminance, occupancy=occupancy)
                    
                    # now reassign the demand depending on whether the lights are actually on.
                    total_light_demand += self.building.lighting_demand
                    if lights.light_type == "Interior":    
                        light_heat += lights.power * numlight * lights.fraction_heat                     
                    elif lights.light_type != "Exterior":
                        raise ValueError("An unknown light type was found: " +lights.light_type + ". Only " +
                                         str(["Interior","Exterior"]) + " are allowed!")
                else:
                    raise Exception("There is an error in the lighting complex appliance. The key is not a string!")
        
            
        wall_ac_power = 0
        wall_ac_sensible_heat = 0
        wall_ac_mdot_condensed = 0
        
        
        
        #This comes direct from building_physics.py for the exact cooling load
        # needed to meet the setpoint.
        # only run wall_ac's if central cooling is not used.
        if self.use_central_ac:
            # all loads are met!
            unmet_cooling_0 = 0.0
            unmet_heating_0 = 0.0
        else:
            unmet_cooling_0 = self.building.energy_demand_unrestricted
            
            if unmet_cooling_0 > 0.0 or len(cap["wall_acs"]) == 0: # remember that cooling energy is negative for building_physics
                if unmet_cooling_0 > 0.0:
                    unmet_heating_0 = unmet_cooling_0
                    unmet_cooling_0 = 0.0
                else:
                    unmet_heating_0 = 0.0
                
            else:      
                unmet_heating_0 = 0.0
                for key,ACunit_tup in cap["wall_acs"].items():
    
                    #TODO - What is the temperature of condensed water, we assume for now that all sensible heat from
                    #       the coil temperature is reabsorbed up to room temperature until
                    ACunit = ACunit_tup[1]
                    num_AC = ACunit_tup[0]
                    if type(key) is str:
                        TC, SC, mdot_condensed, power, unmet_cooling_1 = ACunit.avg_thermal_loads(
                            DBT_internal=T_air_in,
                            rh_internal=self.internal_rh,
                            DBT_external=self.Location.weather_data['drybulb_C'].iloc[ts],
                            Pressure = Pressure,
                            num_AC = num_AC,
                            volume = self.building.room_vol,
                            time_step = 1,
                            energy_demand_unrestricted=unmet_cooling_0) # TODO - create a flexible time step
                        # do not multiply by num_AC it is done in thermal_control
                        unmet_cooling_0 = unmet_cooling_1 # only use the air-conditioners needed to meet the load.
                        wall_ac_power += power
                        wall_ac_sensible_heat += SC  # SC is positive because it is taken away from the equation later.
                        wall_ac_mdot_condensed += mdot_condensed
                        
                        if unmet_cooling_0 > 0.0:
                            break
                
        
                    
        return (fridge_Qnet, fridge_power, wall_ac_power, 
                wall_ac_sensible_heat, wall_ac_mdot_condensed, 
                fan_power, fan_heat, fan_ACH,
                light_heat, total_light_demand, unmet_cooling_0,unmet_heating_0)

    
class ReadInputSpreadSheet(object):
    
    """
    This class holds ALL of the information needed to run an entire tiered loads
    building energy model analysis. 
    
    Further processing is accomplished in TieredAnalysis
    
    
    """
    
    def __init__(self,tiered_load_spreadsheet_path):
        if os.path.isfile(tiered_load_spreadsheet_path):
            dat = pd.read_excel(tiered_load_spreadsheet_path,sheet_name=None)
        else:
            raise ValueError("The file: " + tiered_load_spreadsheet_path + "\n\n does not exist!")
        # schedules
        self.inputs = dat
        self.schedules = dat['Schedules']
                    # grab basic building parameters and profiles
        df_buildings = DF_ops.df_col_index_and_drop(dat,"Buildings","Description")
        
        self.buildings = df_buildings

    
        
     
        

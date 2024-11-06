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

----------------------END OF LICENSE STATEMENT -----------------------------

Created on Fri Oct  9 07:54:43 2020

@author: dlvilla
ec """
if __name__ == "__main__":
    import TEB.simulator.sim as ec_be
    from TEB.simulator.complex_appliances import Wall_AC_Unit
    from TEB.simulator.thermodynamics import thermodynamic_properties as tp
    tp = tp()
import numpy as np
import unittest
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

import os

class test_ElCano_BuildingEnergy_Demand_Load_Model(unittest.TestCase):
    @classmethod 
    def setUpClass(cls):
        cls.lbpft3_to_kgpm3 = 16.0185
        cls.Btuphft2Fpin_toWpmK = 0.1441314338
        cls.Btu_per_hft2F_to_W_per_m2K = 5.68
        cls.Btu_per_ft2F_to_J_per_m2K = 6309.206291
        cls.in_to_m = 0.0254
        cls.include_plots = False
        plt.close('all')
    
    def test_U_value(self):
        dc_IMP = np.array([20,30,40,50,60,70,80,90,100,110,120,130,140,150]) #lb/ft3
        dc = dc_IMP * self.lbpft3_to_kgpm3
        kc_table11_1_2_IMP = np.array([[0.8,1.1,1.4,1.7,2.1,2.5,3.0,3.5,4.1,4.7,5.4,np.nan,np.nan,np.nan],
                                    [0.7,1.0,1.3,1.6,2.0,2.5,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                                    [0.8,1.1,1.5,1.9,2.4,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
                                    [np.nan,np.nan,np.nan,1.7,2.4,2.7,3.0,3.6,4.9,5.0,6.4,np.nan,np.nan,np.nan],
                                    [np.nan,np.nan,np.nan,1.8,2.6,3.0,3.2,3.8,5.3,5.4,6.8,np.nan,np.nan,np.nan],
                                    [np.nan,np.nan,np.nan,1.9,2.5,3.2,4.1,5.1,6.2,7.6,9.1,np.nan,np.nan,np.nan],
                                    [np.nan,np.nan,np.nan,2.1,2.7,3.5,4.4,5.5,6.8,8.2,9.9,np.nan,np.nan,np.nan],
                                    [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,5.5,6.6,7.9,9.4,11.1,13.8],
                                    [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,5.85,7.0,8.3,10.0,11.7,13.75],
                                    [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,10.0,13.8,18.5],
                                    [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,10.7,14.6,19.6],
                                    [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,11.0,15.3,20.5],
                                    [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,11.8,16.5,22.0],
                                    [np.nan,np.nan,np.nan,np.nan,2.8,3.6,4.5,5.5,6.7,8.1,9.7,11.5,13.5,np.nan],
                                    [np.nan,np.nan,np.nan,np.nan,3.1,3.9,4.8,6.0,7.3,8.7,10.5,12.4,17.7,np.nan],
                                    [np.nan,np.nan,np.nan,np.nan,np.nan,2.5,3.0,3.6,4.2,4.9,5.6,6.4,7.4,8.4],
                                    [np.nan,np.nan,np.nan,np.nan,np.nan,3.1,3.7,4.3,5.1,5.9,6.8,7.8,9.0,10.2]]) #Btu/(h*ft2*(F/in))
        kc_table11_1_2 = kc_table11_1_2_IMP * self.Btuphft2Fpin_toWpmK
        copyright_notice = "from [1] Reproduced by permission of IMI from 08/87 report, 'Thermophysical Properties of Masonry and Its Constituents.'"
        column_name = pd.MultiIndex.from_tuples([("Matrix Insul.","Neat cement paste","Protected"),
                                                ("Insul Struct","Autoclaved aerated (cellular)","Protected"),
                                                ("Insul","Expanded polystyrene beads, perlite, vermiculite","Protected"),
                                                ("Blocks Struct.","ASTM C 330 aggregates","Protected"),
                                                ("Blocks Struct.","ASTM C 330 aggregates","UnProtected"),
                                                ("Blocks Struct.","ASTM C 330 L W aggregates with ASTM C 33 sand","Protected"),
                                                ("Blocks Struct.","ASTM C 330 L W aggregates with ASTM C 33 sand","UnProtected"),
                                                ("Blocks Struct.","Limestone","Protected"),
                                                ("Blocks Struct.","Limestone","UnProtected"),
                                                ("Blocks Struct.","Sand gravel < 50% quartz or quartzite","Protected"),
                                                ("Blocks Struct.","Sand gravel < 50% quartz or quartzite","UnProtected"),
                                                ("Blocks Struct.","Sand gravel > 50% quartz or quartzite","Protected"),
                                                ("Blocks Struct.","Sand gravel > 50% quartz or quartzite","UnProtected"),
                                                ("Insul. Struct. Masonry","Cement-sand mortar","Protected"),
                                                ("Insul. Struct. Masonry","Cement-sand mortar","UnProtected"),
                                                ("Insul. Struct. Masonry","Foam concrete solid clay bricks","Protected"),
                                                ("Insul. Struct. Masonry","Foam concrete solid clay bricks","UnProtected")])
        
        concrete_conductivity_df = pd.DataFrame(data=kc_table11_1_2.T, index=dc, columns=column_name )
        if self.include_plots:
            fig,ax = plt.subplots(1,1,figsize=(20,10))
            concrete_conductivity_df.plot(ax=ax,ls="",marker='.',ms=15)
            ax.set_xlabel("density (kg/m3)")
            ax.set_ylabel("thermal conductivity (W/(m*K))")
        Vair = 0.0
        L = 0.1
        
        V_air = [0,0.1,0.2,0.4,0.5,0.6,0.7]
        U_values = np.zeros((len(dc),len(V_air)))
        k_equivalent = U_values
        
        ka = 1
        kp = 4.0
        Va = 0.8
        
        for idx,val_dc in enumerate(dc):
            for idy, Vair in enumerate(V_air):
                if val_dc < 1920:
                    wall1 = ec_be.concrete_wall(val_dc,Vair,L,0.0,calc_HC=False)
                else:
                    wall1 = ec_be.concrete_wall(val_dc,Vair,L,0.0,Va,ka,kp,calc_HC=False)
                U_values[idx,idy] = wall1.U_value
                k_equivalent[idx,idy] = wall1.k_equivalent
        for idy, Vair in enumerate(V_air):
            if self.include_plots:
                ax.plot(dc,k_equivalent[:,idy],linestyle="--",label="model Vair={0:5.1f}".format(Vair))
        if self.include_plots:    
            ax.legend()
            
        
        
    def test_heat_capacity(self):
        # Size of CMU and
        # % solid
        # Density of concrete in CMU, lb/ft³*
        # heat capacity data
        
        # The 6.9 point on row 5 is bad data but I am leaving it here because
        # I eliminate it by deleting it. I want the original data from Table 
        hc_data = {"heat_capacity_IMP" : [3.40, 3.78, 4.17, 4.55, 4.93, 5.56, 5.96,
                                      4.01, 4.47, 4.94, 5.40, 5.86, 6.60, 7.08,
                                      5.05, 5.64, 6.23, 6.82, 7.41, 8.37, 8.99,
                                      4.36, 4.87, 5.37, 5.87, 6.38, 7.19, 7.72,
                                      6.04, 6.76, 7.47, 8.18, 6.90, 10.05, 10.80,
                                      5.57, 6.23, 6.88, 7.52, 8.17, 9.21, 9.89,
                                      8.17, 9.14, 10.11, 11.08, 12.04, 13.61, 14.63,
                                      6.50, 7.25, 8.01, 8.76, 9.51, 10.60, 11.38,
                                      10.26, 11.48, 12.71, 13.93, 15.15, 17.13, 18.41,
                                      7.75, 8.66, 9.57, 10.48, 11.39, 12.86, 13.81,
                                      12.30, 13.77, 15.25, 16.37, 18.20, 20.59, 22.14],
                    "percent_solid" : [65,65,65,65,65,65,65,
                                      78,78,78,78,78,78,78,
                                      100,100,100,100,100,100,100,
                                      55,55,55,55,55,55,55,
                                      78,78,78,78,78,78,78,
                                      52,52,52,52,52,52,52,
                                      78,78,78,78,78,78,78,
                                      48,48,48,48,48,48,48,
                                      78,78,78,78,78,78,78,
                                      48,48,48,48,48,48,48,
                                      78,78,78,78,78,78,78],
                    "thickness_in" : [4,4,4,4,4,4,4,
                                  4,4,4,4,4,4,4,
                                  4,4,4,4,4,4,4,
                                  6,6,6,6,6,6,6,
                                  6,6,6,6,6,6,6,
                                  8,8,8,8,8,8,8,
                                  8,8,8,8,8,8,8,
                                  10,10,10,10,10,10,10,
                                  10,10,10,10,10,10,10,
                                  12,12,12,12,12,12,12,
                                  12,12,12,12,12,12,12],
                    "density_IMP" : [80,90,100,110,120,130,140,
                                80,90,100,110,120,130,140,
                                80,90,100,110,120,130,140,
                                80,90,100,110,120,130,140,
                                80,90,100,110,120,130,140,
                                80,90,100,110,120,130,140,
                                80,90,100,110,120,130,140,
                                80,90,100,110,120,130,140,
                                80,90,100,110,120,130,140,
                                80,90,100,110,120,130,140,
                                80,90,100,110,120,130,140]}
        
        df_IMP = pd.DataFrame(hc_data)
        df = pd.DataFrame({"heat capacity":self.Btu_per_ft2F_to_J_per_m2K * df_IMP["heat_capacity_IMP"],
                              "percent solid":df_IMP["percent_solid"],
                              "thickness":self.in_to_m * df_IMP["thickness_in"],
                              "density":self.lbpft3_to_kgpm3*df_IMP["density_IMP"]})
        if self.include_plots:
            fig,axl = plt.subplots(3,1,figsize=(10,20))
            axl[0].scatter(df["percent solid"],df["heat capacity"], color="red")
            axl[1].scatter(df["thickness"],df["heat capacity"], color="blue")
            axl[2].scatter(df["density"],df["heat capacity"], color="green")

        
        X = df[["percent solid","thickness","density"]]
        Y = df[["heat capacity"]]
        
        model = sm.OLS(Y, X).fit()
        predictions = model.predict(X) 
 
        print_model = model.summary()
        
        # from multi-variate linear to multi-variate polynomial fits.
        if self.include_plots:
            fig2,axl = plt.subplots(1,2,figsize=(20,10))
        for i in range(5):
            poly = PolynomialFeatures(degree=i+1)
            X_ = poly.fit_transform(X,y=Y)
            model = sm.OLS(Y, X_).fit()
            predictions = model.predict(X_)
            df['heat capacity fit polynomial order {0:2d}'.format(i+1)] = predictions
            df['heat capacity errors polynomial order {0:2d}'.format(i+1)] = 100*(predictions - df["heat capacity"])/df["heat capacity"]
            if self.include_plots:
                df['heat capacity errors polynomial order {0:2d}'.format(i+1)].plot(ax=axl[0], label="poly {0:2d}".format(i+1))
        if self.include_plots:   
            axl[0].legend()
            axl[0].set_title("Fits with 1 bad data point")

        # THE np.nan used to be 6.9 but this makes not sense w/r to the the 
        # data and the polynomial fits and associated error proved this by being
        # unable to fit the spurious point.
        df.drop(index=32,inplace=True)
        # redo X and Y because we have dropped a point.
        X = df[["percent solid","thickness","density"]]
        Y = df[["heat capacity"]]
        if self.include_plots:
            fig3,ax3 = plt.subplots(1,1)
        # we not stop at order 3 because it has the best combination of error and complexity:
        #    poly  2 error max: 8.279 min: -8.256 average magnitude: 1.888
        #    poly  3 error max: 2.006 min: -1.714 average magnitude: 0.731 X
        #    poly  4 error max: 2.044 min: -1.665 average magnitude: 0.655
        #    poly  5 error max: 1.958 min: -1.561 average magnitude: 0.635
        for i in range(3):
            poly = PolynomialFeatures(degree=i+1)
            X_ = poly.fit_transform(X,y=Y)
            model = sm.OLS(Y, X_).fit()
            predictions = model.predict(X_)
            df['heat capacity fit polynomial order {0:2d}'.format(i+1)] = predictions
            err = 100*(predictions - df["heat capacity"])/df["heat capacity"]
            df['heat capacity errors polynomial order {0:2d}'.format(i+1)] = err
            if self.include_plots:
                df['heat capacity errors polynomial order {0:2d}'.format(i+1)].plot(ax=axl[1], label="poly {0:2d}".format(i+1))
                print("poly {0:2d} error max: {1:5.3f} min: {2:5.3f} average magnitude: {3:5.3f}".format(i+1,np.max(np.max(err.values)),np.min(err.values),np.mean(abs(err.values))))
        
        if self.include_plots:
            axl[1].legend()
            axl[1].set_title("Fits bad data removed")
            model.summary()
            print("The model parameters are:")
            params = model.params
            for param in params:
                print("{0:12.8e}".format(param))
                
            print(poly.get_feature_names())
        
        # now test the implementation in ElCanoBuildingEnergy_Demand_Load_Model.py.concrete_wall
        HC_val1 = 19.312 / self.Btu_per_hft2F_to_W_per_m2K * self.Btu_per_ft2F_to_J_per_m2K
        HC_val2 = 42.7136 / self.Btu_per_hft2F_to_W_per_m2K * self.Btu_per_ft2F_to_J_per_m2K
        wall1 = ec_be.concrete_wall(1280,0.35,0.1016,0.0)
        wall2 = ec_be.concrete_wall(1760,(100-52)/100,0.2032,0.0)
        
        err1 = np.abs(100*(wall1.HC_value - HC_val1)/HC_val1)
        err2 = np.abs(100*(wall2.HC_value - HC_val2)/HC_val2)
        
        self.assertTrue(err1 < 2.006)
        self.assertTrue(err2 < 2.006)

        
    def test_thermo_props(self):
        # saturated vapor pressure of water
        # from https://en.wikipedia.org/wiki/Vapour_pressure_of_water "Buck"
        temps = [273,293,308,323,348,373]
        h2o_vp_vals = [611.2,2338.3,5626.8,12349.0,38595.0,101325.0]
        tp = ec_be.thermodynamic_properties()     
        percent_error_allowed = 0.85
        
        for temp,val in zip(temps,h2o_vp_vals):
            h2o_vp = tp.h2o_saturated_vapor_pressure(101325,temp-tp.CelciusToKelvin)
            err = 100 * (h2o_vp - val)/val
            if err > 0.85:
                u = 1
            self.assertLess(np.abs(err),percent_error_allowed)
            
        temps = [2,10,30,40,60]
        latent_heat = [2.4962e6, 2.4772e6, 2.4298e6, 2.406e6, 2.3577e6]
        
        for temp,lh in zip(temps,latent_heat):
            lh2 = tp.latent_heat_of_water(temp)
            err = 100 * (lh - lh2)/lh2
            self.assertLess(np.abs(err),1.0)
        
        # test humidity ratio at 100% rh and 25C and 1atm
        temp = 25
        w_test = tp.humidity_ratio(temp,101325,1)
        w_sat25 = 0.019826
        err = 100*(w_test - w_sat25)/w_sat25
        self.assertLess(np.abs(err),2.0)
            
        # test enthalpy
        h_air = tp.humid_air_enthalpy(temp,1,101325)
        err = 100*(h_air - 75300)/75300 
        self.assertLess(np.abs(err),1.0)
        h_air_dry = tp.humid_air_enthalpy(temp,0,101325)
        err = 100*(h_air_dry - 25150)/25150 
        self.assertLess(np.abs(err),1.0)
        # TODO use thermo text to establish a test case
        
        
    def test_wet_bulb_temperatures(self):
        # test relative humidity limit where wetbulb should nearly be drybulb
        Pressure = 101325
        RH_1 = 99
        Drybulb = [10,20,30,40,50,60,70,80]
        for TempC in Drybulb:
            wbt = tp.low_elevation_wetbulb_DBRH(TempC,RH_1,Pressure)
            self.assertTrue(np.abs(wbt - TempC) < 0.5)
            
        # from https://www.omnicalculator.com/physics/wet-bulb
        RH_1 = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
        wetbulb = [10.772, 12.736, 15.9, 18.37, 20.45, 22.297, 23.996, 25.596, 27.13, 28.62, 29.936]
        Drybulb = 30
        for wet, RH in zip(wetbulb,RH_1):
            wbt = tp.low_elevation_wetbulb_DBRH(
                Drybulb,RH,Pressure)
            # less than 1% difference between the online calculator and the model here.
            self.assertTrue(100 * np.abs(wbt - wet)/wet < 1.0)
            
    def test_fan_energy_model(self):
        pass
    
    def test_refrigerator_model(self):
        pass
    
    def test_wall_ac_model(self):
        df_coef = pd.DataFrame([["WindowUnitAC1_MeissnerEtAl","EER",-0.057940002,0.175064,	-0.00321,-0.08422,0.000726,0.001297],
                   ["WindowUnitAC1_MeissnerEtAl","TC",0.536699,0.156761,-0.00274,-0.09107,0.000812,0.000893],
                   ["WindowUnitAC1_MeissnerEtAl","SC",2.137813,0.018914,-0.00133,-0.08013,0.000594,0.001526]],
                            columns=["AC Unit","Curve Type","a0","a1","a2","a3","a4","a5"])
        
        df_thermostat = pd.DataFrame([["WindowUnitAC1_MeissnerEtAl","Cooling (⁰C)",25,25,25,25],
                                      ["WindowUnitAC1_MeissnerEtAl","Heating (⁰C)"	,20,20,20,20]],
                                     columns=["AC Unit","Type","Blue sky","Tier 1","Tier 2","Tier 3"])

        ACtest = Wall_AC_Unit(TC=3500,
                              SCfrac=0.9,
                              derate_frac=0.9,
                              npPower=1500,
                              flow_rated=450,
                              df_coef=df_coef,
                              df_thermostat=df_thermostat,
                              tier="Tier 1",
                              Name="WindowUnitAC1_MeissnerEtAl")
        
        (TC_avg, SC, mdot_condensed, power_avg, unmet_cooling) = ACtest.avg_thermal_loads(
                                 DBT_internal=25.0,
                                 rh_internal=0.6,
                                 DBT_external=35.0,
                                 Pressure=101325,
                                 num_AC=1,
                                 volume=600.0,
                                 time_step=1,
                                 energy_demand_unrestricted=-1000)
    
    
    def test_TieredAnalysis_template(self):
        tiered_load_template_path = os.path.join("ExcelLoadData","TieredLoads_Template.xlsx")
        obj = ec_be.TieredAnalysis(tiered_load_template_path,False,10,"Results")
        results = obj.df_results
        results.plot(subplots=True)
        
        building ="Building 1"
        tier = "Blue sky"
    
        b1bs = results[(results["Building"]==building) & (results["Tier"]==tier)]
        b2bs = results[(results["Building"]=="Building 2") & (results["Tier"]==tier)]
        b2bs.index = b1bs.index
        # the air-conditioned case for building 2 should always be cooler w/r to 
        # indoor air temperature! 
        self.assertTrue((b1bs["IndoorAirTemp"] > b2bs["IndoorAirTemp"]).values.all())
        self.assertTrue((b1bs["StructureTemp"] > b2bs["StructureTemp"]).values.all())
        fig,axl = plt.subplots(2,1)
        b1bs[["IndoorAirTemp","OutsideAirTemp","StructureTemp"]].plot(ax=axl[0])
        b1bs[["SolarGains"]].plot(ax=axl[1])
        
        
        pass
    
    def test_Tiered_Analysis_appliance_schedule_and_appliance_counts(self):
        ElCano_Tiered_load_path = os.path.join("ExcelLoadData", "TieredLoads_Testing2.xlsx")
        obj = ec_be.TieredAnalysis(ElCano_Tiered_load_path,False,24,"Results",True)
        
        building_noncrit = obj.buildings["Townhome2B_Elev"]['Non-critical']['Townhome2B_Elev_Home_Medical']
        app_sch_compare_noncrit = building_noncrit.appliance_schedule
        building_tier1 = obj.buildings["Townhome2B_Elev"]['Tier 1']['Townhome2B_Elev_Home_Medical']
        app_sch_compare_tier1 = building_tier1.appliance_schedule
        building_tier2 = obj.buildings["Townhome2B_Elev"]['Tier 2']['Townhome2B_Elev_Home_Medical']
        app_sch_compare_tier2 = building_tier2.appliance_schedule
        building_tier3 = obj.buildings["Townhome2B_Elev"]['Tier 3']['Townhome2B_Elev_Home_Medical']
        app_sch_compare_tier3 = building_tier3.appliance_schedule
        
        barea = building_tier3.total_area
        # manual calculation: First the schedules that apply to 'Townhome2B_Elev_Home_Medical' then to "Townhome2B_Elev"
        
        range_noncrit = 2000 * np.array([0,0,0,0,0,0,0.5,0.2,0,0,0.1,0.4,0.2,0,0,0,0.5,0.8,0.2,0,0,0,0,0])
        homemed_tier2 = 207 * np.array([0.3,0.3,0.34,0.34,0.34,0.35,0.35,0.4,0.5,0.6,0.7,0.7,0.95,1,1,1,1,0.9,0.7,0.6,0.5,0.4,0.3,0.3])
        phone_tier1 = 10 * np.array([0.1,0.1,1,1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,1,1,0.1,0.1])
        laptop_tier2 = 60 * np.array([0.1,0.1,0.1,0.1,0.1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.1,1,1,0.1,0.1])
        laptop_tier3 = 60 * np.array([0.1,0.1,0.1,0.1,0.1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.1,1,1,0.1,0.1])
        hotwater_tier2 = 4000 * np.array([0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0])
        smoke_tier1 = 2 * np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        wash_noncrit = 500 * np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
        dryer_noncrit = 3000 * np.array([0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0])
        microwave_tier1 = 1000 * np.array([0,0,0,0.025,0.05,0.1,0.15,0.1,0.05,0.06,0.15,0.2,0.15,0.05,0.025,0.03,0.05,0.15,0.1,0.05,0.05,0.025,0.01,0.01])
        tv_tier3 = 200 * np.array([0.05,0.05,0.05,0.1,0.2,0.6,0.85,0.6,0.2,0.2,0.3,0.5,0.5,0.5,0.6,0.7,0.8,0.9,1,0.9,0.8,0.7,0.6,0.2])
        
        app_sch_manual_noncrit = pd.Series(range_noncrit + homemed_tier2 + phone_tier1 + laptop_tier2 +
                                  laptop_tier3 + hotwater_tier2 + smoke_tier1 + wash_noncrit + 
                                  dryer_noncrit + microwave_tier1 + tv_tier3, name="Total Appliance Electric Loads")/barea
        app_sch_manual_tier1 = pd.Series(phone_tier1 + smoke_tier1 + microwave_tier1, name="Total Appliance Electric Loads")/barea
        app_sch_manual_tier2 = pd.Series(homemed_tier2 + phone_tier1 + laptop_tier2 +
                                  hotwater_tier2 + smoke_tier1 + 
                                  microwave_tier1, name= "Total Appliance Electric Loads")/barea
        app_sch_manual_tier3 = pd.Series(homemed_tier2 + phone_tier1 + laptop_tier2 +
                                  laptop_tier3 + hotwater_tier2 + smoke_tier1 + 
                                  microwave_tier1 + tv_tier3, name="Total Appliance Electric Loads")/barea
        # verify that the program works as intended. The underlying tier_map is:
        #     tier_map = {"Non-critical":["Non-critical","Tier 1","Tier 2","Tier 3"],
        #        "Tier 1":["Tier 1"],
        #        "Tier 2":["Tier 1","Tier 2"],
        #        "Tier 3":["Tier 1","Tier 2","Tier 3"]}
        pd.testing.assert_series_equal(app_sch_compare_noncrit,app_sch_manual_noncrit)
        pd.testing.assert_series_equal(app_sch_compare_tier1,app_sch_manual_tier1)
        pd.testing.assert_series_equal(app_sch_compare_tier2,app_sch_manual_tier2)
        pd.testing.assert_series_equal(app_sch_compare_tier3,app_sch_manual_tier3)
        
        # now move on to appliance counts
        num_fullsize_fridge_noncrit = building_noncrit.complex_appliances['fridges']['Full size Fridge'][0]
        num_fullsize_fridge_tier1 = building_tier1.complex_appliances['fridges']['Full size Fridge'][0]
        num_fullsize_fridge_tier2 = building_tier2.complex_appliances['fridges']['Full size Fridge'][0]
        num_fullsize_fridge_tier3 = building_tier3.complex_appliances['fridges']['Full size Fridge'][0]
        # only 1 fridge on Tier 1. All should be 1
        self.assertEqual(num_fullsize_fridge_noncrit,2)
        self.assertEqual(num_fullsize_fridge_tier1,1)
        self.assertEqual(num_fullsize_fridge_tier2,1)
        self.assertEqual(num_fullsize_fridge_tier3,2)
        
        num_mini_fridge_noncrit = building_noncrit.complex_appliances['fridges']['Mini Fridge'][0]
        num_mini_fridge_tier1 = building_tier1.complex_appliances['fridges']['Mini Fridge'][0]
        num_mini_fridge_tier2 = building_tier2.complex_appliances['fridges']['Mini Fridge'][0]
        num_mini_fridge_tier3 = building_tier3.complex_appliances['fridges']['Mini Fridge'][0]
        # only 1 fridge on Tier 1. All should be 1
        # we have a line for Home Medical that puts 5 minifridges on Tier 3 replacing the 1 for the normal "Townhome2B_Elev" input
        self.assertEqual(num_mini_fridge_noncrit,7)  
        self.assertEqual(num_mini_fridge_tier1,1)
        self.assertEqual(num_mini_fridge_tier2,2)
        self.assertEqual(num_mini_fridge_tier3,6)
        
        
        
if __name__ == "__main__":
    unittest.main()
    plt.show()
        
        
        
        
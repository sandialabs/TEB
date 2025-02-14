# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:36:49 2025

@author: dlvilla
"""
if __name__ == "__main__":
    import TEB.simulator.sim as ec_be
#    from TEB.simulator.complex_appliances import Wall_AC_Unit
    from TEB.simulator.thermodynamics import thermodynamic_properties as tp
    tp = tp()
#import numpy as np
import unittest
from matplotlib import pyplot as plt
import shutil
from copy import deepcopy
#import pandas as pd
#import statsmodels.api as sm
#from sklearn.preprocessing import PolynomialFeatures

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
        cls.results_path = "LPGResults"
        cls.data_path = "LPG_data"
        cls.data_path2 = "LPG_data2"
        cls.run_all = False
        cls.lpg_path_dict = {"Townhome2B_Elev":cls.data_path,
                             "Townhome2B_NotElev":cls.data_path2,
                             "SingleFam3B_Elev":cls.data_path,
                             "Townhome3B":cls.data_path,
                             "Commercial":cls.data_path2,
                             "Institutional":cls.data_path2,
                             "Townhome2B_Elev_Home_Medical":cls.data_path2,
                             "Townhome2B_NotElev_Home_Medical":cls.data_path,
                             "SingleFam3B_Elev_Home_Medical":cls.data_path2,
                             "Townhome2B_Elev_Home_Medical_AC":cls.data_path,
                             "Townhome2B_NotElev_Home_Medical_AC":cls.data_path2,
                             "SingleFam3B_Elev_Home_Medical_AC":cls.data_path,
                             "Townhome2B_Elev_AC":cls.data_path2,
                             "Townhome2B_NotElev_AC":cls.data_path,
                             "SingleFam3B_Elev_AC":cls.data_path2}
        cls.incomplete_dict = {val:cls.lpg_path_dict[val] for val in 
                               ["Townhome2B_Elev","SingleFam3B_Elev"]}
        
        cls.bad_path = deepcopy(cls.lpg_path_dict)
        cls.bad_path["SingleFam3B_Elev_Home_Medical"] = "Nonsense!"
        
        cls.tiered_load_test2_path = os.path.join(os.path.dirname(__file__),
                                                  "ExcelLoadData",
                                                  "TieredLoads_Testing2.xlsx")
        plt.close('all')
        
        

    def test_LPG_template(self):  
        """
        Test making the Load Profile Generator (LPG) a Source of load information
        https://www.loadprofilegenerator.de/
        
        The datafiles used, are for example only but constitute the output
        of a house analysis in LPG.
        
        """
        
        if self.run_all:
        
            # Setup the results directory
            if os.path.exists(self.results_path):
                shutil.rmtree(self.results_path)
            os.mkdir(self.results_path)
            
            # setup the TEB input spreadsheet path
            tiered_load_template_path = os.path.join(os.path.dirname(__file__),"ExcelLoadData","TieredLoads_Template.xlsx")
            
            # create the Tiered Analysis object
            obj = ec_be.TieredAnalysis(tiered_load_template_path,False,3,self.results_path,lpg_path=self.data_path)

    def test_2_LPG_template(self):
        
        """
        test the presence of repeat names (i.e., buildings that use another as a template and then have some minor changes.)
        
        also test using a dictionary 
        
        """
        
        if self.run_all:
            # Setup the results directory
            if os.path.exists(self.results_path):
                shutil.rmtree(self.results_path)
            os.mkdir(self.results_path)
            
            # setup the TEB input spreadsheet path
            tiered_load_template_path = os.path.join(os.path.dirname(__file__),"ExcelLoadData","TieredLoads_Testing2.xlsx")
            
            # You must have both all building names on the "Buildings" sheet and all repeat building names 
            # (i.e., the same building but with a different equipment configuration.) on the "RepeatBuildingConfigs"
            # sheet.

            # create the Tiered Analysis object
            obj = ec_be.TieredAnalysis(self.tiered_load_test2_path,
                                       False,
                                       3,
                                       self.results_path,
                                       lpg_path=self.lpg_path_dict)
            
            spot_check = obj.df_results.iloc[100,:]
            self.assertAlmostEqual(spot_check["IndoorAirTemp"], 25.025658, 5)
            self.assertAlmostEqual(spot_check["SolarGains"],0.0, 5)
            self.assertAlmostEqual(spot_check["StructureTemp"],24.546807, 5)
            self.assertAlmostEqual(spot_check["IndoorSurfaceTemp"],24.678683, 5)
            self.assertAlmostEqual(spot_check["OutsideAirTemp"],23.9, 2)
            self.assertAlmostEqual(spot_check["PlugInFans"],0.0, 2)
            self.assertAlmostEqual(spot_check["StaticElectricLoads"],117.612585, 5)
            self.assertAlmostEqual(spot_check["Refrigerators"],20.376264, 5)
            self.assertAlmostEqual(spot_check["Wall_ACs"],0.0, 2)
            self.assertAlmostEqual(spot_check["Lights"],165.0, 2)
            self.assertAlmostEqual(spot_check["TotalElectricity"],302.988849, 5)
            self.assertAlmostEqual(spot_check["IndoorAirRelativeHumidity"],0.845782, 5)
            self.assertAlmostEqual(spot_check["UnmetCooling"],0.0, 2)
            self.assertAlmostEqual(spot_check["OutdoorAirRelativeHumidity"],0.9, 2)
            self.assertAlmostEqual(spot_check["Occupants"],3.0, 2)
            self.assertAlmostEqual(spot_check["UnmetHeating"],0.310002, 5)
            self.assertAlmostEqual(spot_check["HeatLoadToMeetThermostat"],-12.824641, 5)
            self.assertAlmostEqual(spot_check["Central_AC"],0.0, 2)
            self.assertAlmostEqual(spot_check["Month"],1,1)
            self.assertAlmostEqual(spot_check["DayOfWeek"],2,1)
            self.assertAlmostEqual(spot_check["DayOfMonth"],1,1)
            self.assertAlmostEqual(spot_check["HourOfDay"],2,1)
            self.assertEqual(spot_check["MasterBuilding"],"SingleFam3B_Elev")
            self.assertAlmostEqual(spot_check["BuildingArea"],73.67211,5)
            self.assertEqual(spot_check["Tier"],"Tier 3")
            self.assertEqual(spot_check["Building"],"SingleFam3B_Elev")
            self.assertAlmostEqual(spot_check["Hour"],1,1)


        
    def test_bad_dict(self):
        if self.run_all:
            with self.assertRaises(ValueError):
                ec_be.TieredAnalysis(self.tiered_load_test2_path,
                      False,
                      10,
                      self.results_path,
                      lpg_path=self.incomplete_dict)

    
        
    def test_incorrect_path(self):
        if self.run_all:
            with self.assertRaises(FileNotFoundError):
                ec_be.TieredAnalysis(self.tiered_load_test2_path,
                      False,
                      10,
                      self.results_path,
                      lpg_path=self.bad_path)
        

if __name__ == "__main__":
    unittest.main()

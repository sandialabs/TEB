# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:56:23 2025

LPG = load profile generator



@author: dlvilla
"""
import pandas as pd
import os

class LPG_data(object):
    
    lpg_files = {"electricity":"SumProfiles.Electricity.csv",
                 "internal_heat":"SumProfiles.Inner Device Heat Gains.csv"}
    
    def __init__(self, data_path):
        output = {}
        
        for output_name, file_name in self.lpg_files.items():
            
            file_path = os.path.join(data_path,file_name)
            
            if os.path.exists(file_path):
            
                raw_data = pd.read_csv(file_path,sep=";")
                
                data = raw_data["Sum [kWh]"]
                data.index = pd.to_datetime(raw_data["Time"],
                                                format="%m/%d/%Y %I:%M %p")
                data_8760 = data.resample("h").sum()
                                    
                output[output_name] = data_8760
                
                
            else:
                raise FileNotFoundError("The LPG datafile {file_path} does"
                    +" not exist, please place it in the right location!")
        
        self.output = output
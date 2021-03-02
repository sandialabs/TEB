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

Created on Wed Nov 11 15:02:45 2020

@author: dlvilla
"""

import os
import numpy as np
import pandas as pd
from Thermodynamics import thermodynamic_properties as tp
tp = tp()

class Wall_AC_Unit(object):
    
    _water_condensing_temperature = 5  # Degrees Celcius
    _HOURS_TO_SECDONDS = 3600
    _C_to_K = 273.15
    
    _MAX_WARNING_MESSAGES = 3
    
    def __init__(self,TC,SCfrac,derate_frac,npPower,flow_rated,df_coef,df_thermostat,tier,Name):
        """
        Inputs 
        
        TC - total cooling capacity
        
        SCfrac - fraction of TC for sensible cooling
        
        derate_frac - derates TC (and SC) based on poor performance in comparison
                      to manufacturer claims and maintenance
        npPower - name plate power input to produce TC
        
        flow_rated - amount of flow circulation from AC m3/h
        
        df_coeff - data frame with performance curve coefficients
        
        df_thermostat - data frame with thermostat information
        
        """
        self.name = Name
        # see Meisner et. al., 2014 for this model and origin of coefficients form
        self._TC = TC * derate_frac
        self._SC = TC * SCfrac * derate_frac
        self._Power = npPower
        self._EER = self._TC / npPower
        self._flow = flow_rated
        self._coef = {}
        for key in df_coef["Curve Type"]:
            self._coef[key] = df_coef[df_coef["Curve Type"]==key].iloc[0,1:]
        self._cooling_setpoint = df_thermostat[df_thermostat["Type"]=="Cooling (⁰C)"][tier].values[0]
        #TODO heating is not yet used!
        self._heating_setpoint = df_thermostat[df_thermostat["Type"]=="Heating (⁰C)"][tier].values[0]
        self.unit_running = False
        self._DBT_valid_range = [17,45]
        self._WBT_valid_range = [17,32]
        self._num_msg = 0
        
    def _check_range(self,valid_range,val,name):
        if valid_range[0] > val or valid_range[1] < val:
            self._num_msg +=1
            if self._MAX_WARNING_MESSAGES > self._num_msg:
                print("Wall_AC_Unit._thermal_performance: " + name 
                      + " is higher than performance curves allow. The program is"
                      + " flat-lining the performance")
            elif self._MAX_WARNING_MESSAGES == self._num_msg:
                print("Wall_AC_Unit._thermal_performance: Discontinuing repeat messages")
            if valid_range[0] > val:
                val_adj = valid_range[0] 
            else:
                val_adj = valid_range[1]
        else:
            val_adj = val
            
        return val_adj
        
    def _thermal_performance(self,DBT_internal,rh_internal,DBT_external,
                          Pressure):
        # Wet-bulb temperture
        rh_internal_perc = 100 * rh_internal
        WBT_internal = tp.low_elevation_wetbulb_DBRH(DBT_internal, rh_internal_perc, Pressure)
        
        # verify internal wetbulb and external dry bulb are within the limits of
        # the performance curves. If not, then assume performance stays constant 
        # at the edges

        DBT_external_adj = self._check_range(self._DBT_valid_range,DBT_external,"External Dry Bulb Temperature")
        WBT_internal_adj = self._check_range(self._WBT_valid_range,WBT_internal,"Internal Wet Bulb Temperature")
        
        # determine the total cooling capacity of the unit
        TC = self._meissner_equation_6(self._TC,"TC",WBT_internal_adj,DBT_external_adj)
        SC = self._meissner_equation_6(self._SC,"SC",WBT_internal_adj,DBT_external_adj)
        EER = self._meissner_equation_6(self._EER,"EER",WBT_internal_adj,DBT_external_adj)
        
        power = TC / EER
        
        # calculate the mass flow of water vapor that has been extracted by the 
        # amount of latent cooling
        LC = TC - SC
        mdot_condensed = LC / tp.latent_heat_of_water(self._water_condensing_temperature)
        
        return TC, SC, mdot_condensed, power
    
    def avg_thermal_loads(self,DBT_internal,rh_internal,DBT_external,
                          Pressure,num_AC,volume,time_step,
                          energy_demand_unrestricted):
        # cooling is counted as negative heat.
        if energy_demand_unrestricted >= 0.0:
            self.unit_running = False
            self.fraction_time_on = 0.0
            SC_avg = 0.0
            power_avg = 0.0
            TC_avg = 0.0
            mdot_condensed_avg = 0.0
            unmet_cooling = energy_demand_unrestricted
        else:
            TC, SC, mdot_condensed, power = self._thermal_performance(DBT_internal,
                                                                     rh_internal,
                                                                     DBT_external,
                                                                     Pressure)
            unmet_cooling = num_AC * SC + energy_demand_unrestricted
            
            if unmet_cooling > 0: # The units will cycle off in the next hour
                unmet_cooling = 0.0 
                avg_on_time_per_unit_fraction_of_time_step = (
                    -energy_demand_unrestricted)/(num_AC * SC)
                SC_avg = SC * avg_on_time_per_unit_fraction_of_time_step
                power_avg = power * avg_on_time_per_unit_fraction_of_time_step
                TC_avg = TC * avg_on_time_per_unit_fraction_of_time_step
                mdot_condensed_avg = mdot_condensed * avg_on_time_per_unit_fraction_of_time_step
                self.fraction_time_on = avg_on_time_per_unit_fraction_of_time_step
            else:
                SC_avg = SC
                power_avg = power
                TC_avg = TC
                mdot_condensed_avg = mdot_condensed
                self.fraction_time_on = 1.0
                
        return (num_AC * TC_avg, 
               num_AC * SC_avg, 
               num_AC * mdot_condensed_avg, 
               num_AC * power_avg,
               unmet_cooling)
                
            
                
        
        
    def _meissner_equation_6(self,Nominal_value, curve_type, WBT_room,DBT_external):

        a = np.array(self._coef[curve_type])
        if type(a[0]) is str:
            a = a[1:]
        terms = np.array([1,WBT_room,WBT_room ** 2, DBT_external, DBT_external ** 2, WBT_room * DBT_external])
        Z = np.dot(terms,a)
        return Nominal_value * Z
    

class Refrigerator(object):
    
    _hours_to_seconds = 3600
    _cp_air = 1003.5 #J/(kg*K)
    _rho_air = 1.225 # kg/m3   both of these are assumed constant
    _C_to_F = 273.15 # K
    
    def __init__(self,surf_area,
                      aspect_ratio,
                      R_total,
                      T_diff,
                      T_inside,
                      ach_mix,
                      ach_schedule,
                      frac_Carnot,
                      fan_power,
                      compressor_efficiency,
                      compressor_power,
                      name=""):
        self.name = name
        # TODO - add input checking!
        self._Ri = R_total * surf_area
        self._Vol = self._rectangle_volume(aspect_ratio,surf_area)
        self._Tc = T_inside - T_diff
        self._Tdiff = T_diff
        self._Tf = T_inside
        self._ach = ach_mix
        self._ach_sch = ach_schedule
        self._Pf = fan_power
        self._fc = frac_Carnot
        self._eta_c = compressor_efficiency
        self._Pc_max = compressor_power
        self._Af = surf_area
        self.fridge_setpoint_reached = True
    
    def _rectangle_volume(self,AR,S):
        return S / (2 + 4 * AR) * np.sqrt(S/(2/(AR*AR) + 4/AR))
        
    def avg_thermal_loads(self,T_room,hour_of_day,ts):
        """
        Inputs
          ts - time step in hours
        """
        #TODO - create a more dynamic model of the refrigerator. Need heat capacity 
        #       and possibly also a model with a freezer. For now, let's keep this inexpensive.
        #       performance curves could really help w/r to quantifying inefficient modes
        #       of operation.
        mdot = self._Vol * self._ach * self._ach_sch[hour_of_day] * self._rho_air / self._hours_to_seconds
        mdot_cp = mdot * self._cp_air * (T_room - self._Tf + self._C_to_F)
        #max_heat = self._Vol * self._rho_air * self._cp_air * (T_room + self._C_to_F - self._Tf)
        Qc = mdot_cp + 1/self._Ri *(T_room - self._Tf + self._C_to_F)
        Tcondenser = T_room + self._Tdiff
        if Tcondenser + self._C_to_F - self._Tc == 0: # avoid division by zero
            COP = 0.0
        else:
            COP = self._fc * self._Tc / (Tcondenser + self._C_to_F - self._Tc)
            
        if COP == 0:
            Qh = 0
            Pc = 0
        else:
            Qh = Qc * (1 + COP)/COP
            Pc = Qc / (COP * self._eta_c)
        if Pc > self._Pc_max:
            # roughly assume COP stays constant
            Pc = self._Pc_max
            Qc = COP * Pc * self._eta_c
            Qh = Qc * (1 + COP)/COP
            self.fridge_setpoint_reached = False  # a sign food will spoil - we do not track
                                                  # the temperature. 
        else:
            self.fridge_setpoint_reached = True
        
        Qnet = Qh - Qc  # net heat exchange with the room
        Pnet = Pc + self._Pf
    
        return Qnet, Pnet
    
class Fan(object):
    
    def __init__(self, fan_name, power, heat_energy_ratio, ExtACH, SpeedEfficiencies, Curve_Temperatures):
        self.name = fan_name
        self._power = power
        self._heat_ratio = heat_energy_ratio
        self._ExtACH = ExtACH
        self._SpeedEfficiencies = SpeedEfficiencies
        self._Curve_Temperatures = Curve_Temperatures
        
    
    def fan_energy_model(self,T_in,T_out):
        
        """ Fan use behavior given external and internal temperature
        
            conditions
        """
        
        # find what speed 
        speed_index = self._find_speed(T_in)
        if speed_index == -1:
            fan_power = 0.0
            fan_ACH = 0.0
            fan_heat = 0.0
            self.ison = False
        else:
            #TODO improve this model
            speed = self._Curve_Temperatures.index[speed_index]
            fan_power = self._power * speed /  self._SpeedEfficiencies.loc[speed]
            if T_out > T_in:
                fan_ACH = 0.0
            else:
                fan_ACH = self._ExtACH * speed
            fan_heat = fan_power * self._heat_ratio
            self.ison = True

        return fan_power, fan_ACH, fan_heat
    
    def _find_speed(self, T_in):
        ind = np.where(self._Curve_Temperatures.values < T_in)
        if len(ind[0]) == 0:
            return -1
        else:
            return ind[0][-1]
    
class Light(object):
    # simple object
    def __init__(self,name,light_type,lux_threshold,power,fraction_heat):
        self.name = name
        self.light_type = light_type
        self.lux_threshold = lux_threshold
        self.power = power
        self.fraction_heat = fraction_heat
    # energy model is elsewhere. Lights cannot be controlled differently
    # for now stick strictly to interior and exterior lights
    
    
        
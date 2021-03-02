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


Created on Wed Nov 11 15:35:36 2020

@author: dlvilla
"""

import numpy as np

class thermodynamic_properties(object):
    
    
    th2o_boil1atm = 373.15  #Kelvin
    CelciusToKelvin = 273.15 # Kelvin
    th2o_critical = 647 # Critical temperature Kelvin
    hectopascal_to_pascal = 100.0 # conversion ratio
    temp_lower_limit = 180.0 # Kelvin - from IAT lowest allowable temperature
    th2o_triple = 273.16 # triple point temperature of h2o
    ph2o_triple = 611.657  # triple point pressure of h2o Pa
    air_M = 0.02897 # kg/mol
    h2o_M = 0.01801528 # kg/mol
    R_gas = 8.31446261815324 # J/(K*mol)
    rho_h2o = 1000.0 # kg/m3
    rho_h2o_vapor = 0.00485
    rho_air = 1.2 # kg/m3
    air_heat_capacity = 1006 # J/kg
    water_vapor_heat_capacity = 1860 #J/kg
    RdRvRatio = h2o_M / air_M
    
    def __init__(self):
        #These are water vapor specific and are therefore not included as inputs
        # vapor pressure coefficients.
        Aewi = np.zeros(2)
        Bewi = np.zeros(2)
        Cewi = np.zeros(2)
        Dewi = np.zeros(2)
        Eewi = np.zeros(2)
        
        a_ewi = np.zeros(2)
        b_ewi = np.zeros(2)
        c_ewi = np.zeros(2)
        d_ewi = np.zeros(2)
        
        Aewi[0] = 4.8e-4
        Bewi[0] = 3.47e-6
        Cewi[0] = 5.9e-10
        Dewi[0] = 23.8
        Eewi[0] = -3.1e-2
        Aewi[1] = 4.1e-4
        Bewi[1] = 3.48e-6
        Cewi[1] = 7.4e-10
        Dewi[1] = 30.6
        Eewi[1]= -3.8e-2
        a_ewi[0] = 6.1115
        b_ewi[0] = 23.036
        c_ewi[0] = 279.82
        d_ewi[0] = 333.7
        a_ewi[1] = 6.1121
        b_ewi[1] = 18.564
        c_ewi[1] = 255.57
        d_ewi[1] = 254.40
        self._Aewi = Aewi
        self._a_ewi = a_ewi
        
        self._Bewi = Bewi
        self._b_ewi = b_ewi
        
        self._Cewi = Cewi
        self._c_ewi = c_ewi
        
        self._Dewi = Dewi
        self._d_ewi = d_ewi
        
        self._Eewi = Eewi
        
        # latent heat of vaporization coefficients
        coef = np.zeros(7)
        coef[6] = 0.000000000392177
        coef[5] = -0.000000897818
        coef[4] = 0.000826231
        coef[3] = -0.406878
        coef[2] = 114.29
        coef[1] = -19679.4
        coef[0] = 4244030.0
        self._lat_coef = coef
        
        self.warnings = []
        
    def rho_water_vapor(self,P,T):
        Tk = T + self.CelciusToKelvin
        return P / (self.R_gas / self.h2o_M * Tk) 
    
    def humidity_ratio(self,TempC,Pressure,rh):
        Psat = self.h2o_saturated_vapor_pressure(Pressure,TempC)
        return (rh) * Psat * self.RdRvRatio  / (Pressure - Psat)
    
    def humid_air_enthalpy(self,TempC,rh,Pressure):
        w = self.humidity_ratio(TempC, Pressure, rh)
        return (self.air_heat_capacity * TempC 
                + w * (self.water_vapor_heat_capacity * TempC
                + self.latent_heat_of_water(TempC)))
        
    
    def humid_air_molar_mass(self,air_pressure,rh,TempC):
        """
        Not yet tested...
        """
        
        # calculate saturated vapor pressures of water
        Psat_h2o = self.h2o_saturated_vapor_pressure(air_pressure,TempC)
        # water partial pressures                                                                 
        Ph2o = Psat_h2o * rh

        # mole fractions
        yh2o = Ph2o / air_pressure 
        
        # molar masses
        Mh2o = self.h2o_M
        Mair = self.air_M
        # mixed molar mass
        Mmix = yh2o * Mh2o + (1 - yh2o) * Mair
        
        if Mmix < 0.0:
            raise ValueError("Mmix must be greater than zero!")
        return Mmix
    
    def latent_heat_of_water(self,TempC):
        """
        ' valid range from 275K absolute to 475K.
        ' The data was fit to a 6th order polynomial from the NIST REFPROP database
        ' see excel spreadsheet ./MaterialProp/SaturationTableRefProp.xlsx for data table and fit characteristics
        
        Inputs: Temperature in Celcius
        
        Output: Latent heat of vaporizatio in J/kg
        
        """
        TempK = TempC + self.CelciusToKelvin

        if TempK < 275 or TempK > 475:
            raise ValueError("Temperature must be between 275 and 475 kelvin")
        else:
            return np.polynomial.polynomial.polyval(TempK,self._lat_coef)

    
    def h2o_saturated_vapor_pressure(self,pressure,TempC): # return in Pascals
    
        # translated from Fortran90 in the Initial Atmospheric Transport model
        """! ! For temperatures from -80C to 100C, saturated vapor pressure from:
        !       Buck, Arden L. "New Equations for Computing Vapor Pressure and Enhancement Factor" Applied Meteroogy, dec. 1981, v20, 1527-1532,
        
        ! ! For temperatures from 100C to 374C, saturated vapor pressure from:
        !  REFPROP : Reference Fluid Thermodynamic And Transport Properties
        !           NIST Standard Reference Database 23, Version 9.1 
        !           Eric W. Lemmon, Marcia L. Huber, and Mark O. McLinden 
        !           Applied Chemicals and Materials Division
        !           National Institute of Standards and Technology
        !           Boulder, CO 80305
        !           Eric.Lemmon@NIST.gov
        !           Marcia.Huber@NIST.gov
        ! Spreadsheet containing the water data in: REFPROP_SaturationFitForAboveBoilingPointEvaluations.xlsx
        !     The 6th order polynomial fit has error bounds of -.03 to 0.07% error from the NIST REFPROP model
        
        !   ew(T) = a * exp [((b-T/d)T)/(T+c)]   vapor pressure of pure water (no air mixture)
        !   ewstar = ew * f = (saturated water vapor pressure)
        !   f(T,P) = 1 + A + P*[B + C*(T + D + E*P)**2] 
        !   tsat is the saturated temperature
        !
        ! we use ew6, fw5 and ei3, fi5 to achieve a valid temperature range of -80C to 100C for this function.
        
        !   where P in hectopascals (hPa)
        !         T in C
        !         ewstar = hPa
        use precision
        use constants, only: th2o_boil1atm, CelciusToKelvin, th2o_critical, th2o_triple, ph2o_triple, Aewi, &
                            Bewi, Cewi,Dewi,Eewi,a_ewi,b_ewi,c_ewi,d_ewi, hectopascal_to_pascal, temp_lower_limit
        use error_codes,  only : ERROR_TEMPERATURE_OUT_OF_BOUNDS, temp_error_code
        use iat_files, only : write_error_to_iat_files
        use error
    
        implicit none
    
        real(kind=wp) :: pressure
        real(kind=wp), intent(in) :: temperature 
        real(kind=wp) :: temperature_adj, h2o_saturated_vapor_pressure
        real(kind=wp) :: e, estar, temperature_C, t_m_triple
        real(kind=wp) :: f
        real(kind=wp) :: pressure_mb  ! ambient pressure in milibars
        integer :: ind
    
        ! if temperature > th2o_boil1atm and temperature < th2o_critical and rh > 1
        ! then pockets of superheated water vapor will precipitate but no condensation will occur  
        ! The temperature is beyond the valid range of the vapor pressure relationship
        ! the physics here assumes that the additional vapor effects are negligible since
        ! the cloud is rising very quickly a future update might look at a mixture law for specific heats
        ! for the pure vapor phase. The physics of the cloud are fairly uncertain at this point anyway"""
        
        temperature = TempC + self.CelciusToKelvin
        
        if temperature > self.th2o_critical: # changes to temperature do not stick since intent(in) is used
            temperature_adj = self.th2o_critical   # there is no chance of the pressure rising above the critical pressure of 22.064MPa rh is going to be very low anyway! 
                                     # we will assume that slight overestimation of the RH will not have a significant effect
        elif temperature < self.temp_lower_limit: # -80C (193.15Kelvin) is the lowest temperature of the relationship
            # throw an error, iat clouds should not be reaching beyond the stratosphere or dropping to such low temperatures. Something is wrong!
             raise ValueError("The temperature must be greater than {0:5.2f}".format(self.temp_lower_limit)) 
        else:
            temperature_adj = temperature
        # be careful this is not always TempC!
        temperature_C    = temperature_adj - self.CelciusToKelvin
        pressure_mb = pressure/self.hectopascal_to_pascal
    
        # determine if the vapor pressure over water or over ice is needed.
        if temperature < self.th2o_boil1atm: # use Buck (see reference above) relationships
            if temperature < self.th2o_triple:  # use over ice vapor pressures (no-subcooling considered)
                # ei3 and fi5
               ind = 0 # over ice
            else: #fw5 and ew6
               ind = 1 # over water    

            # mixed air correction factor 
            f = 1 + self._Aewi[ind] + pressure_mb*(self._Bewi[ind] + self._Cewi[ind]*(temperature_C + 
                                                self._Dewi[ind] + self._Eewi[ind]*pressure_mb)**2)
            # vapor pressure (over ice or water 
            e = self._a_ewi[ind] * np.exp((self._b_ewi[ind] - temperature_C / self._d_ewi[ind]) * 
                                       temperature_C / (temperature_C + self._c_ewi[ind]))
            # now calculate the saturated temperature (i.e. dewpoint) (Buck eqn (4b)) - coming from:
            # Bogel, W., 1979: New approximate equations for the saturation pressure of water vapor and for humidity parameters
            #                  used in meteorology. European Space Agency, ESA-TT-509 (revised), 150pp.
            # BUCK's RELATIONSHIP DOESN'T WORK AND APPEARS TO HAVE AN ERROR
            #z = log(e/a_ewi[ind])
            #tsat = (d_ewi[ind]/2.)*(b_ewi[ind] - z - (b_ewi[ind]-z)**2 - 4.*c_ewi[ind]*z/(d_ewi[ind])**(0.5))
        
            # corrected water-air mixture vapor pressure.
            estar    = f * e * self.hectopascal_to_pascal # return the answer in Pascal
        else: # use REFPROP relationships - the cloud is still superheated.
            t_m_triple = temperature_adj - self.th2o_triple
            # no correction factor, just use the pure water vapor pressure.        
            e = (10.0**(-1.2184E-16*t_m_triple**6 + 4.4721E-13*t_m_triple**5 - 5.0626E-10*t_m_triple**4 + 
                 2.9910E-07*t_m_triple**3 - 1.1175E-04*t_m_triple**2 + 3.0751E-02*t_m_triple + 
                 2.7933E+00) + self.ph2o_triple)
            estar = e

        h2o_saturated_vapor_pressure = estar
        return h2o_saturated_vapor_pressure
    
    def low_elevation_wetbulb_DBRH(self,TempC,RH_Perc,Pressure):
        """
        Wet bulb temperature as a function of temperature in Celcius and 
        relative humidity in percent.
        
        Only valid at low altitudes (pressure ~ 101325 Pa)
        
        and 1 < RH < 99 , T - 0 to 80C
        
        Mean error is -0.0052C, median is 0.026C mean absolute error is 0.28C
        
        low_elevation_wetbulb_DBRH(TempC,RH_Perc)
        
        Inputs:
            TempC : 80.0 > float > 0.0 : Temperature in Celcius
            
            RH_Perc : 99.0 >= float >= 1.0 : Percent relative humidity
            
            Pressure : float > 80,000 Pa : Atmospheric pressure. Only used to
                                           verify that the realm of accuracy
                                           of the function is maintained.
        """
        
        if RH_Perc < 1.0 or RH_Perc > 99.0:
            #raise ValueError("Relative humidity out of bounds of regression")
            self.warnings.append("thermodynamic_properties.low_elevation_wetbulb_DBRH: "+
                  "Relative humidity out of bounds of regression: Simulation has flat-lined the RH!")
            if RH_Perc < 1.0:
                RH_Perc = 1.0
            else:
                RH_Perc = 99.0
            
        elif TempC < 0.0 or TempC > 80.0:
            raise ValueError("Dry bulb temperature out of bounds of regression")
        elif Pressure < 80000.0:
            raise ValueError("Atmospheric pressure must remain > 80kPa")
        
        return (TempC * np.arctan(0.151977 * (RH_Perc + 8.313659)**0.5) 
              + np.arctan(TempC + RH_Perc) - np.arctan(RH_Perc - 1.676331) 
              + 0.00391838 * (RH_Perc) ** 1.5 * np.arctan(0.023101 * RH_Perc)
              - 4.686035)
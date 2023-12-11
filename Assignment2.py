#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:52:21 2023

@author: Kavi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_mortality_data(Mortality_rate):
    df_data = pd.read_csv('/Users/Kavi/Downloads/Mortality data/Mortality_rate.csv', skiprows = 3)
    df_data.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code'], inplace=True)
    print(df_data)
    
read_mortality_data("Mortality_rate.csv")

def FilterData(df):
    
    countries = ['United States', 'United Kingdom', 'France', 'India', 'China',
                 'Germany', 'Russian Federation', ]
    df = df.loc[df['Country Name'].isin(countries)]

    
    indicator = ['SP.URB.TOTL', 'SP.POP.TOTL', 'SH.DYN.MORT', 'SH.DYN.MORT',
                 'ER.H2O.FWTL.K3', 'EN.ATM.GHGT.KT.CE', 'EN.ATM.CO2E.KT',
                 'EN.ATM.CO2E.SF.KT', 'EN.ATM.CO2E.LF.KT', 'EN.ATM.CO2E.GF.KT',
                 'EG.USE.ELEC.KH.PC', 'EG.ELC.RNEW.ZS', 'AG.LND.FRST.K2',
                 'AG.LND.ARBL.ZS', 'AG.LND.AGRI.K2']
    df = df.loc[df['Indicator Code'].isin(indicator)]
    return df

def Preprocess(df):
   
    df.drop('Country Code', axis=1, inplace=True)
      
    return df

def VariableTrend(df, indicator):
    
    var_df = df.loc[df['Indicator Code'] == indicator]
  
    var_df.drop(['Indicator Name', 'Indicator Code'],
                axis=1, inplace=True)
    var_df.reset_index(drop=True, inplace=True)
   
    var_df = var_df.T
    var_df = var_df.rename(columns=var_df.iloc[0])
    var_df.drop(labels=['Country Name'], axis=0, inplace=True)
    var_df.rename(columns={'United Kingdom': 'UK',
                           'Russian Federation': 'Russia',
                           'United States': 'US'}, inplace=True)
    var_df.reset_index(inplace=True)
    var_df.drop(var_df.tail(1).index, inplace=True)

    
    columns = list(var_df.columns)
    for col in columns:
        if col != 'Year':
            var_df[col] = var_df[col].astype('float64')

    return var_df

    columns = list(var_df.columns)
    for col in columns:
        if col != 'Year':
        var_df[col] = var_df[col].astype('float64')

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 01 10:42:26 2023

@author: Kavi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stats as st

def PlotCorrMap(df, size=10):
    """This function creates heatmap for the correlation matrix.
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    caxes = ax.matshow(corr, cmap='coolwarm')
    fig.colorbar(caxes)
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

def FilterCountryData(df):
    """
    This function filters the dataframe for the required countries such as US,
    UK, France, Japan, Canada, Germany, Italy and keeps only the required
    fields.
    """
    # filter out data for US, UK, France, India, China, Germany, Russia
    countries = ['United States', 'United Kingdom', 'France', 'Japan',
                 'Canada', 'Germany', 'Italy']
    df = df.loc[df['Country Name'].isin(countries)]

    # required indicators
    indicator = ['SP.URB.TOTL', 'SP.POP.TOTL', 'SH.DYN.MORT',
                 'ER.H2O.FWTL.K3', 'EN.ATM.GHGT.KT.CE', 'EN.ATM.CO2E.KT',
                 'EN.ATM.CO2E.SF.KT', 'EN.ATM.CO2E.LF.KT', 'EN.ATM.CO2E.GF.KT',
                 'EG.USE.ELEC.KH.PC', 'EG.ELC.RNEW.ZS', 'AG.LND.FRST.K2',
                 'AG.LND.ARBL.ZS', 'AG.LND.AGRI.K2']
    df = df.loc[df['Indicator Code'].isin(indicator)]
    return df


def PreprocessData(df):
    """
    This function preprocesses the data by removing the country code and
    fill NaN with zeroes
    """
    df.drop('Country Code', axis=1, inplace=True)
    df.fillna(0, inplace=True)
    return df


def VariableFilter(df, indicator):
    """
    This function filters the trend for particular indicator
    """
    # filter the given indicator
    var_df = df.loc[df['Indicator Code'] == indicator]
    # drops unneccessary columns
    var_df.drop(['Indicator Name', 'Indicator Code'],
                axis=1, inplace=True)
    var_df.reset_index(drop=True, inplace=True)
    # transpose the dataframe and renamed the column names
    var_df = var_df.T
    var_df = var_df.rename(columns=var_df.iloc[0])
    var_df.drop(labels=['Country Name'], axis=0, inplace=True)
    var_df.rename(columns={'United Kingdom': 'UK',
                           'United States': 'US'}, inplace=True)
    var_df.reset_index(inplace=True)
    var_df.drop(var_df.tail(1).index, inplace=True)

    # change the data type of the dataframe to float
    columns = list(var_df.columns)
    for col in columns:
        if col != 'Year':
            var_df[col] = var_df[col].astype('float64')

    return var_df


def HeatmapPreprocess(df, country):
    """
    This function processes the given data frame so that resulting data frame
    can be used to generate the heatmap
    """
    hdf = df.loc[df['Country Name'] == country]

    # filters required indicators
    indicator = ['SP.URB.TOTL', 'SP.POP.TOTL', 'SH.DYN.MORT', 'ER.H2O.FWTL.K3',
                 'AG.LND.FRST.K2', 'AG.LND.ARBL.ZS', 'AG.LND.AGRI.K2',
                 'EN.ATM.CO2E.KT']
    hdf = hdf.loc[hdf['Indicator Code'].isin(indicator)]

    # removes the unnecessary columns
    hdf.drop(['Country Name', 'Indicator Code'], axis=1, inplace=True)
    hdf.reset_index(drop=True, inplace=True)

    # transpose the data
    hdf = hdf.T
    hdf.reset_index(inplace=True)
    hdf = hdf.rename(columns=hdf.iloc[0])
    hdf.drop(0, inplace=True)

    # removes the last three rows as there is no data for those years
    hdf.drop(hdf.tail(3).index, inplace=True)

    # rename the columns names
    hdf.rename(columns={'Indicator Name': 'Year',
                        'Population, total': 'Total Population',
                        'Mortality rate, under-5 (per 1,000 live births)':
                        'Mortality rate',
                        'Annual freshwater withdrawals, total (billion cubic meters)': 'Annual freshwater withdrawals',
                        'CO2 emissions (kt)': 'CO2 emissions',
                        'Forest area (sq. km)': 'Forest area',
                        'Arable land (% of land area)': 'Arable land',
                        'Agricultural land (sq. km)': 'Agricultural land'},
               inplace=True)

    hdf['Year'] = hdf['Year'].astype('int')
    # keeps data after 1990 as the data is available from 1990
    hdf = hdf[hdf['Year'] >= 1990]
    hdf.reset_index(drop=True, inplace=True)
    hdf['Year'] = hdf['Year'].astype('object')

    # type casting of the dataframe columns
    columns = list(hdf.columns)
    for col in columns:
        if col != 'Year':
            hdf[col] = hdf[col].astype('float64')

    return hdf


def ProcessCO2(df):
    """
    This function processes the data for CO2 emission so that grouped bar
    graph can be produced.
    """
    # five years average will be displayed in these years
    years = [1994, 1999, 2004, 2009, 2014, 2019]
    columns_list = ['index', 'US', 'UK', 'France', 'Japan',
                 'Canada', 'Germany', 'Italy']

    # creates empty dataframe
    co2_df = pd.DataFrame(columns=columns_list)

    # iterates over year to produce the data
    for year in years:
        temp = df[(df['index'] >= (year-4)) & (df['index'] <= year)]
        # arithmetic mean for five years are calculated
        temp_series = temp.mean()
        temp_dict = temp_series.to_dict()
        co2_df = co2_df.append(temp_dict, ignore_index=True)

    co2_df['index'] = co2_df['index'] + 2
    co2_df.rename(columns={'index': 'Year'}, inplace=True)
    co2_df['Year'] = co2_df['Year'].astype('int')
    co2_df['Year'] = co2_df['Year'].astype('object')

    co2_df.set_index('Year', inplace=True)
    co2_df = co2_df.T
    co2_df.reset_index(inplace=True)

    co2_df.rename(columns={'Year': ''}, inplace=True)

    return co2_df


# main code
# Reads data from the world bank climate data
df = pd.read_csv('API_19_DS2_en_csv_v2_6183479.csv', skiprows=4)

# filters the data for the required countries
df = FilterCountryData(df)
df = PreprocessData(df)

# greenhouse emission is filtered
greendf = VariableFilter(df, 'EN.ATM.GHGT.KT.CE')

greendf['index'] = greendf['index'].astype('int')
# data is available from 1991 to 2014
greendf = greendf[(greendf['index'] > 1990) & (greendf['index'] < 2015)]
greendf['index'] = pd.to_datetime(greendf['index'], format='%Y')

# line plot for greenhouse emissions
# line growth for various countries
plt.figure(figsize=(8, 8), dpi=500)
plt.plot(greendf['index'], greendf['Canada'], label='Canada')
plt.plot(greendf['index'], greendf['Germany'], label='Germany')
plt.plot(greendf['index'], greendf['France'], label='France')
plt.plot(greendf['index'], greendf['UK'], label='UK')
plt.plot(greendf['index'], greendf['Italy'], label='Italy')
plt.plot(greendf['index'], greendf['Japan'], label='Japan')
plt.plot(greendf['index'], greendf['US'], label='US')
# set labels, title, legend and display them
plt.xlabel('Years')
plt.ylabel('kilo ton')
plt.title('Total greenhouse gas emissions',
          fontweight="bold")
plt.legend()
plt.savefig('GreenHouseEmissions.png')
plt.show()

# heatmap for Germany
hdf_Germany = HeatmapPreprocess(df, 'Germany')
hdf_Germany.describe()

plt.figure(dpi=500)
PlotCorrMap(hdf_Germany)
plt.title('Heatmap for Germany', fontweight="bold")
plt.savefig('hmap_Germany.png')
plt.show()

# heatmap for UK
hdf_UK = HeatmapPreprocess(df, 'United Kingdom')
hdf_UK.describe()

plt.figure(dpi=500)
PlotCorrMap(hdf_UK)
plt.title('Heatmap for UK', fontweight="bold")
plt.savefig('hmap_UK.png')
plt.show()


# grouped bar graph for average CO2 emission
co2 = VariableFilter(df, 'EN.ATM.CO2E.KT')
co2['index'] = co2['index'].astype('int')
co2 = co2[(co2['index'] >= 1990) & (co2['index'] <= 2019)]

# average CO2
avg_df = ProcessCO2(co2)
# display the grpah
plt.figure(dpi=500)
avg_df.plot(x='index',
            kind='bar',
            stacked=False,
            title='Mean CO2 Emission for every five years')
plt.xlabel('Countries')
plt.ylabel('CO2 emission in million kilo Ton')
plt.legend()
plt.savefig('CO2 Emission.png')
plt.show()

# pie chart for the annual fresh water withdrawls
fw_df = VariableFilter(df, 'ER.H2O.FWTL.K3')
fw_df.rename(columns={'index': 'Year'}, inplace=True)
fw_df["Year"] = fw_df['Year'].astype('int')
fw_df = fw_df.loc[fw_df['Year'] == 2019]
fw_df.drop(['Year'], axis=1, inplace=True)
fw_df.reset_index(drop=True, inplace=True)
fw_df = fw_df.T
fw_df.reset_index(inplace=True)
fw_df.rename(columns={0: 'Freshwater withdrawals',
                       'index': 'Countries'},
              inplace=True)

# display the graph
plt.figure(dpi=500)
plt.pie(fw_df['Freshwater withdrawals'], labels=fw_df['Countries'])
plt.title('Annual freshwater withdrawals in 2019', fontweight="bold")
plt.savefig('Freshwater.png')
plt.show()

# skewness and Kurtosis
coun_df = VariableFilter(df, 'EG.USE.ELEC.KH.PC')
UK_df = coun_df[['UK']]
index = coun_df[['index']]
    
#removes the last eight data points as they does not have data
uk_df = UK_df[1:-8]
index = index[1:-8]

plt.figure(figsize=(8, 8), dpi=500)
plt.plot(index, uk_df)
plt.xlabel('Years')
plt.ylabel('Kilo Watt Hour')
plt.title('Electric power consumption per capita for the UK',
          fontweight="bold")
plt.legend()
plt.savefig('ElectricPowerConsumption.png')
plt.show()

# Skew for electricity per capita consumption for the UK
st.skew(uk_df)

# Kurtosis for electricity per capita consumption for the UK
st.kurtosis(uk_df)

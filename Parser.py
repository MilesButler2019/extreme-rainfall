
import pandas as pd
import pymannkendall as mk
from ftplib import FTP
import numpy as np 
import sys
import statsmodels.api as sm
import time 
import os
import csv


def read_ghcn(file):
    filename = 'ghcnd_all/' + file
    #Defines Columns
    data_header_col_specs = [(0,  11),(11, 15),(15, 17),(17, 21)]
    data_col_specs = [[(21 + i * 8, 26 + i * 8),(26 + i * 8, 27 + i * 8),(27 + i * 8, 28 + i * 8),(28 + i * 8, 29 + i * 8)]for i in range(31)]
    metadata_names = ["ID","LATITUDE","LONGITUDE","ELEVATION","STATE","NAME","GSN FLAG","HCN/CRN FLAG","WMO ID"]
    data_header_names = ["ID","YEAR","MONTH","ELEMENT"]
    data_col_names = [["VALUE" + str(i + 1),"MFLAG" + str(i + 1),"QFLAG" + str(i + 1),"SFLAG" + str(i + 1)]for i in range(31)]
    # Join sub-lists
    data_col_names = sum(data_col_names, [])
    data_col_specs = sum(data_col_specs, [])
    #####################################
    
    reader = pd.read_fwf(filename,colspecs=data_header_col_specs + data_col_specs,names=data_header_names + data_col_names)
    arr = []
    for i in range(1,32):
        arr.append("VALUE" + str(i))
    #Filters only Precipitation events
    correct_element = reader[reader['ELEMENT']=="PRCP"]
    #Removes Flags 
    clean_data_temp_1 = correct_element[arr]


    #Filters Negitive Values 
    no_negitives = clean_data_temp_1[clean_data_temp_1[arr]>= 0]

    # #Add ID,Year,Month and Element to the dataframe 
    info = (correct_element[['ID','YEAR','MONTH','ELEMENT']])

    #print(info)

    #Data is in tenths of milimeters

    clean_data = info.join(no_negitives)
    #return(len(clean_data.YEAR.unique()))
    return(clean_data)
def find_thirty_year_data(directory): 
    counter = 0
    thirty_year_data = []
    start_time = time.time()
    with open('30yeardata.csv', mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for filename in os.listdir(directory):
            if filename.endswith(".dly"):
                counter += 1
                if read_ghcn(filename) >= 30:
                    thirty_year_data.append(filename)
                    employee_writer.writerow([filename])
                #    print("Has read: ",counter,"Files")
    print("Total Files in the directory",counter)
    print("My program took", time.time() - start_time, "to run")
    return(thirty_year_data)
#find_thirty_year_data('ghcnd_all')



#print(read_ghcn("US1FLCY0024.dly"))

def reshape_data(df):
    #Reshaped data into a workable form from wide to long
    arr = []
    for i in range(1,32):
        arr.append("VALUE" + str(i))
    long_clean_data = df.melt(id_vars=['ID',"MONTH",'YEAR','ELEMENT'],value_vars = [i for i in arr])
    #Remove
    long_clean_data['DAY'] = long_clean_data['variable'].map(lambda x: x.lstrip('VALUE').rstrip('aAbBcC'))
    long_clean_data.drop(columns='variable')
    #Rearranges columns index's  
    cleanest_data = long_clean_data[["ID","MONTH","DAY",'YEAR','value']]
    cleanest_data=(cleanest_data.sort_values(['YEAR','MONTH']))
    #Removes_Null vlaues
    cleanest_data = cleanest_data[cleanest_data['value'].notnull()]
    cleanest_data['Day'] =  pd.to_numeric(cleanest_data['DAY'])
    cleanest_data.drop(columns='DAY')
    cleanest_data['value'] = pd.to_numeric(cleanest_data['value'])
    
    cleanest_data['Date']= pd.to_datetime(cleanest_data[['Day','MONTH','YEAR']],dayfirst=True)
    cleanest_data = cleanest_data[['ID','MONTH','Day','YEAR','value','Date']]
    return(cleanest_data)
    #print(cleanest_data.dtypes)

reshape_data(read_ghcn("USW00093822.dly"))

def top_value_annual(df):
    max_per_year = df.loc[df.groupby("YEAR")["value"].idxmax()]
    return max_per_year


def read_giss_JD(file):
    filename ='ghcnd_all/' + file
    temperature_data = pd.read_csv(filename)
    temp_avg = temperature_data[['Year','J-D']]
    temp_avg.drop(temp_avg.tail(1).index,inplace=True)
    temp_avg['J-D']= pd.to_numeric(temp_avg['J-D'])
    temp_avg.columns = ['YEAR','J-D']
    # temp_avg['YEAR'] = temp_avg["Year"]
    # temp_avg.drop(columns='Year')
    
    return temp_avg
#print(read_giss_JD('GISS.csv'))

def frequency_func_of_temp(df_rain,df2_temp):
    mergedStuff = pd.merge(df_rain, df2_temp, on=['YEAR'], how='inner')
    #mergedStuff['Func_of_temp'] = mergedStuff['J-D']*(mergedStuff['value'])
    #mergedStuff['J-D'].astype('float64').dtypes
    #print(mergedStuff[['J-D','value']].dtypes)
    final = mergedStuff[['J-D','value']]
    return final
   

#print(frequency_func_of_temp(read_giss_JD('GISS.csv'),top_value_annual(reshape_data(read_ghcn('USW00093822.dly')))))

def trend_test(df):
    result = mk.original_test(df)
    print(result)
#print(trend_test(frequency_func_of_temp(read_giss_JD('GISS.csv'),top_value_annual(reshape_data(read_ghcn('USW00093822.dly'))))))
# Data generation for analysis
#data = np.random.rand(200,1)

#result = mk.original_test(data)
#print(result)
def linear_regress(df):
    #INDEPENDENT VARIABLE 
    X = df['J-D']
    #Dependent Variable 
    y = df['value']
    #adds constant 
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    #print(model.summary())
    print(model.params)
   

def find_slope_of_all_data(thirty_year_data):
    linear_regress(frequency_func_of_temp(read_giss_JD('GISS.csv'),top_value_annual(reshape_data(read_ghcn('USW00093822.dly')))))










# ftp_path_dly_all = '/pub/data/ghcn/daily/all/'

# def connect_to_ftp():
#     """
#     Get FTP server and file details
#     """
#     ftp_path_root = 'ftp.ncdc.noaa.gov'
#     # Access NOAA FTP server
#     ftp = FTP(ftp_path_root)
#     message = ftp.login()  # No credentials needed
#     print(message)
#     return ftp


# #Reader = pd.read_fwf("/ghcn_all/CA1NS000049.dly")


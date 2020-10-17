#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:43:48 2020

@author: tommasobassignana
"""
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from sklearn import preprocessing

del(xml_file, xroot, xtree)
data = df
del(df)

def resample(data, freq):
    """
    :param data: dataframe
    :param freq: sampling frequency
    :return: resampled data between the the first day at 00:00:00 and the last day at 23:60-freq:00 at freq sample frequency
    """
    start = data.datetime.iloc[0].strftime('%Y-%m-%d') + " 00:00:00"
    end = datetime.strptime(data.datetime.iloc[-1].strftime('%Y-%m-%d'), "%Y-%m-%d") + timedelta(days=1) - timedelta(
        minutes=freq)
    index = pd.period_range(start=start,
                            end=end,
                            freq=str(freq) + 'min').to_timestamp()
    data = data.resample(str(freq) + 'min', on="datetime").agg({'glucose': np.mean, 'CHO': np.sum, "insulin": np.sum})
    data = data.reindex(index=index)
    data = data.reset_index()
    data = data.rename(columns={"index": "datetime"})
    return data

data_resampled = resample(data, 5)

#fill na's

data_resampled["glucose"].interpolate(method = "polynomial", order = 3, inplace = True)#impostare limit se no vengono dei valori di glucosio negativi

#fill na's with zeros
for col in data_resampled.columns:
    if "insulin" in col or "CHO" in col:
        data_resampled[col] = data_resampled[col].fillna(0)
        
#train test split
n = len(data_resampled)
n
train_df = data_resampled[0:int(n*0.7)]
test_df = data_resampled[int(n*0.7):int(n*1)]
train_df.shape
test_df.shape

#standardizzazione
x_cols = [x for x in train_df.columns if x != 'datetime']
min_max_scaler = preprocessing.MinMaxScaler()# 0 1 scale
train_df = min_max_scaler.fit_transform(train_df[x_cols])
#handle sparses data well, but re read documentation
#how to apply the same transformation to the tast set
#check if correct!
test_df = min_max_scaler.transform(test_df[x_cols])
#recreate the dataframe
train_df = pd.DataFrame(data=train_df, columns=x_cols)
test_df = pd.DataFrame(data=test_df, columns=x_cols)

train_df.shape
test_df.shape

#add datetime again?
train_df["datetime"] = pd.DatetimeIndex(data_resampled.iloc[:int(len(train_df["glucose"])), 0].values)

test_df["datetime"] = pd.DatetimeIndex(data_resampled.iloc[len(train_df["glucose"]):n, 0].values)

train_df = train_df[['datetime',"glucose", "CHO", "insulin"]]
test_df = test_df[['datetime',"glucose", "CHO", "insulin"]]

train_df.shape
test_df.shape

#da togliere
#train_df.dropna(inplace = True)
#test_df.dropna(inplace = True)

train_df.shape
test_df.shape

#test_df.to_csv("test_longitudinal", index = False)


def create_samples(data, ph, hist):
    n_samples = data.shape[0] - ph - hist + 1
    # number of rows
    y = data.loc[ph + hist - 1:, "glucose"].values.reshape(-1, 1)
    d = pd.DatetimeIndex(data.loc[ph + hist - 1:, "datetime"].values)
       # t = np.concatenate([np.arange(day_len) for _ in range(len(data) // day_len)], axis=0)[ph + hist - 1:].reshape(-1, 1)
    g = np.array([data.loc[i:i + n_samples - 1, "glucose"] for i in range(hist)]).transpose()
    c = np.array([data.loc[i:i + n_samples - 1, "CHO"] for i in range(hist)]).transpose()
    i = np.array([data.loc[i:i + n_samples - 1, "insulin"] for i in range(hist)]).transpose()
    
    new_columns = np.r_[["glucose_" + str(i) for i in range(hist)], ["CHO_" + str(i) for i in range(hist)], [
            "insulin_" + str(i) for i in range(hist)], ["y"]]
    len(new_columns)
    new_data = pd.DataFrame(data=np.c_[g, c, i, y], columns=new_columns)
    new_data["datetime"] = d
    new_data = new_data.loc[:, np.r_[["datetime"], new_columns]]  # reorder the columns, with datetime first
    return new_data


train = create_samples(train_df, 6, 24)
train.dropna(inplace = True)   


test = create_samples(test_df, 6, 24)
test.dropna(inplace = True) 
#test.to_csv("test_reshaped", index = False)



#del(col, data, data_resampled, min_max_scaler, n, train_df, test_df, x_cols)



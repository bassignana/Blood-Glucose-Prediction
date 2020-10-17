#!/usr/bin/env python3
# -*- coding: utf-8 -*-
del(xml_file, xroot, xtree)
 
data = df
del(df)

import pandas as pd
import numpy as np
from datetime import timedelta, datetime


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

#custom samples
# v = np.arange(len(data_resampled["glucose"]))
# data_resampled["iter_index"] = v
# import itertools as it

# streams = it.tee(data_resampled["iter_index"], 12*24)
# slices = zip(*[it.islice(stream, i, None, 1) for stream, i in zip(streams, it.count(step=1))])
# X= list(slices)
# X= np.asarray(X)
# X[0]
# min(X[0])
# max(X[0])
# data_resampled.iloc[min(X[0]):max(X[0])]



# def moving_window(x, length, step=1):
#     streams = it.tee(x, length)
#     return zip(*[it.islice(stream, i, None, step) for stream, i in zip(streams, it.count(step=step))])
# x_=list(moving_window(x, 3))
# x_=np.asarray(x_)
# print(x_)

#fill na's

data_resampled["glucose"].interpolate(method = "polynomial", order = 3, inplace = True)#impostare limit se no vengono dei valori di glucosio negativi

for col in data_resampled.columns:
    if "insulin" in col or "CHO" in col:
        data_resampled[col] = data_resampled[col].fillna(0)
        
#train test split
n = len(data_resampled)
train_df = data_resampled[0:int(n*0.7)]
test_df = data_resampled[int(n*0.7):]

#standardizzazione
from sklearn import preprocessing
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
#add datetime again
train_df["datetime"] = pd.DatetimeIndex(data_resampled.iloc[:int(len(train_df["glucose"])), 0].values)

test_df["datetime"] = pd.DatetimeIndex(data_resampled.iloc[int(len(train_df["glucose"])):, 0].values)

train_df = train_df[['datetime',"glucose", "CHO", "insulin"]]
test_df = train_df[['datetime',"glucose", "CHO", "insulin"]]

# def create_samples(data, ph, hist, day_len):
#     """
#     Create samples consisting in glucose, insulin and CHO histories (past hist-length values)
#     :param data: dataframe
#     :param ph: prediction horizon in minutes in sampling frequency scale
#     :param hist: history length in sampling frequency scale
#     :param day_len: length of day in sampling frequency scale
#     :return: dataframe of samples
#     """
#     n_samples = data.shape[0] - ph - hist + 1

#     y = data.loc[ph + hist - 1:, "glucose"].values.reshape(-1, 1)
#     d = pd.DatetimeIndex(data.loc[ph + hist - 1:, "datetime"].values)
#     t = np.concatenate([np.arange(day_len) for _ in range(len(data) // day_len)], axis=0)[ph + hist - 1:].reshape(-1, 1)
#     g = np.array([data.loc[i:i + n_samples - 1, "glucose"] for i in range(hist)]).transpose()
#     c = np.array([data.loc[i:i + n_samples - 1, "CHO"] for i in range(hist)]).transpose()
#     i = np.array([data.loc[i:i + n_samples - 1, "insulin"] for i in range(hist)]).transpose()

#     new_columns = np.r_[["time"], ["glucose_" + str(i) for i in range(hist)], ["CHO_" + str(i) for i in range(hist)], [
#         "insulin_" + str(i) for i in range(hist)], ["y"]]
#     new_data = pd.DataFrame(data=np.c_[t, g, c, i, y], columns=new_columns)
#     new_data["datetime"] = d
#     new_data = new_data.loc[:, np.r_[["datetime"], new_columns]]  # reorder the columns, with datetime first

#     return new_data

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

del(col, data, data_resampled, min_max_scaler, n, train_df, test_df, x_cols)
# def fill_nans(data, day_len, n_days_test):
#     """
#     Fill NaNs inside the dataframe of samples following:
#     - CHO and insulin values are filled with 0
#     - glucose history are interpolated (linearly) when possible and extrapolated if not
#     :param data: sample dataframe
#     :param day_len: length of day, scaled to sampling frequency
#     :param n_days_test: number of test days
#     :return: cleaned sample dataframe
#     """
#     data_nan = data.copy()

#     # fill insulin and CHO nans with 0
#     for col in data.columns:
#         if "insulin" in col or "CHO" in col:
#             data[col] = data[col].fillna(0)
# #penso si possa fare meglio...


#     # fill glucose nans
#     g_col = [col for col in data.columns if "glucose" in col]
#     g_mean = data.y.mean()
#     for i in range(len(data.index)):
#         g_i = data.loc[i, g_col]
#         if g_i.notna().all():  # no nan
#             pass
#         elif g_i.isna().all():  # all nan
#             if i > len(data) - n_days_test * day_len + 1:
#                 last_known_gs = data_nan.loc[:i - 1, "glucose_0"][data_nan.loc[:i - 1, "glucose_0"].notna()]
#                 if len(last_known_gs) >= 1:
#                     for col in g_col:
#                         # data.loc[i,col] = last_known_gs.iloc[-1]
#                         data.loc[i, col] = g_mean
#                 else:
#                     for col in g_col:
#                         data.loc[i, col] = g_mean
#         else:  # some nan
#             # compute insample nan indices, and last outsample + insample nonnan indices
#             isna_idx = i + np.where(g_i.isna())[0]
#             notna_idx = i + np.where(g_i.notna())[0]
#             if data_nan.loc[:i - 1, "glucose_0"].notna().any():
#                 mask = data_nan.loc[:i - 1, "glucose_0"].notna().values
#                 notna_idx = np.r_[data_nan.loc[:i - 1, "glucose_0"][mask].index[-2:], notna_idx]

#             # for all nan
#             for isna_i in isna_idx:
#                 # get the two closest non nan values
#                 idx_diff = notna_idx - isna_i

#                 if np.any(idx_diff > 0) and np.any(idx_diff < 0):
#                     # we got a start and an end
#                     start = notna_idx[np.where(idx_diff < 0, idx_diff, -np.inf).argmax()]
#                     end = notna_idx[np.where(idx_diff > 0, idx_diff, np.inf).argmin()]

#                     start_idx = _compute_indexes(i, start, len(data_nan))
#                     end_idx = _compute_indexes(i, end, len(data_nan))

#                     start_val = data_nan.loc[start_idx]
#                     end_val = data_nan.loc[end_idx]

#                     # interpolate between them
#                     rate = (end_val - start_val) / (end - start)
#                     data.loc[i, g_col[isna_i - i]] = data_nan.loc[start_idx] + rate * (isna_i - start)
#                 elif np.any(idx_diff > 0):
#                     # we only have end(s)
#                     # backward extrapolation - only used in very first day where there is no start
#                     if len(idx_diff) >= 2:
#                         # we have two last values so we can compute a rate
#                         end1, end2 = notna_idx[0], notna_idx[1]
#                         [end1_idx, end2_idx] = [_compute_indexes(i, _, len(data_nan)) for _ in [end1, end2]]
#                         end1_val, end2_val = data_nan.loc[end1_idx], data_nan.loc[end2_idx]
#                         rate = (end2_val - end1_val) / (end2 - end1)
#                         data.loc[i, g_col[isna_i - i]] = data_nan.loc[end1_idx] - rate * (end1 - isna_i)
#                     else:
#                         # we have only one value so we cannot compute a rate
#                         end = notna_idx[0]
#                         end_idx = _compute_indexes(i, end, len(data_nan))
#                         end_val = data_nan.loc[end_idx]
#                         data.loc[i, g_col[isna_i - i]] = end_val
#                 elif np.any(idx_diff < 0):
#                     # forward extrapolation
#                     if len(idx_diff) >= 2:
#                         end1, end2 = notna_idx[-2], notna_idx[-1]
#                         [end1_idx, end2_idx] = [_compute_indexes(i, _, len(data_nan)) for _ in [end1, end2]]
#                         end1_val, end2_val = data_nan.loc[end1_idx], data_nan.loc[end2_idx]
#                         rate = (end2_val - end1_val) / (end2 - end1)
#                         data.loc[i, g_col[isna_i - i]] = data_nan.loc[end1_idx] - rate * (end1 - isna_i)
#                     else:
#                         # we only have one value, so we cannot compute a rate
#                         last_val = g_i[g_i.notna()][-1]
#                         data.loc[i, g_col[isna_i - i]] = last_val

#     return data

# def _compute_indexes(i, index, len):
#     if index >= len:
#         return (i, "glucose_" + str(index - i))
#     else:
#         return (index, "glucose_0")

#nonan = fill_nans(sample1, 12*24, 5)



# class expanding_window(object):
#     '''	
#     Parameters 
#     ----------
    
#     Note that if you define a horizon that is too far, then subsequently the split will ignore horizon length 
#     such that there is validation data left. This similar to Prof Rob hyndman's TsCv 
    
    
#     initial: int
#         initial train length 
#     horizon: int 
#         forecast horizon (forecast length). Default = 1
#     period: int 
#         length of train data to add each iteration 
#     '''
    

#     def __init__(self,initial= 1,horizon = 1,period = 1):
#         self.initial = initial
#         self.horizon = horizon 
#         self.period = period 


#     def split(self,data):
#         '''
#         Parameters 
#         ----------
        
#         Data: Training data 
        
#         Returns 
#         -------
#         train_index ,test_index: 
#             index for train and valid set similar to sklearn model selection
#         '''
#         self.data = data
#         self.counter = 0 # for us to iterate and track later 


#         data_length = data.shape[0] # rows 
#         data_index = list(np.arange(data_length))
         
#         output_train = []
#         output_test = []
#         # append initial 
#         output_train.append(list(np.arange(self.initial)))
#         progress = [x for x in data_index if x not in list(np.arange(self.initial)) ] # indexes left to append to train 
#         output_test.append([x for x in data_index if x not in output_train[self.counter]][:self.horizon] )
#         # clip initial indexes from progress since that is what we are left 
         
#         while len(progress) != 0:
#             temp = progress[:self.period]
#             to_add = output_train[self.counter] + temp
#             # update the train index 
#             output_train.append(to_add)
#             # increment counter 
#             self.counter +=1 
#             # then we update the test index 
            
#             to_add_test = [x for x in data_index if x not in output_train[self.counter] ][:self.horizon]
#             output_test.append(to_add_test)

#             # update progress 
#             progress = [x for x in data_index if x not in output_train[self.counter]]	
            
#         # clip the last element of output_train and output_test
#         output_train = output_train[:-1]
#         output_test = output_test[:-1]
        
#         # mimic sklearn output 
#         index_output = [(train,test) for train,test in zip(output_train,output_test)]
        
#         return index_output
        

# # example
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])

# y = np.array([1, 2, 3, 4, 5, 6])
# tscv = expanding_window()
# for train_index, test_index in tscv.split(X):
#     print(train_index)
#     print(test_index)

























# import sklearn.model_selection as sk_model_selection
# import misc.constants as cs

# def split(data, day_len, test_n_days, cv_factor):
#     """
#     Split samples into training, validation, and testing days. Testing days are the last test_n_days days, and training
#     and validation sets are created by permuting of splits according to the cv_factor.
#     :param data: dataframe of samples
#     :param day_len: length of day in freq minutes
#     :param test_n_days: number of testing days
#     :param cv_factor: cross validation factor
#     :return: training, validation, testing samples folds
#     """
#     # the test split is equal to the last test_n_days days of data
#     test = [data.iloc[-test_n_days * day_len:].copy()]

#     # train+valid samples = all minus first and test days
#     fday_n_samples = data.shape[0] - (data.shape[0] // day_len * day_len)
#     train_valid = data.iloc[fday_n_samples:-test_n_days * day_len].copy()

#     # split train_valid into cv_factor folds for inner cross-validation
#     n_days = train_valid.shape[0] // day_len
#     days = np.arange(n_days)

#     kf = sk_model_selection.KFold(n_splits=cv_factor, shuffle=True, random_state=cs.seed)

#     train, valid = [], []
#     for train_idx, valid_idx in kf.split(days):
#         def get_whole_day(data, i):
#             return data[i * day_len:(i + 1) * day_len]

#         train.append(pd.concat([get_whole_day(train_valid, i) for i in train_idx], axis=0, ignore_index=True))
#         valid.append(pd.concat([get_whole_day(train_valid, i) for i in valid_idx], axis=0, ignore_index=True))

#     return train, valid, test

# train, valid, test = split(nonan, 12*24, 5, 5)
# train[1]
# valid[1]
# test
# def remove_nans(data):
#     """
#     Remove samples that still have NaNs (in y column mostly)
#     :param data: dataframe of samples
#     :return: no-NaN dataframe
#     """
#     new_data = []
#     for df in data:
#         new_data.append(df.dropna())
#     return new_data

# [train, valid, test] = [remove_nans(set) for set in [train, valid, test]]

# from sklearn.preprocessing import StandardScaler

# def standardize(train, valid, test):
#     """
#     Standardize (zero mean and unit variance) the sets w.r.t to the training set for every fold
#     :param train: training sample fold
#     :param valid: validation sample fold
#     :param test: testing sample fold
#     :return: standardized training, validation, and testing sets
#     """
#     columns = train[0].columns.drop("datetime")
#     train_scaled, valid_scaled, test_scaled, scalers = [], [], [], []
#     for i in range(len(train)):
#         scaler = StandardScaler()

#         # standardize the sets (-> ndarray) without datetime
#         train_i = scaler.fit_transform(train[i].drop("datetime", axis=1))
#         valid_i = scaler.transform(valid[i].drop("datetime", axis=1))
#         test_i = scaler.transform(test[0].copy().drop("datetime", axis=1))

#         # recreate dataframe
#         train_i = pd.DataFrame(data=train_i, columns=columns)
#         valid_i = pd.DataFrame(data=valid_i, columns=columns)
#         test_i = pd.DataFrame(data=test_i, columns=columns)

#         # add datetime
#         train_i["datetime"] = pd.DatetimeIndex(train[i].loc[:, "datetime"].values)
#         valid_i["datetime"] = pd.DatetimeIndex(valid[i].loc[:, "datetime"].values)
#         test_i["datetime"] = pd.DatetimeIndex(test[0].loc[:, "datetime"].values)

#         # reorder
#         train_i = train_i.loc[:, train[i].columns]
#         valid_i = valid_i.loc[:, valid[i].columns]
#         test_i = test_i.loc[:, test[0].columns]

#         # save
#         train_scaled.append(train_i)
#         valid_scaled.append(valid_i)
#         test_scaled.append(test_i)

#         scalers.append(scaler)

#     return train_scaled, valid_scaled, test_scaled, scalers

# train, valid, test, scalers = standardize(train, valid, test)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates
from statsmodels.tsa import stattools
df.glucose.mean()
df.mean()

df.glucose.min()
df.glucose.max()
df.glucose.count()
df.glucose.cummax()

#dimensioni prima del resampling
df.shape

#resampling a dataFrame
df = df.set_index("datetime")
df_resampled = df.resample('5min')
df_resampled = df.resample('5min').pad()
df_resampled = df.asfreq('5min')
df_resampled = df.resample('5min').fillna(method = "ffill", limit = 5 )
#limit = 5 intende cinque periodi di 5min, se combiassi a 7 minuti sarebbero 5 periodi da 7 minuti ecc
df_resampled = df.resample('30min').mean()

#dimensioni dopo il resampling
df_resampled.shape

#rolling windows function
#for plotting
x = [datetime.datetime.fromtimestamp(element[0]) for element in df_resampled]
x.plot(col = "orange")

r = df.rolling(window=20)
r.mean()

#autocorrelation
result_acf = stattools.acf(df_resampled["glucose"].to_numpy()) 
type(result_acf)

result_acf = np.log(result_acf)
result_acf_diff = result_acf - result_acf.shift()#non funge
result_acf_diff.plot()

type(df["glucose"])
type(df_resampled["glucose"])
type(df_resampled["glucose"].to_numpy())
type(df["glucose"].to_numpy())

df = pd.DataFrame(df_resampled["glucose"].to_numpy())
df.shift()

first_diff_df = df - df.shift()
first_diff_df.plot()


result_acf = stattools.acf(first_diff_df) 
type(result_acf)
plt.plot(result_acf)

#
grid = np.linspace(0, 720, 500)
noise = np.random.rand(500)
result_curve =  noise
plt.plot(grid, result_curve)
type(result_curve)

acf_result = stattools.acf(result_curve)
type(acf_result)

#shift deve essere applicato ad un dataframe





















#!/usr/bin/env python

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import math

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

if __name__ == "__main__":
    data = "./data/"
    df = load_pickle("df_processed.pkl")
    ### split into location-wise time-series of demands
    cols = df["geohash6"].unique()
    start = df.index.min()
    indexes = [start + 15*i for i in\
               range(int((df.index.max() - start)/15) + 1)]
    df_series = pd.DataFrame(index=indexes,columns=cols)
    for col in cols:
        df_series[col] = df[df["geohash6"] ==  col]["demand"]
    df_series.fillna(value=df["demand"].min()/100, inplace=True)
    save_as_pickle("df_series.pkl", df_series)
    
    ### convert to relative demands rather than absolute
    df_series_relative = df_series - df_series.shift(periods=1)
    save_as_pickle("df_series_relative.pkl", df_series_relative)
    
    ### distribution of relative demands
    d = []
    for col in df_series_relative.columns:
        d.extend(list(set(df_series_relative[col])))
    d = list(set(d)); d.sort(); d = d[1:]
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.scatter([i for i in range(len(d))],d)
    ax.set_title("Relative Demand distribution", fontsize=20)
    ax.set_ylabel("Relative Demand", fontsize=17)
    ax.set_xlabel("Data point", fontsize=17)
    
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.scatter([i for i in range(len(d))],[math.log(i) for i in d])
    ax.set_title("Log-Relative Demand distribution", fontsize=20)
    ax.set_ylabel("Log-Relative Demand", fontsize=17)
    ax.set_xlabel("Data point", fontsize=17)
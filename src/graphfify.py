#!/usr/bin/env python

import pandas as pd
import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from geohash import decode
import geopy.distance as gdist

def save_as_pickle(filename, data):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def process_datetime(day, timestamp):
    return (1440*day + 60*timestamp.hour + timestamp.minute)

def compute_dtw(df1,df2):
    distance, _ = fastdtw(np.array([df1.index, df1["demand"]/df1["demand"].max()]).T, \
                         np.array([df2.index, df2["demand"]/df2["demand"].max()]).T, \
                         dist=euclidean)
    return distance

def get_dtw_between_locs(df):
    dum = []; dist = []; counter = 0; total_len = len(df["geohash6"].unique())
    for loc1 in df["geohash6"].unique():
        for loc2 in df["geohash6"].unique():
            if (counter % 50000) == 0:
                print(str(100*counter/total_len) + "% completed")
            counter += 1
            if (loc1 != loc2) and (loc1,loc2) not in dum:
                dum.append((loc2,loc1))
                df1 = df[df["geohash6"] == loc1]; df2 = df[df["geohash6"] == loc2]
                if (len(df1) > 10) and (len(df2) > 10):
                    dist.append((loc1, loc2, compute_dtw(df1,df2)))
    return dist

def build_loc_edges(locs,threshold_distance=2):
    loc_edges = []; l = list(locs.keys()); length = len(l); counter = 0
    for i1, loc1 in enumerate(l):
        #print(i1,loc1)
        if (i1 + 1) == length:
            break
        for loc2 in l[(i1+1):]:
            if counter % 100000 == 0:
                print(counter/(length**2))
            counter += 1
            if gdist.geodesic(locs[loc1],locs[loc2]).km <= threshold_distance:
                loc_edges.append((loc1,loc2))
    return loc_edges

if __name__ == "__main__":
    data = "./data/"
    df = pd.read_csv(os.path.join(data, "training.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%H:%M")
    df["minutes"] = df.apply(lambda x: process_datetime(x["day"], x["timestamp"]), axis=1)
    df.drop(["day", "timestamp"], axis=1, inplace=True)
    df.set_index("minutes",inplace=True)
    df.sort_index(ascending=True, inplace=True)
    save_as_pickle("df_processed.pkl", df)
    
    df1 = df[df["geohash6"] == "qp03wc"]
    df2 = df[df["geohash6"] == "qp03pn"]
    
    plt.scatter(df1.index, df1["demand"], c="red")
    plt.scatter(df2.index, df2["demand"], c="blue")
    
    ##### get coordinates of each unique location and plot
    locs = {}
    for loc in df["geohash6"].unique():
        locs[loc] = decode(loc)
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    for key in locs.keys():
        ax.scatter(locs[key][0], locs[key][1])
    ax.set_title("Map of locations", fontsize=20)
    ax.set_ylabel("Longitude", fontsize=17)
    ax.set_xlabel("Latitude", fontsize=17)
    
    ##### print distribution of each location
    loc_dist = df["geohash6"].value_counts()
    
    ### demand distribution
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.scatter([i for i in range(len(df["demand"]))],df["demand"].sort_values())
    ax.set_title("Demand distribution", fontsize=20)
    ax.set_ylabel("Demand", fontsize=17)
    ax.set_xlabel("Data point", fontsize=17)
    
    ### build location edges if geodesic distance less than threshold distance
    loc_edges = build_loc_edges(locs)
    save_as_pickle("loc_edges.pkl", loc_edges)
    
    ### build graph
    G = nx.Graph()
    G.add_nodes_from(locs.keys())
    G.add_edges_from(loc_edges)
    save_as_pickle("Graph.pkl", G)
    
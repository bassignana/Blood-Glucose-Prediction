#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:37:37 2020

@author: tommasobassignana
"""

import pandas as pd
import xml.etree.ElementTree as et

xml_file = "/Users/tommasobassignana/Desktop/GLYFE-master/data/ohio/OhioT1DM-training/559-ws-training.xml"
xtree = et.parse(xml_file)
xroot = xtree.getroot()

def extract_attribute_values(root, child_index):
    """
    to add : element = "event"
    extract attribute values inside the specified element 
    :param root: tree root
    :param fchild_index:  index of the child element from which i want the attributes names
    :return:
    """
    for event in root[child_index].iter("event"):
        yield list(event.attrib.values())

def extract_attribute_names(root, child_index):
    """
    extract the name of the attributes from xml tree element
    :param root: root node
    :param child_index:  index of the child element from which i want the attributes names
    :return:
    """
    return list(root[child_index][0].attrib.keys())


def create_glucose_df_from_root(xroot):
    """
    Extract glucose values from xml
    :param xml:
    :return: glucose dataframe
    """
    labels = extract_attribute_names(xroot, child_index=0)
    glucose = list(extract_attribute_values(xroot, child_index=0))
    glucose_df = pd.DataFrame(data=glucose, columns=labels)
    glucose_df["ts"] = pd.to_datetime(glucose_df["ts"], format="%d-%m-%Y %H:%M:%S")
    glucose_df["value"] = glucose_df["value"].astype("float")
    glucose_df.rename(columns={'ts': 'datetime', 'value': 'glucose'}, inplace=True)

    return glucose_df


def create_CHO_df_from_root(xroot):
    """
    Extract CHO values from xml
    :param xml:
    :return: CHO dataframe
    """
    labels = extract_attribute_names(xroot, child_index=5)
    CHO = list(extract_attribute_values(xroot, child_index=5))
    CHO_df = pd.DataFrame(data=CHO, columns=labels)
    CHO_df.drop("type", axis=1, inplace=True)
    CHO_df["ts"] = pd.to_datetime(CHO_df["ts"], format="%d-%m-%Y %H:%M:%S")
    CHO_df["carbs"] = CHO_df["carbs"].astype("float")
    CHO_df.rename(columns={'ts': 'datetime', 'carbs': 'CHO'}, inplace=True)
    return CHO_df


def create_insuline_df_from_root(xroot):
    """
    Extract insulin values from xml
    :param xml:
    :return: insulin dataframe
    """
    labels = extract_attribute_names(xroot, child_index=4)
    insulin = list(extract_attribute_values(xroot, child_index=4))
    insulin_df = pd.DataFrame(data=insulin, columns=labels)
    for col in ["ts_end", "type", "bwz_carb_input"]:
        insulin_df.drop(col, axis=1, inplace=True)
    insulin_df["ts_begin"] = pd.to_datetime(insulin_df["ts_begin"], format="%d-%m-%Y %H:%M:%S")
    insulin_df["dose"] = insulin_df["dose"].astype("float")
    insulin_df.rename(columns={'ts_begin': 'datetime', 'dose': 'insulin'}, inplace=True)
    return insulin_df

    
def compose_final_df(xroot):
    """
    extract glucose, CHO, and insulin from xml and merge the data
    :param xml:
    :return: dataframe
    """
    glucose_df = create_glucose_df_from_root(xroot)
    CHO_df = create_CHO_df_from_root(xroot)
    insulin_df = create_insuline_df_from_root(xroot)

    df = pd.merge(glucose_df, CHO_df, how="outer", on="datetime")
    df = pd.merge(df, insulin_df, how="outer", on="datetime")
    df = df.sort_values("datetime")

    return df

df = compose_final_df(xroot)

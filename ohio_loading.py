# -*- coding: utf-8 -*-
# da https://medium.com/@robertopreste/from-xml-to-pandas-dataframes-9292980b1c1c
import pandas as pd
import xml.etree.ElementTree as et

def parse_XML(xml_file, df_cols): 
    """
    The first element of df_cols is supposed to be the identifier 
    variable, which is an attribute of each node element in the 
    XML data; other features will be parsed from the text content 
    of each sub-element. 
    """
    
    xtree = et.parse(xml_file)
    xroot = xtree.getroot()
    rows = []
    
    for node in xroot: 
        res = []
        res.append(node.attrib.get(df_cols[0]))
        for el in df_cols[1:]: 
            if node is not None and node.find(el) is not None:
                res.append(node.find(el).text)
            else: 
                res.append(None)
        rows.append({df_cols[i]: res[i] 
                     for i, _ in enumerate(df_cols)})
    
    out_df = pd.DataFrame(rows, columns=df_cols)
        
    return out_df

#spiegazione

xml_file = "/Users/tommasobassignana/Desktop/GLYFE-master/data/ohio/OhioT1DM-training/559-ws-training.xml"
df_cols = ("time", "glucose_level")

xtree = et.parse(xml_file)
xroot = xtree.getroot()
rows = []

for node in xroot: print(node)

for node in xroot: print(node.tag, node.text)

for element in node[0]: print(element)

#
def extract_attribute_names(root, child_index):
    """
    extract the name of the attributes from xml tree element
    :param root: root node
    :param child_index:  index of the child element from which i want the attributes names
    :return:
    """
    return list(root[child_index][0].attrib.keys())

#strange
xroot[0]
xroot[1]
xroot[0][0]
xroot[0][0].attrib.keys()   #dictionary keys
xroot[0][0].attrib.values() #dictionary values


for element in xroot[1].iter("event"): print(element) 
#"event" è il nome dell'element 

for element in xroot[1].iter("event"): print(element.attrib.values()) 
#"event" è il nome dell'element 


it = xroot[0].itertext()
for i in it: print(i)

it = xroot.find("glucose_level").get("value", default="NotFind")
print(it)

for element in xroot:
    print(element)


glu = extract_element_attribute_labels_names(xroot, 0)

#
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


# #iter(tag=None)
# Creates a tree iterator with the current element as the root. The iterator iterates over this element and all elements below it, in document (depth first) order. If tag is not None or '*', only elements whose tag equals tag are returned from the iterator.

for event in xroot[0].iter("event"): print(list(event.attrib.values())[0])
for event in xroot[0].iter("event"): print(list(event.attrib.values())[1])
#sono valori di un dizionario




glu_values = extract_element_attribute_singlelabel_values(xroot, 1)
glu_values_time = extract_element_attribute_singlelabel_values(xroot, 0)

#

len(glu_values)
len(glu_values_time)

dataset = pd.DataFrame(data = {"glu_values":glu_values, "glu_values_time":glu_values_time }, columns = ("glu_values", "glu_values_time"))

##########
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









import pandas
from matplotlib import rcParams
import requests
import json
import seaborn as sns
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
import hashlib
import os
from operator import itemgetter

rcParams['figure.figsize'] = 11.7,8.27

def pretty_json(json_object_):
    print(json.dumps(json_object_, indent=2))

# def get_entity(url_frgmnt):
#     api_key_ = "IoRY0Epk2bFThKBiXgmUWO8uJzndy5Ct"
#     api_endpoint_ = "https://api.core.ac.uk/v3/"
    
#     headers  = {"Authorization":"Bearer "+api_key_}
#     response = requests.get(api_endpoint_ + url_frgmnt, headers=headers)

#     if response.status_code == 200:
#         return response.json(), response.elapsed.total_seconds()
#     else:
#         print(f"Error code {response.status_code}, {response.content}")

def queryapi(url_frgmnt, query, is_scroll = False, limit = 100, scrollId=None ):
    api_key = "IoRY0Epk2bFThKBiXgmUWO8uJzndy5Ct"
    api_endpoint = "https://api.core.ac.uk/v3/"
    
    headers  = {"Authorization":"Bearer "+api_key}
    query = {"q":query, "limit":limit}

    if not is_scroll:
        response = requests.post(f"{api_endpoint}{url_frgmnt}", data=json.dumps(query), headers=headers)
    elif not scrollId:
        query["scroll"]="true"  
        response = requests.post(f"{api_endpoint}{url_frgmnt}", data=json.dumps(query), headers=headers)
    else:
        query["scrollId"]=scrollId    
        response = requests.post(f"{api_endpoint}{url_frgmnt}", data=json.dumps(query), headers=headers)

    if response.status_code == 200:
        return response.json(), response.elapsed.total_seconds()
    else:
        print(f"Error code {response.status_code}, {response.content}")   


query = f"covid"      
results, elapsed = queryapi("search/works", query, limit=5)
pretty_json(results)               

# data_provider_, elapsed_ = get_entity("data-providers/1")
# pretty_json(data_provider_)
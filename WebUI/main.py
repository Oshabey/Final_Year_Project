from flask import Flask, jsonify, render_template, request
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

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    search_query = ''
    if request.method == 'POST':
        # Get the search query from the form
        search_query = request.form['search_query']
  
        print("Search Query:", search_query)

    if search_query == '':
        return render_template('index.html')
    else:
        json_obj = query_build(search_query)
        return render_template('index.html', jsonfile=json.dumps(json_obj))

def pretty_json(json_object_):
    print(json.dumps(json_object_, indent=2))


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

def query_build(user_input):
    query = user_input      
    results, elapsed = queryapi("search/works", query, limit=5)
    return results
    # pretty_json(results)     

if __name__ == '__main__':
    app.run(debug=True)

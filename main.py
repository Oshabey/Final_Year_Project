# =================================== import libraries ===================================
import torch
from sentence_transformers import util
import pickle
import numpy as np
from tensorflow import keras
from flask import Flask, request, render_template
app = Flask(__name__)
# =================================== load save categorization models ===================================

embeddings = pickle.load(open('models/embeddings.pkl','rb')) 
sentences = pickle.load(open('models/sentences.pkl','rb'))
rec_model = pickle.load(open('models/rec_model.pkl','rb'))


# ==================================== custom functions ====================================
def recommendation(input_paper):
    # Calculate cosine similarity scores between the embeddings of input_paper and all papers in the dataset.
    cosine_scores = util.cos_sim(embeddings, rec_model.encode(input_paper))

    # Get the indices of the top-k most similar papers based on cosine similarity.
    top_similar_papers = torch.topk(cosine_scores, dim=0, k=15, sorted=True)

    # Retrieve the titles of the top similar papers.
    papers_list = []
    for i in top_similar_papers.indices:
        papers_list.append(sentences[i.item()])

    return papers_list



# ========================================= create app =========================================

# Route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for recommendation
@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method=='POST':
        input_paper = request.form['paper_title']
        recommended_papers = recommendation(input_paper)
        searched_papers = recommended_papers[:10]
        suggested_papers   = recommended_papers[10:]

        return render_template('index.html', recommended_papers=searched_papers, suggested_papers=suggested_papers)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

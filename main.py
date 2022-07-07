from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from ast import literal_eval
import numpy
import json
app = FastAPI()

@app.get("/getrecommendations")
def hello(name: str):
    metadata = pd.read_csv('Overviews.csv', low_memory=True)

    def convert_to_list(item):
        return item.split(",")

    def create_soup(x):
        return ' '.join(x['cast']) + ' ' + x['director']


    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            #Check if director exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''
          
    tfidf = TfidfVectorizer(stop_words='english')
    metadata['overview'] = metadata['overview'].fillna('')
    
    features = ['cast', 'director']

    for feature in features:
        metadata[feature] = metadata[feature].apply(clean_data)
    metadata['cast'] = metadata['cast'].apply(convert_to_list)

    metadata['soup'] = metadata.apply(create_soup, axis=1)

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(metadata['soup'])

    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    metadata = metadata.reset_index()
    indices = pd.Series(metadata.index, index=metadata['title'])

    tfidf_matrix = tfidf.fit_transform(metadata['overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

    def get_recommendations(title, cosine_sim=cosine_sim):
        idx=indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices=[]
        for i in sim_scores:
            if i[1]!=0:
                movie_indices.append(i[0])
        return pd.Series(metadata['tmdbId'].iloc[movie_indices]).tolist()

    overview_matches = get_recommendations(name)
    cast_matches = get_recommendations(name, cosine_sim2)
    
    objects = []
    objects.append(overview_matches)    
    objects.append(cast_matches)

    return objects

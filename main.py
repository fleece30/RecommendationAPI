from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json
from scipy.sparse import save_npz, load_npz, csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from config import OVERVIEWS_CSV_PATH, COSINE_SIM2_PATH, INDICES_PATH

app = FastAPI()

origins = [
    "https://www.spookydeck.com/",
    "phenomenal-gumdrop-ed8e23.netlify.app",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


metadata = pd.read_csv(OVERVIEWS_CSV_PATH, low_memory=True)
num_parts = 2  # Number of parts to split the file into
parts = []
for i in range(num_parts):
    part = load_npz(f'data/cosine_sim_part_{i + 1}.npz')
    parts.append(part)

# Combine the parts back into a single matrix
cosine_sim = vstack(parts)
cosine_sim2 = load_npz(COSINE_SIM2_PATH)

with open(INDICES_PATH, "r") as txt_file:
    indices_dict = json.load(txt_file)
indices = pd.Series(indices_dict).astype(int)
indices.index = indices.index.map(int)


@app.get("/createsimilaritymatrices")
def hello():
    global cosine_sim, cosine_sim2, indices, metadata

    def convert_to_list(item):
        return item.split(",")

    def create_soup(x):
        return ' '.join(x['cast']) + ' ' + x['director']

    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            # Check if director exists. If not, return empty string
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
    tfidf_matrix = tfidf.fit_transform(metadata['overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    row_indices = np.array_split(np.arange(cosine_sim.shape[0]), num_parts)

    for i, rows in enumerate(row_indices):
        part = cosine_sim[rows]
        save_npz(f'data/cosine_sim_part_{i + 1}.npz', csr_matrix(part))

    indices = pd.Series(
        metadata.index, index=metadata['tmdbId']).drop_duplicates()

    save_npz(COSINE_SIM2_PATH, csr_matrix(cosine_sim2))
    with open(INDICES_PATH, "w") as txt_file:
        json.dump(indices.to_dict(), txt_file)

    return []


@app.get("/getrecommendations")
def hello(tmdbId: int, resultCount: int = 10):
    if tmdbId not in indices:
        return {"error": "tmdbId not found"}
    idx = indices[tmdbId]

    def get_similar_movies(cosine_sim_matrix):
        sim_scores = list(enumerate(cosine_sim_matrix[idx].toarray().flatten()))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:resultCount + 1]
        movie_indices = [i[0] for i in sim_scores if i[1] != 0]
        return metadata['tmdbId'].iloc[movie_indices].tolist()

    overview_matches = get_similar_movies(cosine_sim)
    cast_matches = get_similar_movies(cosine_sim2)
    objects = [overview_matches, cast_matches]
    return objects

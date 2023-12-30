import os
os.environ['FLASK_ENV'] = 'production'

from flask import Flask, render_template, request, redirect, url_for
from pickle import load
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

df = pd.read_csv('movies_processed.csv')
model = load(open("knn_neighbors_project.sav", "rb"))
df.fillna('', inplace=True)
vector = TfidfVectorizer().fit_transform(df['tags'])

def rec(movie):
    recs = []
    if movie not in df['title'].values:
        return 'Movie not in database, please choose another one'
    else:

        movie_index = df[df['title'] == movie].index[0]
        distances, indices = model.kneighbors(vector[movie_index], n_neighbors=6)
        similar_movies = [(df['title'][i], distances[0][j]) for j, i in enumerate(indices[0])]
        for m in range(1, len(similar_movies)):
            recs.append(similar_movies[m][0])
    return recs

@app.route('/', methods=['GET', 'POST'])
def rootpage():
    movie = ''
    recommendations = []
    movie_not_in_database = None

    if request.method == 'POST' and 'movie' in request.form:
        movie = request.form.get('movie')
        if movie not in df['title'].values:
            movie_not_in_database = 'Movie not in database, please choose another one'
        else:
            recommendations = rec(movie)

    return render_template('index.html', movie=movie, recommendations=recommendations, movie_not_in_database=movie_not_in_database)

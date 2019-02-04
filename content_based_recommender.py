# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:51:59 2019

@author: Franco
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 17:14:38 2019

@author: Franco
"""

import pandas  as pd
# Load Movies Metadata
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

# Print the first three rows
# print(metadata.head(3))

# Calculate C
C = metadata['vote_average'].mean()

# Calculate the minimum number of votes required to be in the chart, m
m = metadata['vote_count'].quantile(0.90)
# print(m)

# Filter out all qualified movies into a new DataFrame.
# Loc => Access a group of rows and columns by label(s) or a boolean array.
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
# print(q_movies.shape)
q_movies.reindex()

# Function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
# Apply => Apply a function along an axis of the DataFrame.
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)


q_movies = q_movies.reset_index()
#Print the top 15 movies
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))


#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
#metadata['overview'] = metadata['overview'].fillna('')
q_movies['overview'] = q_movies['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data. Now using q_movies because of matrix size
#tfidf_matrix = tfidf.fit_transform(metadata['overview'])
tfidf_matrix = tfidf.fit_transform(q_movies['overview'])


#Output the shape of tfidf_matrix (Rows,Columns)
print('tfidf_matrix shape:')
print(tfidf_matrix.shape)


# Compute the cosine similarity matrix

from sklearn.metrics.pairwise import linear_kernel
#cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print('cosine_sim:')
print(cosine_sim.shape)

#Construct a reverse map of indices and movie titles
indices = pd.Series(q_movies.index, index=q_movies['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return q_movies['title'].iloc[movie_indices]

print(get_recommendations('The Dark Knight Rises'))



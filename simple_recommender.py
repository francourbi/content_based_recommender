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

#Print the top 15 movies
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))
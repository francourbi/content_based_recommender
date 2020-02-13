# Simple Recommender

Simple recommenders are basic systems that recommends the top items based on a certain metric or score.

The following are the steps involved:

* Decide on the metric or score to rate movies on.
* Calculate the score for every movie.
* Sort the movies based on the score and output the top results.

# Content-based recommender

Content based recommenders suggest similar items based on a particular item. This system uses item metadata, such as genre, director, description, actors, etc. for movies, to make these recommendations. The general idea behind these recommender systems is that if a person liked a particular item, he or she will also like an item that is similar to it.

The following are the steps involved:

* Compute the Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each document using Scikit TfIdfVectorizer class.
* Use the cosine similarity to calculate a numeric quantity that denotes the similarity between two movies.
* Define a function that takes in a movie title as an input and outputs a list of the 10 most similar movies



Copied from this site: https://www.datacamp.com/community/tutorials/recommender-systems-python


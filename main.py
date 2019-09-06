import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.metrics.pairwise import linear_kernel,cosine_similarity

from ast import literal_eval



credit_d = pd.read_csv("the-movies-dataset/credits.csv")

movie_d = pd.read_csv("the-movies-dataset/movies_metadata.csv")


#cleaning the data
movie_d = movie_d[movie_d.id!='1997-08-20']
movie_d = movie_d[movie_d.id!='2012-09-29']
movie_d = movie_d[movie_d.id!='2014-01-01']

#converting id to int so that it can merge
movie_d['id'] = movie_d['id'].astype('int')

#ValueError: You are trying to merge on object and int64 columns. If you wish to proceed you should use pd.concat


#merging the two data set
movie_d = movie_d.merge(credit_d,on='id')


#print(movie_d.columns)
"""Index(['adult', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'id',
       'imdb_id', 'original_language', 'original_title', 'overview',
       'popularity', 'poster_path', 'production_companies',
       'production_countries', 'release_date', 'revenue', 'runtime',
       'spoken_languages', 'status', 'tagline', 'title', 'video',
       'vote_average', 'vote_count', 'cast', 'crew'],
      dtype='object')"""



#Converting each overview into it's corresponding tf-idf vector


vectorized = TfidfVectorizer(stop_words='english')

#we can't vectorize Nan so we will have to convert into string
movie_d['overview'] = movie_d['overview'].fillna("")


"""while calculating it showed me an error that MemoryError: Unable to allocate array
with shape (45538, 45538) and data type float64
So I will have to try reducing the size of the dataset
"""

dataset_size = 1000

movie_d = movie_d.head(dataset_size)

vectorized_matrix = vectorized.fit_transform(movie_d['overview'])

cosine_sim = linear_kernel(vectorized_matrix,vectorized_matrix)

#cosine_sim.shape  (25000, 25000)

indexes = pd.Series(movie_d.index,index=movie_d['title']).drop_duplicates()

def recommendation(title,cosine_sim=cosine_sim):
    
    index = indexes[title]
    
    scores = list(enumerate(cosine_sim[index]))
    
    #sorting on the basis of the scores
    
    scores = sorted(scores,key=lambda i:i[1],reverse=True)
    
    scores = scores[1:11]
    
    ids = [i[0] for i in scores]
    
    return movie_d['title'].iloc[ids]


print("Recommendation based on overview")
print(recommendation("Jumanji",cosine_sim=cosine_sim))

"""23699               Beware of Pity
3668             Project Moon Base
18074        The War of the Robots
18516                    Apollo 18
15774                 The American
16135                Bloodbrothers
20020    Welcome to the Space Show
2466                    The Matrix
21225                Europa Report
13944         The Inhabited Island
Name: title, dtype: object"""


#Code for genre based recommendation


#since genres are in the form of list of dictionary, we will have to just extract the name of genre



def obtain_genre(l):
    
    if(isinstance(l,list)):
        genres = [i['name'] for i in l]
        
        if(len(genres) > 3):
            genres = genres[:3]
        return genres
    
    return []



#our data is in "stringified" lists so we will have to convert this data
from ast import literal_eval

movie_d['genres'] = movie_d['genres'].apply(literal_eval)

movie_d['genres'] = movie_d['genres'].apply(obtain_genre)




#i will create a new column which will consist of all the genre separated by space

def join_genre(x):
    
    return ' '.join(x['genres'])

movie_d['joined_genres'] = movie_d.apply(join_genre,axis=1)


#since we are dealing with genre words instead of context we can use count vectorizer

count_vec = CountVectorizer(stop_words='english')

cont_vec_matrix = count_vec.fit_transform(movie_d['joined_genres'])



cosine_sim2 = cosine_similarity(cont_vec_matrix, cont_vec_matrix)


#we can use our same function to make predictions but before that we will have to reset our indices
movie_d = movie_d.reset_index(drop=True)

indexes = pd.Series(movie_d.index, index=movie_d['title']).drop_duplicates()

print()
print("Recommendation based on Genre")
print(recommendation("Jumanji",cosine_sim2))
"""
528              The Shadow
1239             Highlander
1469      The Fifth Element
1523    Conan the Barbarian
1527                  Spawn
1542     Kull the Conqueror
1748          The Borrowers
2268              Red Sonja
2511              The Mummy
2534               Superman
Name: title, dtype: object
"""



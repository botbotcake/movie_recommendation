from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords

import string

import gensim

import pandas as pd



stop_words = set(stopwords.words('english'))

punctuation_char = set(string.punctuation)

lemmatizing = WordNetLemmatizer()

def cleaning_doc(doc):

    stopwords_removed = " ".join([i for i in doc.lower().split() if i not in stop_words])
    punctuation_removed = ''.join(i for i in stopwords_removed if i not in punctuation_char)
    lemmatized = " ".join(lemmatizing.lemmatize(word) for word in punctuation_removed.split())

    return lemmatized

movie_d = pd.read_csv("the-movies-dataset/movies_metadata.csv")

movie_d['overview'] = movie_d['overview'].fillna("")


docs = list(movie_d["overview"])[:500]


doc_clean = [cleaning_doc(doc).split() for doc in docs]       


#Now we have cleaned our overviews!

#We will use genism to convert our document into document term matrix format

dictionary = gensim.corpora.Dictionary(doc_clean)

doc_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]


#now we will create an lda model and train it on our document term matrix

Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_matrix, num_topics=3, id2word = dictionary, passes=50)


print(ldamodel.print_topics(num_topics=3, num_words=3))

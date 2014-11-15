from pymongo import MongoClient
import xmlrpclib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import numpy as np
import gensim
from gensim.models import Word2Vec
#from gensim.models import Doc2Vec
#from gensim.models.doc2vec import LabeledSentence
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora.textcorpus import TextCorpus
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn
#import matplotlib.pyplot as plt
from sklearn.svm import SVC

from j_kmeans import Kmeans
from collections import Counter

class ticketClassifier(object):
  def __init__(self, parms,clean_up=True):
    self.parms = parms
    self.mongo_coll = parms['mongo_coll']
    self.stemmer = SnowballStemmer('english')
    self.clean_up = clean_up

    self.X = None
    self.ids = None
    self.all_sentences = None
    self.all_labels = None
    self.urls = None
    self.all_or_nothing_vectors = None
    self.tfidf = None
    self.all_t = None
    self.labeled_t = None
    self.all_kmeans = None
    self.labeled_kmeans = None
    self.labels_raw = None
    self.guesses_zipped = None

  def run(self):
    self.extract_features()
    self.fit_universal_models()
    self.fit_and_predict_for_categories()
    self.update_mongo()
    if self.clean_up:
        self._clean_up()

  def extract_features(self):
    self.X = []
    self.ids = []
    self.all_sentences = []
    self.all_labels = []
    self.urls = []
    self.all_or_nothing_vectors = {}

    labeled_count = 0
    for x in self.mongo_coll.find():
        text = x['title'] + ' ' + x['body']
        stemmed = token_stem(text,self.stemmer)

        self.ids.append(x['_id'])
        self.all_sentences.append(stemmed)
        labels = np.unique([self.parms['labeldict'][l] for l in x['labels']])
        self.all_labels.append(labels)
        if sum(1 if l in self.parms['mutually_exclusive_labels'] else 0 for l in labels) == 1:
          self.X.append(stemmed)
          self.urls.append(x['ticket_url'])
          for l in labels:
            if l not in self.all_or_nothing_vectors:
                self.all_or_nothing_vectors[l] = [0] * labeled_count
          for l in self.all_or_nothing_vectors:
            if l in labels:
              self.all_or_nothing_vectors[l].append(1)
            else:
              self.all_or_nothing_vectors[l].append(0)
          labeled_count += 1

    for x in self.all_or_nothing_vectors:
        self.all_or_nothing_vectors[x] = np.array(self.all_or_nothing_vectors[x])


  def fit_universal_models(self):

    vec = CountVectorizer(stop_words='english',max_features=10000)
    vec_t = vec.fit_transform(' '.join(x) for x in self.all_sentences)

    id2word = {v: k for k, v in vec.vocabulary_.iteritems()}
    vec_corpus = gensim.matutils.Sparse2Corpus(vec_t.T)

    lda = LdaMulticore(corpus=vec_corpus,id2word=id2word,iterations=200,num_topics=2,passes=10,workers=4)
    all_counts = vec.transform(' '.join(x) for x in self.all_sentences)
    self.all_probas = lda.inference(gensim.matutils.Sparse2Corpus(all_counts.T))[0]
    labeled_counts = vec.transform(' '.join(x) for x in self.X)
    self.labeled_probas = lda.inference(gensim.matutils.Sparse2Corpus(labeled_counts.T))[0]

    w2vmodel = Word2Vec(self.all_sentences, size=100, window=5, min_count=3, workers=4)

    best_centroids = None
    best_score = None
    for _ in xrange(10):  # todo -- implement kmeans++ instead of best of 10
      km = Kmeans(50)
      km.fit(w2vmodel.syn0)
      score = km.compute_sse(w2vmodel.syn0)
      if best_score is None or score < best_score:
            best_score = score
            best_centroids = km.centroids
    km.centroids = best_centroids

    self.tfidf = TfidfVectorizer(stop_words=set(stopwords.words()))
    self.all_t = self.tfidf.fit_transform(' '.join(x) for x in self.all_sentences)
    self.labeled_t = self.tfidf.transform(' '.join(x) for x in self.X)

    self.all_kmeans = kmeans_word2vecify(self.all_sentences,w2vmodel,km,self.all_t,self.tfidf)
    self.labeled_kmeans = kmeans_word2vecify(self.X,w2vmodel,km,self.labeled_t,self.tfidf)

  def fit_and_predict_for_categories(self):
    self.labels_raw =  self.all_or_nothing_vectors.keys()
    guesses_raw = map(lambda x: self.fit_and_predict_for_category(self.all_or_nothing_vectors[x]),self.labels_raw)
    self.guesses_zipped = zip(*guesses_raw)

  def fit_and_predict_for_category(self, y_vector):

    lr_lda = LogisticRegression()
    lr_lda.fit(self.labeled_probas,y_vector)
    lr_tfidf = LogisticRegression()
    lr_tfidf.fit(self.labeled_t, y_vector)
    svc_kmeans = SVC(probability=True)
    svc_kmeans.fit(self.labeled_kmeans,y_vector)
    lr_ensemble_labeled_X = np.hstack((
                               lr_tfidf.predict_proba(self.labeled_t)[:,0].reshape(-1,1),
                               lr_lda.predict_proba(self.labeled_probas)[:,0].reshape(-1,1),
                               svc_kmeans.predict_proba(self.labeled_kmeans)[:,0].reshape(-1,1)
                               ))
    lr_ensemble = LogisticRegression()
    lr_ensemble.fit(lr_ensemble_labeled_X,y_vector)
    lr_all_sentences = np.hstack((
                           lr_tfidf.predict_proba(self.all_t)[:,0].reshape(-1,1),
                           lr_lda.predict_proba(self.all_probas)[:,0].reshape(-1,1),
                           svc_kmeans.predict_proba(self.all_kmeans)[:,0].reshape(-1,1)
                           ))
    all_preds = np.array(lr_ensemble.predict_proba(lr_all_sentences)[:,1])
    return all_preds

  def update_mongo(self):
    for i, g in enumerate(self.guesses_zipped):
        guess = {c[0]: c[1] for c in zip(self.labels_raw,g)}
        tid = self.ids[i]
        self.mongo_coll.update({'_id':tid},{"$set":{"guesses": guess}})

  def _clean_up(self):
    self.X = None
    self.ids = None
    self.all_sentences = None
    self.all_labels = None
    self.urls = None
    self.all_or_nothing_vectors = None
    self.tfidf = None
    self.all_t = None
    self.labeled_t = None
    self.all_kmeans = None
    self.labeled_kmeans = None
    self.labels_raw = None
    self.guesses_zipped = None
    self.parms = None


def token_stem(sentence,stemmer):
    date = re.compile(u'(201[0-9]+)')
    sentence = date.sub('regex_date',sentence)
    num = re.compile(u'([0-9]+)')
    sentence = num.sub('regex_num',sentence) # replace with # of digits
    pyfile = re.compile(u'(\/[\w\/]+.py)')
    sentence = pyfile.sub('regex_pyfile',sentence)
    dunders = re.compile(u'(__\w+|\w+__)')
    sentence = dunders.sub('regex_dunders',sentence)
    return [stemmer.stem(x) for x in re.findall(u'(?u)\\b\\w\\w+\\b',sentence.lower())]

def kmeans_word2vecify(in_docs,w2vmodel,km,tfidf_t,tfidf):
    distances = np.array(km.calc_distances(w2vmodel.syn0))
    b = distances == np.min(distances,axis=0)
    cluster_distances = np.array(distances).T
    closest_clusters = b.T * 1
    output = []
    revdict = {w: i for i,w in enumerate(w2vmodel.index2word)}
    for i, doc in enumerate(in_docs):
        x_dists = np.zeros(len(cluster_distances[0]))
        x_closest = np.zeros(len(closest_clusters[0]))
        count = 0
        for word, word_count in Counter(doc).iteritems():
            if word in revdict:
                try:
                    tfidf_score = tfidf_t[i,tfidf.vocabulary_[word]]
                    count += 1
                    x_dists = x_dists + cluster_distances[revdict[word]] * tfidf_score
                    x_closest = x_closest + closest_clusters[revdict[word]] * word_count
                except:
                    pass
        if count > 0:
            x_dists = x_dists / np.sum(x_dists)
            x_closest = x_closest / np.sum(x_closest)
        output.append(np.hstack((x_dists,x_closest)))
    return output

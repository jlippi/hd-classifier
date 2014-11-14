
# coding: utf-8

# In[6]:

from pymongo import MongoClient
import xmlrpclib
from ticketScraper import ticketScraper
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora.textcorpus import TextCorpus
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
get_ipython().magic(u'pylab inline')


# In[7]:

from j_kmeans import Kmeans
from collections import Counter


# In[8]:

client = MongoClient('localhost',27017)
db = client['hd-test']


# In[9]:

password = 'abc'


# In[10]:

SALT_parms = {'tracker': 'github',
              'project': 'saltstack/salt',
              'mongo_coll': db['salt'],
              'url': None,
              'auth': ('jlippi', password)}
SALT_parms['mutually_exclusive_labels'] = ['Bug','Feature','Documentation']
SALT_parms['labeldict'] = {'Bug': 'Bug',
             'bug': 'Bug',
             'feature': 'Feature',
             'Medium Severity': 'Bug',
             'Feature': 'Feature',
             'Low Severity': 'Bug',
             'Documentation': 'Documentation',
             'Fixed Pending Verification': 'Bug',
             'in progress': 'unk',
             'Bugfix - [Done] back-ported': 'Bug',
             'High Severity': 'Bug',
             'Low-Hanging Fruit': 'unk',
             'Windows': 'unk',
             'Salt-Cloud': 'unk',
             'Regression': 'Bug',
             'Pending Discussion': 'unk',
             'Duplicate': 'unk',
             'Expected Behavior': 'unk',
             'Cannot Reproduce': 'Bug',
             'Info Needed': 'unk',
             'Question': 'unk',
             'Salt-SSH': 'unk',
             'Packaging': 'unk',
             "Won't Fix For Now": 'unk',
             'Execution Module': 'unk',
             'State Module': 'unk',
             'Upstream Bug': 'Bug',
             'Needs Testcase': 'unk',
             'Multi-Master': 'unk',
             'Critical': 'Bug',
             'RAET': 'unk',
             'Confirmed': 'Bug',
             'Core': 'unk',
             'Salt-API': 'unk',
             'tt:not_started': 'unk',
             'Bugfix - back-port': 'Bug',
             'Other Module': 'unk',
             'Blocker': 'unk',
             'uncategorized':'unk'}


# In[12]:

#ts = ticketScraper(**SALT_parms)
#ts.run()


# In[13]:

#----#


# In[14]:

client = MongoClient('localhost', 27017)
db = client['hd-test']
coll = db['salt']
project = 'saltstack'
parms = SALT_parms


# In[15]:

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
X = []
ids = []
all_sentences = []
all_labels = []
urls = []
stemmer = SnowballStemmer('english')
labeled_count = 0
all_or_nothing_vectors = {}
for x in coll.find():
    if not x['title']:
        x['title'] = ''
    if not x['body']:
        x['body'] = ''
    text = x['title'] + ' ' + x['body']
    stemmed = token_stem(text,stemmer)
    ids.append(x['_id'])
    all_sentences.append(stemmed)
    labels = np.unique([parms['labeldict'][l] for l in x['labels']])
    all_labels.append(labels)
    if len(x['labels']) > 0:
        if sum(1 if l in parms['mutually_exclusive_labels'] else 0 for l in labels) == 1:
          X.append(stemmed)
          urls.append(x['ticket_url'])
          for l in labels:
            if l not in all_or_nothing_vectors:
                all_or_nothing_vectors[l] = [0] * labeled_count
          for l in all_or_nothing_vectors:
            if l in labels:
              all_or_nothing_vectors[l].append(1)
            else:
              all_or_nothing_vectors[l].append(0)
          labeled_count += 1

for x in all_or_nothing_vectors:
    all_or_nothing_vectors[x] = np.array(all_or_nothing_vectors[x])


# In[16]:

y_vector = all_or_nothing_vectors['Bug']


# In[17]:

vec = CountVectorizer(stop_words='english',max_features=10000)
vec_t = vec.fit_transform(' '.join(x) for x in all_sentences)

id2word = {v: k for k, v in vec.vocabulary_.iteritems()}
vec_corpus = gensim.matutils.Sparse2Corpus(vec_t.T)

lda = LdaMulticore(corpus=vec_corpus,id2word=id2word,iterations=200,num_topics=2,passes=1,workers=4)
all_counts = vec.transform(' '.join(x) for x in all_sentences)
all_probas = lda.inference(gensim.matutils.Sparse2Corpus(all_counts.T))[0]
labeled_counts = vec.transform(' '.join(x) for x in X)
labeled_probas = lda.inference(gensim.matutils.Sparse2Corpus(labeled_counts.T))[0]

w2vmodel = Word2Vec(all_sentences, size=100, window=5, min_count=3, workers=4)

best_centroids = None
best_score = None
for _ in xrange(1):
  km = Kmeans(50)
  km.fit(w2vmodel.syn0)
  score = km.compute_sse(w2vmodel.syn0)
  if best_score is None or score < best_score:
        best_score = score
        best_centroids = km.centroids
km.centroids = best_centroids

distances = np.array(km.calc_distances(w2vmodel.syn0))
b = distances == np.min(distances,axis=0)
cluster_distances = np.array(distances).T
closest_clusters = b.T * 1

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

all_kmeans = kmeans_word2vecify(all_sentences,w2vmodel,km,all_t,tfidf)
labeled_kmeans = kmeans_word2vecify(X,w2vmodel,km,labeled_t,tfidf)
tfidf = TfidfVectorizer(stop_words=set(stopwords.words()))
all_t = tfidf.fit_transform(' '.join(x) for x in all_sentences)
labeled_t = tfidf.transform(' '.join(x) for x in X)


# In[110]:

def fit_and_predict_for_category(y_vector, labeled_probas, labeled_t, labeled_kmeans,
                                           all_t, all_probas, all_kmeans):
    lr_lda = LogisticRegression()
    lr_lda.fit(labeled_probas,y_vector)
    lr_tfidf = LogisticRegression()
    lr_tfidf.fit(labeled_t, y_vector)
    svc_kmeans = SVC(probability=True)
    svc_kmeans.fit(labeled_kmeans,y_vector)
    svc_kmeans.score(labeled_kmeans,y_vector)
    lr_ensemble_labeled_X = np.hstack((
                               lr_tfidf.predict_proba(labeled_t)[:,0].reshape(-1,1),
                               lr_lda.predict_proba(labeled_probas)[:,0].reshape(-1,1),
                               svc_kmeans.predict_proba(labeled_kmeans)[:,0].reshape(-1,1)
                               ))
    lr_ensemble = LogisticRegression()
    lr_ensemble.fit(lr_ensemble_labeled_X,y_vector)
    lr_all_sentences = np.hstack((
                           lr_tfidf.predict_proba(all_t)[:,0].reshape(-1,1),
                           lr_lda.predict_proba(all_probas)[:,0].reshape(-1,1),
                           svc_kmeans.predict_proba(all_kmeans)[:,0].reshape(-1,1)
                           ))
    all_preds = np.array(lr_ensemble.predict_proba(lr_all_sentences)[:,1])
    return all_preds


# In[111]:

a = fit_and_predict_for_category(all_or_nothing_vectors['Bug'],labeled_probas,labeled_t,labeled_kmeans,all_t,all_probas,all_kmeans)


# In[112]:

print np.mean(all_or_nothing_vectors['Bug'])
print np.mean(a)


# In[113]:

print np.mean(guesses_raw[3])


# In[114]:

guesses_raw = map(lambda x: fit_and_predict_for_category(all_or_nothing_vectors[x], 
                                           labeled_probas, labeled_t, labeled_kmeans,
                                           all_t, all_probas, all_kmeans)
              ,all_or_nothing_vectors)
labels_raw = [x for x in all_or_nothing_vectors]


# In[115]:

guesses_zipped = zip(*guesses_raw)


# In[116]:

for i, g in enumerate(guesses_zipped):
    guess = {c[0]: c[1] for c in zip(labels_raw,g)}
    tid = ids[i]
    coll.update({'_id':tid},{"$set":{"guesses": guess}})


# In[117]:

labels_raw


# In[118]:

np.mean(guesses_raw[3])


# In[97]:

1. - all_or_nothing_vectors['Bug'].mean()


# In[102]:

y_vector.mean()


# In[101]:

guesses_raw[3].mean()


# In[ ]:





# coding: utf-8

# In[1]:

import requests
from pymongo import MongoClient
from requests.auth import HTTPBasicAuth
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from nltk.stem import SnowballStemmer
import re
from gensim.models.doc2vec import LabeledSentence
from gensim.models.ldamodel import LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora.textcorpus import TextCorpus
from sklearn.svm import SVC
get_ipython().magic(u'pylab inline')


# In[3]:

#from sklearn.cluster import _k_means
#a = requests.get('https://issues.apache.org/jira/browse/HIVE-8710')
#client = MongoClient('localhost', 27017)
#db = client['helpdesk']
#coll = db['tickets']


# In[2]:

import sklearn
sklearn.__version__


# In[232]:

#baseurl = 'https://issues.apache.org/jira/browse/'
#project = 'HIVE'
#for x in range(6418,8711):
#    ticket = make_db_entry(baseurl, project, x)
#    if x % 100 == 0:
#        print 'x:', x, ' status: ', ticket['http status']
#    insert_db_entry(ticket,coll)


# In[233]:

#def make_db_entry(baseurl, project, num):
#    url = baseurl + project + '-' + str(num)
#    a = requests.get(url)
#    ticket = {'url': url,
#              'ticketnum': num,
#              'project': project,
#              'http status': a.status_code,
#              'text': a.text}
#    return ticket
#
#def insert_db_entry(ticket, coll):
#    coll.insert(ticket)


# In[234]:

#coll.find({'http status': 200}).count()


# In[235]:

#coll.find({},{"$max": {"ticketnum": 1}}).count()


# In[236]:

#for x in a


# In[3]:

baseurl = 'https://api.github.com/repos/saltstack/salt/issues?sort=updated&direction=asc&state=all'
#a = requests.get('https://api.github.com/repos/saltstack/salt/issues?sort=updated&direction=asc&state=all')
client = MongoClient('localhost', 27017)
db = client['helpdesk']
coll = db['github_saltstack']
project = 'saltstack'


# In[8]:

def db_loop_entry(baseurl, project, coll, page):
    url = baseurl + '&page=' + str(page)
    a = requests.get(url, auth=HTTPBasicAuth(uid,secret_pass))
    json_list = a.json()
    if len(json_list) > 0:
      for j_obj in json_list:
         ticket = {'url': url,
                   'project': project,
                   'json': j_obj}
         coll.insert(ticket)
      return True
    return False


# In[114]:

#latest = db_loop_entry(baseurl, project, coll)
page = 1
more = True
while more:
    more = db_loop_entry(baseurl, project, coll, page)
    if page % 20 == 0:
        print page
    page += 1


# In[7]:

count_with_labels = 0
count = 0
count_of_labels = 0
all_labels = Counter()
for x in coll.find():
    num_labels = len(x['json']['labels'])
    if num_labels > 0:
        count_with_labels += 1
        count_of_labels += num_labels
    for label in x['json']['labels']:
        all_labels[label['name']] += 1
    count += 1


# In[10]:

print count_with_labels
print count
print count_of_labels
print all_labels


# In[11]:

for x in coll.find():
  if 'body' not in x['json'] or x['json']['body'] is None:
    x['json']['body'] = ''
  if 'text' not in x['json'] or x['json']['text'] is None:
    x['json']['text'] = ''
  x['json']['text'] = x['json']['body'] + x['json']['title']
  for label in x['json']['labels']:
        x['json']['label' + label['name']] = 1
  coll.update({"_id": x['_id']},x)


# In[4]:

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


# In[20]:

X = []
y_bug = []
y_feature = []
#y_documentation = []
all_sentences = []
all_labels = []
sentences_for_word2vec = []
stemmer = SnowballStemmer('english')
for i, x in enumerate(coll.find()):
    all_sentences.append(token_stem(x['json']['text'],stemmer))
    sentences_for_word2vec.append(token_stem(x['json']['text'],stemmer))
    all_labels.append([y['name'] for y in x['json']['labels']])
    if len(x['json']['labels']) > 0:
      if 'labelBug' in x['json'] or 'labelFeature' in x['json']:
          X.append(token_stem(x['json']['text'],stemmer))
      #X.append(x['json']['title'])
          if 'labelBug' in x['json']:
              y_bug.append(1)
          else:
              y_bug.append(0)
          if 'labelFeature' in x['json']:
              y_feature.append(1)
          else:
              y_feature.append(0)
      #if 'labelDocumentation' in x['json']:
      #    y_documentation.append(1)
      #else:
      #    y_documentation.append(0)
y_bug = np.array(y_bug)
y_feature = np.array(y_feature)
#y_documentation = np.array(y_documentation)


# In[174]:

y_comb = np.argmax(np.hstack((y_feature.reshape(-1,1), y_bug.reshape(-1,1))),axis=1)


# In[7]:

X = [' '.join(x) for x in X]


# In[253]:

#vec = TfidfVectorizer(stop_words='english',max_features=10000)
#tfidf_t = vec.fit_transform(' '.join(x) for x in all_sentences)
#id2word = {v: k for k, v in vec.vocabulary_.iteritems()}
#tfidf_corpus = gensim.matutils.Sparse2Corpus(tfidf_t.T)
lda = LdaMulticore(corpus=tfidf_corpus,id2word=id2word,iterations=200,num_topics=20,passes=20,workers=4)


# In[254]:

lda.save('ldalippi.lda')


# In[255]:

labeled_counts = vec.transform(X)


# In[256]:

idx = np.random.rand(labeled_counts.shape[0]) < .8
labeled_counts_train = labeled_counts[idx]
labeled_counts_test = labeled_counts[~idx]
y_comb_train = y_comb[idx]
y_comb_test = y_comb[~idx]


# In[257]:

labeled_probas = lda.inference(gensim.matutils.Sparse2Corpus(labeled_counts.T))


# In[258]:

labeled_probas_train = labeled_probas[0][idx]
labeled_probas_test = labeled_probas[0][~idx]


# In[259]:

#f = RandomForestClassifier(n_estimators=1000)
#f.fit(labeled_probas_train,y_comb_train)
lr_lda = LogisticRegression()
lr_lda.fit(labeled_probas_train,y_comb_train)


# In[260]:

lr_lda.score(labeled_probas_test,y_comb_test)


# In[122]:

#doc2vec_inputs = []
#for i in xrange(0,len(sentences_for_word2vec)):
#    doc2vec_inputs.append(LabeledSentence(sentences_for_word2vec[i],labels_for_doc2vec[i]))


# In[214]:

#tfidf = TfidfVectorizer(max_features=10000,stop_words=set(stopwords.words()))
#tfidf_t = tfidf.fit_transform(X)
#tfidf = CountVectorizer(stop_words=set(stopwords.words()))
#tfidf_t = tfidf.fit_transform(' '.join(x) for x in X)


# In[261]:

#idx = np.random.rand(len(X),1)[:,0] < .8
tfidf_t_train = tfidf_t[idx]
tfidf_t_test = tfidf_t[~idx]
y_bug_train = y_bug[idx]
y_bug_test = y_bug[~idx]
y_feature_train = y_feature[idx]
y_feature_test = y_feature[~idx]
#y_documentation_train = y_documentation[idx]
#y_documentation_test = y_documentation[~idx]

#y_comb = np.argmax(np.hstack((np.zeros((len(y_feature),1)),y_feature.reshape(-1,1), y_bug.reshape(-1,1))),axis=1)
y_comb = np.argmax(np.hstack((y_feature.reshape(-1,1), y_bug.reshape(-1,1))),axis=1)
y_comb_train = y_comb[idx]
y_comb_test = y_comb[~idx]


# In[262]:

lr_bug = LogisticRegression()
lr_bug.fit(tfidf_t_train, y_bug_train)
confusion_matrix(lr_bug.predict(tfidf_t_test),y_bug_test)


# In[263]:

lr_feature = LogisticRegression()
lr_feature.fit(tfidf_t_train, y_feature_train)
confusion_matrix(lr_feature.predict(tfidf_t_test),y_feature_test)
# high FPR, low FNR


# In[264]:

#lr_documentation = LogisticRegression()
#lr_documentation.fit(tfidf_t_train, y_documentation_train)
#confusion_matrix(lr_documentation.predict(tfidf_t_test), y_documentation_test)
# high FPR, low FNR


# In[265]:

rf_in_bug_feature_test = np.hstack((
                           lr_bug.predict_proba(tfidf_t_test)[:,0].reshape(-1,1),
                           lr_feature.predict_proba(tfidf_t_test)[:,0].reshape(-1,1),
                           lr_lda.predict_proba(labeled_probas_test)[:,0].reshape(-1,1)))
rf_in_bug_feature_train = np.hstack((lr_bug.predict_proba(tfidf_t_train)[:,0].reshape(-1,1),
                              lr_feature.predict_proba(tfidf_t_train)[:,0].reshape(-1,1),
                              lr_lda.predict_proba(labeled_probas_train)[:,0].reshape(-1,1)))


# In[266]:

lr = LogisticRegression()
lr.fit(rf_in_bug_feature_train,y_comb_train)
confusion_matrix(lr.predict(rf_in_bug_feature_test),y_comb_test)


# In[43]:




# In[21]:

model = Word2Vec(sentences_for_word2vec, size=100, window=5, min_count=5, workers=4)


# In[22]:

print model.most_similar(['featur'])
print model.most_similar(['bug'])


# In[23]:

from j_kmeans import Kmeans
from copy import deepcopy
vectorspace = deepcopy(np.array(model.syn0))


# In[114]:




# In[87]:

#import matplotlib.pyplot as plt
#plt.plot(range(1,100,5),scores)


# In[83]:

from j_kmeans import Kmeans
scores = []
for i in xrange(1,20,1):
  print i
  km = Kmeans(i)
  km.fit(model.syn0)
  scores.append(km.compute_sse(model.syn0))
plt.plot(range(1,20,1),scores)


# In[85]:

model = Word2Vec(sentences_for_word2vec, size=100, window=5, min_count=5, workers=4)
km = Kmeans(5)
km.fit(model.syn0)


# In[122]:

distances = np.array(km.calc_distances(model.syn0))
b = distances == np.min(distances,axis=0)
#distances_normed = distances/distances.sum(axis=0)


# In[130]:

#word_features = np.hstack((np.array(distances).T,b.T * 1))
cluster_distances = np.array(distances).T
closest_clusters = b.T * 1


# In[167]:

X_kmeans = []
revdict = {w: i for i,w in enumerate(model.index2word)}
for doc in X:
    x_dists = np.zeros(len(cluster_distances[0]))
    x_closest = np.zeros(len(closest_clusters[0]))
    count = 0
    for word in doc:
      if word in revdict:
        count += 1
        x_dists = x_dists + cluster_distances[revdict[word]]
        x_closest = x_closest + closest_clusters[revdict[word]]
    if count > 0:
      x_dists = x_dists / count
      x_closest = x_closest / count
    X_kmeans.append(np.hstack((x_dists,x_closest)))


# In[175]:

X_kmeans = np.array(X_kmeans)
idx = np.random.rand(X_kmeans.shape[0]) < .8
X_kmeans_train = X_kmeans[idx]
X_kmeans_test = X_kmeans[~idx]
y_comb_train = y_comb[idx]
y_comb_test = y_comb[~idx]


# In[177]:

lr = LogisticRegression()
lr.fit(X_kmeans_train,y_comb_train)


# In[178]:

lr.score(X_kmeans_test,y_comb_test)


# In[181]:

confusion_matrix(lr.predict(X_kmeans_test),y_comb_test)


# In[ ]:




# In[139]:

d2vmodel = Doc2Vec(workers=4,dm=1,window=5,size=100)
d2vmodel.build_vocab(doc2vec_inputs)


# In[153]:

idx = np.random.rand(len(doc2vec_inputs),1)[:,0] < .8
doc2vec_inputs = np.array(doc2vec_inputs)
doc2vec_inputs_train = doc2vec_inputs[idx]
doc2vec_inputs_test = doc2vec_inputs[~idx]
d2vmodel.train(doc2vec_inputs_train)
#d2vmodel.train_words = False
#d2vmodel.train_lbls = False


# In[174]:

for _ in xrange(10):
    d2vmodel.train(doc2vec_inputs_train)
d2vmodel.train_words = False


# In[274]:

for _ in xrange(10):
    d2vmodel.train(doc2vec_inputs_test)


# In[213]:

svc_X_train = []
svc_labels_train = []
for i in np.where(idx)[0]:
    if 'Feature' in labels_for_doc2vec[i] or 'Bug' in labels_for_doc2vec[i]:
      try:
          svc_X_train.append(d2vmodel['document_' + str(i)])
          if 'Feature' in labels_for_doc2vec[i]:
            svc_labels_train.append('Feature')
          else:
            svc_labels_train.append('Bug')
      except:
          pass


# In[265]:

d2v_pred_test = []
for i in np.where(~idx)[0]:
  if 'Feature' in labels_for_doc2vec[i] or 'Bug' in labels_for_doc2vec[i]:
    try:
      d2v_pred_test.append('Bug' if d2vmodel.similarity('document_' + str(i),'Bug') > d2vmodel.similarity('document_' + str(i),'Feature') else 'Feature')
    except:
      pass


# In[271]:

confusion_matrix(d2v_pred_test,svc_labels_test)


# In[217]:

svc_X_test = []
svc_labels_test = []
for i in np.where(~idx)[0]:
    if 'Feature' in labels_for_doc2vec[i] or 'Bug' in labels_for_doc2vec[i]:
      try:
          svc_X_test.append(d2vmodel['document_' + str(i)])
          if 'Feature' in labels_for_doc2vec[i]:
            svc_labels_test.append('Feature')
          else:
            svc_labels_test.append('Bug')
      except:
          pass


# In[262]:

np.where(~idx)


# In[275]:

d2vmodel['document_7']


# In[40]:

x = np.array([[1,1],[-1,1],[1,1]])
y = np.array([1,1]).reshape(-1,1)


# In[41]:

x


# In[42]:

y


# In[60]:

1 - np.array(np.dot(x,y.T).T/np.linalg.norm(x,axis=1)/np.linalg.norm(y.T))[0]


# In[33]:

np.linalg.norm(x,axis=1)


# In[52]:

np.linalg.norm((x - y.T),axis=1)


# In[51]:

y.T


# In[53]:

x


# In[54]:

y


# In[ ]:




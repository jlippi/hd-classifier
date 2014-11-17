from ticketClassifier import ticketClassifier
from ticketScraper import ticketScraper
import os
from pymongo import MongoClient

client = MongoClient('localhost',27017)
db = client['hd-test']

DJANGO_parms = {'tracker': 'rpc',
              'project': 'django',
              'mongo_coll': db['django'],
              'url': 'https://code.djangoproject.com/xmlrpc',
              'auth': (os.getenv('GH_USER'),os.getenv('GH_PASS'))}
DJANGO_parms['mutually_exclusive_labels'] = ['Cleanup/optimization',
                                             'Bug',
                                             'enhancement']
DJANGO_parms['labeldict'] = { 
  'Cleanup/optimization': 'Cleanup/optimization',
  'defect': 'Bug',
  'enhancement': 'enhancement',
  'bug / defect': 'Bug',
  'New feature': 'enhancement',
  'task': 'unk',
  u'd\xe9faut': 'unk',
  'Uncategorized': 'unk',
  'Bug': 'Bug',
  u'': 'unk' } 

ts = ticketScraper(DJANGO_parms)
ts.run()

tc = ticketClassifier(DJANGO_parms)
tc.run()

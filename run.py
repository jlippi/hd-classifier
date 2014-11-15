from ticketClassifier import ticketClassifier
from ticketScraper import ticketScraper
import os
from pymongo import MongoClient

client = MongoClient('localhost',27017)
db = client['hd-test']

SALT_parms = {'tracker': 'github',
              'project': 'saltstack/salt',
              'mongo_coll': db['salt'],
              'url': None,
              'auth': (os.getenv('GH_USER'),os.getenv('GH_PASS'))}
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

print 'start scraping'
ts = ticketScraper(SALT_parms)
ts.run()

print 'done scraping. start classifying'
tc = ticketClassifier(SALT_parms)
tc.run()
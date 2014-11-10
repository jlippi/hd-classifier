from pymongo import MongoClient
import requests
from requests.auth import HTTPBasicAuth
import xmlrpclib
from bson import InvalidDocument
from bs4 import BeautifulSoup
from lxml import objectify
import json
import lxml
from xmltodict import parse

class ticketScraper(object):

  def __init__(self, tracker, project, mongo_coll, url=None, auth=None):
    print tracker
    self.tracker = tracker
    self.project = project
    self.url = url
    self.auth = auth
    self.mongo_coll = mongo_coll

    if self.tracker == 'github':
      self.scraper = self.scrape_github
      if not url:
        self.url = 'https://api.github.com/repos/'
    if self.tracker == 'jira':
      self.scraper = self.scrape_jira
    if self.tracker == 'rpc':
      self.scraper = self.scrape_rpc
 
  def run(self):
    return self.scraper(start=1)

  def resume(self, start):
    return self.scraper(start=start)

  def scrape_github(self, start=1):
    page = start
    more = True
    while more:
      if page % 20 == 0:
        print 'requesting page', page
      more = self._github_loop(page)
      page += 1

  def _github_loop(self, page):
    url = self.url + self.project + '/issues?sort=updated&direction=asc&state=all&page=' + str(page)
    a = requests.get(url, auth=HTTPBasicAuth(*self.auth))
    json_list = a.json()
    if len(json_list) > 0:
      for j_obj in json_list:
        ticket = {'url': url,
                  'project': self.project,
                  'json': j_obj}
        self.mongo_coll.insert(ticket)
      return True
    return False
    
  def scrape_jira(self, start=1):
    more = True
    ticket = start
    while more:
      more = self._jira_loop(ticket)
      ticket += 1
    print 'could not get ticket #', ticket

  def _jira_loop(self, ticket):
    a = requests.get(self.url + self.project + '-' + str(ticket) + '/')
    if not a.ok:
      return False
    bs = BeautifulSoup(a.text,'html.parser')
    for item in bs.findAll('item'):
      try: 
        self.mongo_coll.insert(parse(item.encode('utf-8')))
      except:
        print 'error. call _jira_loop with ticket ' + str(ticket)
        return item
      ticket += 1
    return True

  def scrape_rpc(self, start=1):
    server = xmlrpclib.Server(self.url)
    tickets = server.ticket.query('max=0')
    for ticket in tickets[start-1:]:
      a = server.ticket.get(ticket)
      for x in a:
        if type(x) == dict:
          try:
            coll.insert(x)
          except InvalidDocument:
            for k in x.keys():
              try:
                bson.BSON.encode({k: x[k]})
              except InvalidDocument:
                x.pop(k)
            coll.insert(x)
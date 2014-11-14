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
  required_cols = ['labels','title','severity','ticket_url',
                          'repo_url','project','created_at']

  def __init__(self, parms):
    self.tracker = parms['tracker']
    self.project = parms['project']
    self.url = parms['url']
    self.auth = parms['auth']
    self.mongo_coll = parms['mongo_coll']


    if self.tracker == 'github':
      self.scraper = self.scrape_github
      if not self.url:
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
        ticket = {'repo_url': url,
          'project': self.project,
          'title': j_obj['title'],
          'body': j_obj['body'],
          'severity': 3,
          'ticket_url': j_obj['html_url'],
          'created_at': j_obj['created_at'],
          'labels': list()}
        for label in j_obj['labels']:
          ticket['labels'].append(label['name'])
        self.insert_ticket(ticket)
      return True
    return False
    
  def scrape_jira(self, start=1):
    more = True
    ticket_num = start
    while more:
      more = self._jira_loop(ticket_num)
      ticket_num += 1
    print 'could not get ticket #', ticket_num

  def _jira_loop(self, ticket_num):
    a = requests.get(self.url + self.project + '-' + str(ticket_num) + '/')
    if not a.ok:
      return False
    bs = BeautifulSoup(a.text,'html.parser')
    for item in bs.findAll('item'):
      try: 
        ticket = {'repo_url': self.url + self.project,
          'project': self.project,
          'title': item.find("title").text,
          'body': item.find("description").text,
          'severity': 3,
          'ticket_url': item.find("link").text,
          'created_at': datetime.datetime(item.find("created").text,'%a, %d %b %Y %H:%M%S +0000)'),
          'labels': list()}
        for label in item.find("labels").findAll("label"):
          ticket['labels'].append(label.text)
        self.insert_ticket(ticket)
      except:
        print 'error. call _jira_loop with ticket ' + str(ticket_num)
        return item
      ticket_num += 1
    return True

  def scrape_rpc(self, start=1):
    server = xmlrpclib.Server(self.url)
    tickets = server.ticket.query('max=0')
    for ticket_num in tickets[start-1:]:
      a = server.ticket.get(ticket_num)
      for x in a:
        if type(x) == dict:
          ticket = {'repo_url': server_url,
                    'project': self.project,
                    'title': b['summary'],
                    'body': b['description'],
                    'severity': 3,
                    'ticket_url': '',
                    'created_at': datetime.datetime.strptime(c.__str__(),'%Y%m%dT%H:%M:%S'),
                    'labels': [b['type']]
          }
          self.insert_ticket(ticket)

  def insert_ticket(self, ticket):
    for col in ticketScraper.required_cols:
      if col not in ticket:
        raise missingCol(col)
    for x in ticket:
      if not ticket[x]:
        ticket[x] = ''
        
    self.mongo_coll.insert(ticket)
    return

class missingCol(Exception):
    def __init__(self,col):
        self.args = ["missing attribute: " + col]
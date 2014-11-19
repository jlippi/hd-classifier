from flask import Flask
from flask import request
from flask import render_template
from flask import jsonify
from flask import send_from_directory
from flask import redirect
import os
import requests
import bs4
import random
import numpy as np
import pickle
import pandas as pd
import datetime
from bson import ObjectId
from pymongo import MongoClient
import pymongo
from flask.ext.restful import Resource, fields, reqparse

app = Flask(__name__)

# no '/' page for this flask app
@app.route('/')
def redir():
    return redirect('/hd-ticket')

# '/project' goes to '/hd-ticket'
@app.route('/project')
def redir2():
    return redirect('/hd-ticket')

# render the project.html template
@app.route('/hd-ticket')
def index():
    return render_template('project.html')

# this method is to change the flags on a ticket and update them in the database
@app.route('/flag', methods = ['POST'])
def flag():
    mutually_exclusive_labels = ['Cleanup/optimization',
                                             'Bug',
                                             'enhancement'] # this should match the mutualy_exclusive_labels used in run_django.py
    # get variabel from POST request
    flag = request.form['flag']
    event_id = request.form['id']
    if isalnum(event_id) or isdigit(event_id):
        a = coll.find({"_id": ObjectId(event_id)}).next()
    else:
        return None
    # get current labels
    labels = a['labels']
    # retain non 'mutually-exclusive-labels'
    labels = [l for l in labels if l not in mutually_exclusive_labels]
    # only allow update for labels in mutually_exclusive_labels
    # if this is changed, it still needs to validate labels so ticketClassifier won't blow up
    if flag in mutually_exclusive_labels:
        labels.append(flag.lower())
        a['labels']= labels
        coll.update({"_id": a['_id']},a)
    else:
        return None
    return str(a)

# get data and return it for JQUERY call
@app.route('/data', methods = ['GET'])
def data():
    return jsonify(get_data())

# get the most recent 100 tickets. return as a json object
def get_data():
    results = {'children': []}
    for entry in coll.find().sort([("created_at", pymongo.DESCENDING)]).limit(100):
        result = {'_id' : str(entry['_id']),
                  'title': entry['title'],
                  'guesses': entry['guesses'],
                  'url': '/get_ticket?raw=true&tid=' + str(entry['_id']),
                  'text': entry['title'] if entry['title'] else '' + entry['body'] if entry['body'] else '',
                  'created_at': entry['created_at'],
                  'priority': entry['severity']}
        results['children'].append(result)
    return results

# return a ticket with a particular ticket ID.
@app.route('/get_ticket', methods = ['GET'])
def get_ticket():
    parser = reqparse.RequestParser()
    parser.add_argument('tid',type=str)
    parser.add_argument('pure_json', type=bool)
    parser.add_argument('raw',type=bool)
    args = parser.parse_args()
    tid = args['tid']
    tickets = list(coll.find({'_id': ObjectId(tid)}))
    if args['raw']:
        return render_template('ticket_details.html', data=tickets)
    if args['pure_json']:
        return '\n'.join(str(t) for t in tickets)
    if len(tickets) < 1:
        return 'please specify a valid ticket ID'
    return render_template('show_ticket.html',data=tickets)

#these can be used in the jinja templates
@app.context_processor
def utility_processor():
    def equals(a,b,c):
        if a == b:
            return c
    def has_label(i, labelName):
        return labelName in i

    return dict(equals=equals,
                has_label=has_label)

if __name__ == '__main__':
    client = MongoClient('mongodb://localhost:27017/')
    db = client['hd-test']
    coll = db['django']
    app.run(host='0.0.0.0', port=8142, debug=True)

from flask import Flask
from flask import request
from flask import render_template
from flask import app
import requests
import bs4
import random
import numpy as np
import pickle
import pandas as pd
import datetime
from bson import ObjectId
from pymongo import MongoClient
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('templates/index.html')

#@app.route('/score', methods=['POST'])
#def score():
#    data = request.get_json()
#    df = pd.DataFrame(pd.Series(data)).T
#    data['prediction'] = str(''.join(fm.predict(df)))
#    data['prediction_probability'] = str(fm.predict_proba(df)[0][0])
#    data['flag'] = 'uncategorized'
#    coll.insert(data)#
#
#    return '<html>'

#@app.route('/dashboard', methods = ['GET'])
#def dashboard():
#    data = list(coll.find({"flag": "uncategorized","prediction":"fraud"}).limit(10))
#    return render_template('dashboard.html', data = data)

#@app.route('/flag', methods = ['POST'])
#def flag():
#    flag = request.form['flag']
#    event_id = request.form['id']
#    a =  coll.update({"_id": ObjectId(event_id)},{"$set":{"flag":flag}})
#    return str(a)

@app.route('/get_ticket', methods = ['GET'])
def get_ticket():
    tid = request.args.get('tid')
    tickets = list(coll.find({'_id': ObjectId(tid)}))
    return render_template('show_ticket.html',data=tickets)


@app.context_processor
def utility_processor():
    def equals(a,b,c):
        if a == b:
            return c
    return dict(equals=equals)

if __name__ == '__main__':
    client = MongoClient('mongodb://localhost:27017/')
    db = client['helpdesk']
    coll = db['github_saltstack']

    # load pickled models
    #fm = pickle.load(open('pickle_init.pkl'))

    # run application
    app.run(host='0.0.0.0', port=7000, debug=True)

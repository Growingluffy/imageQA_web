# -*- coding: utf-8 -*-

from flask import Flask
from flask import request
from flask import render_template
from models import RNN_GRU
import cPickle
import numpy as np
import pandas as pd



app = Flask(__name__, static_url_path = "/tmp", static_folder = "tmp")


# http://127.0.0.1:5000/
@app.route('/')
def my_form():
    return render_template('test.html')


@app.route('/', methods=['POST'])
def my_form_post(answer=None):
    
    filename='/home/seonhoon/Desktop/workspace/ImageQA/data/dict.pkl'
    
    with open(filename, 'rb') as fp:
        idx2word, word2idx, idx2answer, answer2idx = cPickle.load(fp)

    text = request.form['text']
    print text
    
    question=text.split()
    
    q_idx=[]
    for i in range(len(question)):
        q_idx.append(word2idx[question[i]])
    q_idx=np.array(q_idx)
    
    print q_idx
    x = np.zeros((60, 1)).astype('int32')
    x_mask = np.zeros((60, 1)).astype('float32')
    x[:len(q_idx),0] = q_idx
    x_mask[:len(q_idx),0] = 1.

    # for speed
    pd.read_pickle('/home/seonhoon/Desktop/workspace/ImageQA_Web/cnn.pkl')
    cnn_feature = np.array([pd.read_pickle('/home/seonhoon/Desktop/workspace/ImageQA_Web/cnn.pkl')['cnn_feature'][0].tolist()])
    
    n_vocab = 12047
    y_vocab = 430
    dim_word = 1024
    dim = 1024
    
    model = RNN_GRU(n_vocab, y_vocab, dim_word, dim)    
    pred_y = model.prediction(x, x_mask, cnn_feature, lr=0.001)
    
    print idx2answer[pred_y[0]]
    
    params = {'answer' : idx2answer[pred_y[0]], 'text' : text}
    
    return render_template('test.html', **params) 


    
if __name__ == '__main__':
    app.run()
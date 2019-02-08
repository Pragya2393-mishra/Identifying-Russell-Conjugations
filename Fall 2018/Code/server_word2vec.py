# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 18:15:20 2018

@author: pragy
"""

import nltk
from nltk.corpus import brown
import requests 
import nltk
from bs4 import BeautifulSoup 
from inspect import getsourcefile
from flask import Flask,  request
from flask_cors import CORS, cross_origin
from os.path import abspath, join, dirname
from urllib.parse import urlparse
from urllib.parse import urljoin
import numpy
import nltk
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from nltk.parse.stanford import StanfordDependencyParser
import os
from collections import defaultdict
java_path = "C:\Program Files (x86)\Common Files\Oracle\Java\javapath\java.exe"
os.environ['JAVAHOME'] = java_path

app = Flask(__name__)

"""
Listing word in vocabulary with their respective emotional polarity using dict_vocab
"""
def dict_vocab ():
    dictio = {}
    full_filepath = join(dirname(abspath(getsourcefile(lambda:0))), "data")
    full_filepath = join(full_filepath, "output.vocab")
    with open(full_filepath, "r", encoding="utf-8" ) as f:
        text = f.readlines()
        for i in text: 
            i = i.split()
            dictio[i[0]]=i[1]
    return dictio


"""
We look for the synonymous in the dictionary and apply a threshold.
"""
def vocab (name, word,lexicon_dictio):
    if (word in lexicon_dictio) & (name in lexicon_dictio):
        polarity = float(lexicon_dictio[word])
        polarity_sys = float(lexicon_dictio[name])
        threshold = 0.0001
        if ((polarity > threshold) & (polarity_sys > threshold)) | ((polarity< threshold) & (polarity_sys < threshold))  :
            return 1
        if ((polarity > -threshold) & (polarity < threshold)) & (polarity_sys> -threshold) & (polarity_sys < threshold)  :
            return 1 
        else: 
            return 0
    return 0
 
    
"""
We analyze each synonym of our word, in case it passes 
the filter we introduce it into our web page.
"""
def add(word, lista, POS):
    result = '<div id="text'+str(POS)+'" class="dropdown-content">'
    for i in lista:                             
        result += str('<a onclick="mytoggle(this,dropdownclass'+str(POS)+')">'+i+'</a>')
    result += '<a onclick="mytoggle2(this, '+str(POS)+')" id="last'+str(POS)+'">All</a>'
    result += str('</div></div>')
    return result

def synonyms(word, POS, lexicon_dictio, polarity):
    dictio ={}
    lista = []
    for syns in nltk.corpus.wordnet.synsets(word,nltk.corpus.wordnet.ADJ):
        for l in syns.lemmas():
            if l.name().lower() != word.lower():
                if l.name() not in dictio:
                    name = str(l.name()).replace("_", " ")
                    if polarity != 3:
                        back = vocab(name, word, lexicon_dictio)
                        if back == 0 and polarity == 0:
                            lista.append(name)
                            dictio[l.name()]=1
                        if back == 1 and polarity == 1:
                            lista.append(name)
                            dictio[l.name()]=1
                    elif polarity == 3:
                        lista.append(name)
                        dictio[l.name()]=1
    return lista

"""
Create the button in each of the adjectives that we have highlighted 
to display the synonyms
"""
def button(word, sys, number):
    text_html = '<div id="'+str(number)+'" class="dropdown">'
    text_html +='<button onclick="iitbutton(\''+str(number)+'\')"'
    text_html += 'class="dropbtn"'
    text_html += 'id="dropdownclass'+str(number)+'">'+word+sys
    text_html += "</button>"
    return text_html

def MENU(word, synset):
    text_html = '<select align=center> <option>' + word + '</option> '
    for syn in synset:
        text_html += '<option>' + syn + '</option> '
    text_html += "</select> \n"
    return text_html

def BOLD(word, args):
    if args != '':
        return "<B>" + word + "</b>" + '[' + args + ']'
    else:
        return "<B>" + word + "</b>"


"""
Look at all the words and highlight the adjectives we have.
"""                   
def highlight(url, POS_tag, polarity):
    f1 = open("src/newnew.html")
    script_soup = BeautifulSoup(f1, 'html.parser')
    f1.close()
    path1 = join(dirname(abspath(getsourcefile(lambda:0))), "GoogleNews-vectors-negative300-normed.bin")
    path1 = path1.replace('\\', '/')
    model = KeyedVectors.load(path1, mmap='r')
    model.syn0norm = model.syn0  # prevent recalc of normed vectors
    model.most_similar('stuff')
    path_to_jar= join(dirname(abspath(getsourcefile(lambda:0))), "stanford-parser-full-2018-10-17")
    path_to_jar = join(path_to_jar, "stanford-parser.jar")
    path_to_models_jar = join(dirname(abspath(getsourcefile(lambda:0))), "stanford-english-corenlp-2018-10-05-models.jar")    
    result = ''
    page = requests.get(url)
    page_html = page.content
    page_soup=BeautifulSoup(page_html, 'html.parser')
    lexicon_dictio = dict_vocab()
    print("ready to loop over document")
    number = 0
    para_list = page_soup.find_all(['p'])
    for para in para_list:
        result=''
        sent_text = nltk.sent_tokenize(para.text)
        for sent in sent_text:
            dict1={}
            tokens = nltk.word_tokenize(sent)
            tagged = nltk.pos_tag(tokens)
            position = 0
            dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
            parse = dependency_parser.raw_parse(sent)
            dep = next(parse)
            l1= list(dep.triples())
            for i in l1:
                if i[0][1]=='NN':
                    if i[2][1]=='JJ':
                        dict1[(i[2][0])]=(i[0][0])
                        
                if i[0][1]=='JJ':
                    if i[2][1]=='NN':
                        noun=i[2][0]
                        adjective=i[0][0]
                        dict1[(i[0][0])]=(i[2][0])
            for word, tag in tagged:
                if 'J' in tag:
                    if word in dict1.keys():
                        adjective= word
                        noun= dict1[word]
                        syns=[]
                        try:
                            l= model.most_similar(positive=[adjective],negative=[noun])
                            for i in l:
                                antonyms=[]
                                try:
                                    for syn in nltk.corpus.wordnet.synsets(i[0]): 
                                        for x in syn.lemmas(): 
                                            if x.antonyms(): 
                                                antonyms.append(x.antonyms()[0].name()) 
                                except:
                                    pass
                                if len(list(nltk.corpus.wordnet.synsets(i[0],nltk.corpus.wordnet.ADJ))) >0:
                                    if i[0].lower()!=adjective.lower():
                                        print(i[0],antonyms)
                                        if i[0] not in antonyms:
                                            syns.append(i[0])
                        except:
                            pass                                    
                                                
                        list_str = ';'.join(syns)
                        if len(syns) > 0:
                            text = MENU(word, syns)
                        else:
                            text = BOLD(word, '')
                    else: 
                        text = BOLD(word, '')
                    result += ' ' + text +' '
                    number += 1
                else:
                    result += word + ' '
                position +=1
        para.contents = BeautifulSoup(result, 'html.parser')
    div= page_soup.new_tag('div')
    a_str='''
            <div class=pragya>
            <form method="POST"> 
            <input name="text">
            <input type="submit">
            </form>
            </div>
            <style>
            div.pragya {
                    position: fixed;
                    bottom: 0;
                    border: 1px solid black;
                    z-index:1;
                  } 
            </style>
            '''
    div.append(BeautifulSoup(a_str, 'html.parser'))
    page_soup.append(div)
    return str(page_soup)

"""
Parameters of the server where we collect the variables that we need.
"""
#@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
@app.route('/')
def hello():
    print("Hello, World!")
    return("Hello!")
app.add_url_rule('/','hello',hello)


@app.route('/worker',methods=['GET', 'POST']) ####
def worker():
    print("Worker")
    if request.method == 'POST':    
        POS = 'NN'
        polarity = 3.0
        src_url=request.form['text']
        new_html = highlight(src_url, POS, polarity)
        #print(new_html)
        
        return new_html
    
    return '''<form method="POST"> 
            <input name="text">
            <input type="submit">
            </form>'''


@app.errorhandler(404)
def page_not_found(error):
    return 'ACK!! This page does not exist', 404

@app.errorhandler(400)
def page_not_found(error):
    return 'Could not understand: ' + str(error), 400

def exception_handler(*args):
    print("BAD REQUEST:")
    print(request.url)
    return "Something wrong happened", 400

if __name__ == '__main__':

    app.run(debug=True,use_reloader=False)

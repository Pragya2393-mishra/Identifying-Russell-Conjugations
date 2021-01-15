# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 18:15:20 2018

@author: pragy
"""
# Importing libraries
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import requests 
from bs4 import BeautifulSoup 
from inspect import getsourcefile
from flask import Flask,  request
from flask_cors import CORS, cross_origin
from os.path import abspath, join, dirname
from urllib.parse import urlparse
from urllib.parse import urljoin
import numpy
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from nltk.parse.stanford import StanfordDependencyParser
import os
from collections import defaultdict
import re

#Setting up environment
java_path = "C:\Program Files\Common Files\Oracle\Java\javapath\java.exe"
os.environ['JAVAHOME'] = java_path
os.chdir("C:\\Users\\pragy\\RS\\Fall 2018\\Code")
f1 = open("src/new1.html")
script_soup = BeautifulSoup(f1, 'html.parser')
f1.close()

#Loading pre-trained vectors
path1 = join(dirname(abspath(getsourcefile(lambda:0))), "GoogleNews-vectors-negative300-normed.bin")
path1 = path1.replace('\\', '/')
model = KeyedVectors.load(path1, mmap='r')
model.syn0norm = model.syn0  # prevent recalc of normed vectors
model.most_similar('stuff') #test

#loading POS taggers
path_to_jar= join(dirname(abspath(getsourcefile(lambda:0))), "stanford-parser-full-2018-10-17")
path_to_jar = join(path_to_jar, "stanford-parser.jar")
path_to_models_jar = join(dirname(abspath(getsourcefile(lambda:0))), "stanford-english-corenlp-2018-10-05-models.jar")    
ps =PorterStemmer()

app = Flask(__name__)


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
    #for syn in synset:
    text_html += '<option>' + synset + '</option> '
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
def highlight(url):
    result = ''
    page = requests.get(url)
    page_html = page.content
    page_soup=BeautifulSoup(page_html, 'html.parser')
    print("ready to loop over document")
    number = 0
    para_list = page_soup.find_all('h1',class_='pg-headline')# for title
    #para_list = page_soup.find_all('div',class_='zn-body__paragraph')# for para in cnn
    
    #tokenize text
    for para in para_list:
        result=''
        sent_text = nltk.sent_tokenize(para.text)
        for sent in sent_text:
            dict1={}
            tokens = nltk.word_tokenize(sent)
            
            #POS tagging
            tagged = nltk.pos_tag(tokens)
            position = 0
            dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
            parse = dependency_parser.raw_parse(sent)
            dep = next(parse)
            l1= list(dep.triples())
            
            #Extracting Adjective-Noun and Verb Noun pairs
            dict1 = dict()
            for i in range(len(l1)):
                print(l1[i][0],l1[i][2])
                t1 = l1[i][0]
                t2 = l1[i][2]             
                if('NN' in t1[1] and 'JJ' in t2[1]) or ('NN' in t1[1] and 'VB' in t2[1]):
                    print("case 1")
                    dict1.update({t2[0]:t1[0]})                 
                elif('JJ' in t1[1] and 'NN' in t2[1]) or ('VB' in t1[1] and 'NN' in t2[1]):
                    print("case 2")
                    dict1.update({t1[0]:t2[0]})


            for word, tag in tagged:
                if word in dict1.keys():
                    
                    #String Normalization
                    adjective= word.strip().lower()
                    adjective=adjective.replace("-", "_") # for example, fact-checking to fact_checking
                    adjective = re.sub('[^a-zA-Z_]', '', adjective) #Extracting only the alphabetical part (and _) from the word
                    print(adjective)
                    noun= dict1[word].strip().lower()
                    noun=noun.replace("-", "_") # for example, fact-checking to fact_checking
                    noun = re.sub('[^a-zA-Z_]', '', noun) #Extracting only the alphabetical part (and _) from the word
                    print(noun)
                    
                    #Extracting contexually and denotationally similar words
                    syns=[]
                    try:
                        l= model.most_similar(positive=[adjective],negative=[noun],topn=10)
                        #print('printing most similar words:'+'/n')
                        print(l)
                        for i in l:
                            #Filter out words with same stem
                            if ps.stem(i[0].strip().lower())==ps.stem(adjective):
                                continue
                            
                            #Filter out words with same lemma
                            elif (WordNetLemmatizer().lemmatize(i[0].strip().lower(),'v'))== (WordNetLemmatizer().lemmatize(adjective,'v')):
                                 continue
                            elif i[0].strip().lower() in adjective:
                                continue
                            elif adjective in i[0].strip().lower():
                                continue
                            
                            else:
                                print(i[0].strip().lower()+'ok')
                                
                                #Filter out antonyms
                                antonyms=[]
                                try:
                                    for syn in nltk.corpus.wordnet.synsets(i[0]):
                                        #print(syn)                                    
                                        for x in syn.lemmas(): 
                                            #print(x)
                                            if x.antonyms(): 
                                                #print('TRUE for antonyms')
                                                antonyms.append(x.antonyms()[0].name()) 
                                except:
                                    print("In pass loop")
                                    pass
                                if len(list(nltk.corpus.wordnet.synsets(i[0]))) >0:
                                    if i[0].lower()!=adjective.lower():
                                        print(i[0],antonyms)
                                        if adjective not in antonyms:
                                            syns.append(i[0])
                    except:
                        pass   
                    if len(syns)>0:
                            syns=syns[0]                        
                    print('syns: ******************')
                    print(syns)
                    if len(syns) > 0:
                        text = MENU(word, syns)
                    else:
                        text = BOLD(word, '')
                else: 
                    text = word
                result += ' ' + text +' '
                number += 1
            # else:
            #     result += word + ' '
            position +=1
        #Final content
        para.contents = BeautifulSoup(result, 'html.parser')
    
    #Putting back together
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
        src_url=request.form['text']
        new_html = highlight(src_url)
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

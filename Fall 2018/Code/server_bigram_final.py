# -*- coding: utf-8 -*-
import requests 
import nltk
from bs4 import BeautifulSoup 
from inspect import getsourcefile
from flask import Flask,  request
from flask_cors import CORS, cross_origin
from os.path import abspath, join, dirname
from urllib.parse import urlparse
from urllib.parse import urljoin
import pickle
import os
from collections import defaultdict
from nltk.parse.stanford import StanfordDependencyParser
java_path = "C:\Program Files (x86)\Common Files\Oracle\Java\javapath\java.exe"
os.environ['JAVAHOME'] = java_path

#app = Flask(__name__, static_url_path='/settings')
app = Flask(__name__)
#app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
#app.config['CORS_HEADERS'] = 'Content-Type'

#cors = CORS(app, resources={r"/foo": {"origins": "http://localhost:port"}})

"""
We have to take out the vocabulary and the result of our analyzer to 
filter all the synonyms we get
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
Translate to the language used by nltk
"""
def translate(POS):
    if POS == 0:
        return 'JJ'
    elif POS == 1:
        return 'NN'
    elif POS == 2:
        return 'RB'
    return 'JJ'

       
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
    syn=synset
    #for syn in synset:
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
    path_to_jar = 'F:\\somebody gonna get hurt\\stanford-parser-full-2018-10-17\\stanford-parser.jar'
    path_to_models_jar = 'F:\\somebody gonna get hurt\\stanford-english-corenlp-2018-10-05-models.jar'
     
    d=pickle.load(open("F:\\somebody gonna get hurt\\file1.pkl","rb"))
    
#    page_soup=BeautifulSoup("<html>data</html>")
#    page_soup= page_soup.new_tag('div')
#    a_str='''<form method="POST"> 
#            <input name="text">
#            <input type="submit">
#            </form>'''
#    page_soup.append(BeautifulSoup(a_str, 'html.parser'))

    result = ''
    page = requests.get(url)
    page_html = page.content
    page_soup=BeautifulSoup(page_html, 'html.parser')
#    print(page_soup.prettify())
#    print('\n\n******\n\n')

#    header.replaceWith(BeautifulSoup(headerText,'html.parser'))

    # Add script to header
#    header.insert(0,base_soup)
    # Load vocabulary

    lexicon_dictio = dict_vocab()
    # Find text (P elements) and replace selected words with menu buttons
    print("ready to loop over document")
    number = 0

#    return str(page_soup)
    para_list = page_soup.find_all(['p'])
    for para in para_list:
        
#        print("FOUND P tag")
#        result = '''<form method="POST"> 
#            <input name="text">
#            <input type="submit">
#            </form>'''
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
                        #adjective= word
                        noun= dict1[word]
                        syns=''
                        try:
                            syns=d[noun]
                            
                        except:
                            pass
                    list_str = ';'.join(syns)
                    if len(syns) > 0:
#                        text = button(word, sys1, number)
                        text = MENU(word, syns)
                    else: 
                        text = BOLD(word, '')
                    result += ' ' + text +' '
                    number += 1
                else:
#                    result += word + '/' + tag + ' '
                    result += word + ' '
                position +=1
#        para.replaceWith()
        para.contents = BeautifulSoup(result, 'html.parser')

#    soup1.body.append(soup)
#     header = page_soup.find('head')
#     print(str(script_soup))
#     header.append(script_soup)
#     print("2")
# #    urlParse = urlparse(url)
# #    urlBase  = urlParse.scheme + "://" + urlParse.netloc
#     base_string = "<base href=\"" + url + "\"/>"
#     base_soup = BeautifulSoup(base_string,'html.parser')
#     print(str(base_soup))

##    header.append(base_soup)    # 
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
#    return page_soup.prettify()
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
    if request.method == 'POST': ####
#        src_url = request.args.get('url')
#        POS = request.args.get('position')
#        polarity = request.args.get('polarity')
#        print(src_url)
#        print(POS)
#        print(polarity)
        POS = 'NN'
        polarity = 1.0
        #src_url="https://en.wikipedia.org/wiki/WordPress"
        src_url=request.form['text'] ####
        new_html = highlight(src_url, POS, polarity)
        print(new_html)
        
        return new_html
    
    return '''<form method="POST"> 
            <input name="text">
            <input type="submit">
            </form>'''
#app.add_url_rule('/','worker',worker)

#@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
@app.route('/worker1', methods = ['GET', 'POST'])
def worker1():
    print("Worker1")
    src_url = request.args.get('url')
    POS = request.args.get('position')
    polarity = request.args.get('polarity')
    POS = 'JJ'
    polarity = 1.0
    f1 = open("new.js")
    soup_js = BeautifulSoup(f1, 'html.parser')
    f1.close()
#    script = soup_js.find('script')
#    print(script)
    return soup_js.prettify()
app.add_url_rule('/','worker1',worker1)

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
#    from flask_plugins import PluginManager
#
#    plugin_manager = PluginManager(app)

 

#/usr/bin/env python
# -*- coding:utf-8 -*-

from flask import Flask, request
import json 
from flask.ext.cors import CORS

from datetime import datetime
from elasticsearch import Elasticsearch
import demjson

app = Flask(__name__)
CORS(app)
es = Elasticsearch() 

domains = [u"بحث عام" , u"الكلمات المتقاربة", u"4"]
index2 = "words-index"
index = "quran-index"
tachkil =   [u'ٍ'  ,u'ِ',  u'ْ', u'ٌ' , u'ُ' , u'ً' , u'َ', u'ّ']
huruf = [ u"ال", u"و", u"ف", u"ب" , u"ك", u"لل", u"ل", u"ا", u"اف", u"وال", u"فال", u"بال", u'كال', u"تال", u"وَ"]

@app.route('/words', methods=['POST'])
def search_words():
    print request.form
    new_words = []
    post =  demjson.decode(list(request.form)[0])
     
    if  post[u"domain"] == domains[0] : 
        
        if post[u"query"][u"query"]  != u"":
            w_body = {u"query":  { u"bool": { 
                                 u"must":{u"query_string" :{u"query" : post[u"query"][u"query"]} },
                     }}}
            allwords =  es.search(index=index2, doc_type="words",  body = w_body)
            try : 
                #print allwords 
                id = allwords[u"hits"][u"hits"][0][u"_id"]
                #print id 
                allwords = allwords[u"hits"][u"hits"][0][u"_source"][id]
                allwords = list(set(allwords)) 
                #for w in allwords :
                #    print w
            except : 
                allwords = [post[u"query"][u"query"]]
                    
            new_words = list(set (search_word_with_val(allwords, es )+allwords))     
            new_words = [ w for w in new_words if [t for t in tachkil if t in w  ] != [] ]

            
        return json.dumps(new_words, ensure_ascii=False) ;
    
    
    if  post[u"domain"] == domains[1] :
        
         pass 
     
     
   
def search_word_with_val (words, es  ):
   
    
    new_words = [] 
    for word in words : 
            for  h in huruf :
                allwords = [] 
                word1 = h+word  
                w_body = {u"query": {u"bool": { 
                                u"must":{u"query_string" :{u"query" :word1} }}
                                     } }
                w =  es.search(index=index2, doc_type="words", 
                                  body = w_body)
                
                if w[u"hits"][u"hits"] != []:  
                    
                    id = w[u"hits"][u"hits"][0][u"_id"]
                    allwords = w[u"hits"][u"hits"][0][u"_source"][id]  
                    new_words += allwords     
    return new_words         
        
 
 
        
@app.route('/ayats', methods=['POST'])
def search_ayats():
    post =  demjson.decode(list(request.form)[0])
    ###################################################
    #  output data : 
    time_citation = 0 
    percent_appear = 0 
    surats = []
    allwords = []
    ##############
    if  post[u"domain"] == domains[0] : 
        for word in post[u"query"] : 
            w_body = {u"size" : 7000, u"query": {
                                                  u"bool": { 
                                                        u"must":{u"query_string" :{u"query" :word} },   
                                                        u"must_not" :  { u"term": { u"السورة": word}}
                                                   }
                                                }} 
            #print w_body
            res =  es.search(index=index, doc_type="ayats", 
                                  body = w_body)

            time_citation += len( res[u"hits"][u"hits"])
        
            allwords += res[u"hits"][u"hits"]#[:][u"_source"]  
            
         
        makia = 0 
        madania = 0     
        
        for p in allwords :
            i=0 
            flag = False 
            for  s in surats : 
                if s[u"السورة"]  ==p[u"_source"][u"السورة"] :
                    surats[i][u"الآية"].append (p[u"_source"][u"الآية"] )
                    flag = True  
                    
                i+=1 
            if not flag : 
                  surats.append({ u"السورة": p[u"_source"][u"السورة"] ,
                                u"مكان_النزول" : p[u"_source"][u"مكان_النزول"] ,
                                u"ترتيب_التنزيل" : p[u"_source"][u"ترتيب_التنزيل"],
                                u"الترتيب" : p[u"_source"][ u"الترتيب" ],
                                u"الآية" : [p[u"_source"][u"الآية"]]
                               } ) 
                
            if p[u"_source"][u"مكان_النزول"] ==   u"مَكيَّة"  :
                makia +=1 
            else : 
                madania +=1 
                
        
    surats =  sorted(surats, key=lambda k: len( k[u"الآية"]),  reverse=True )
    allwords  = sorted(allwords, key=lambda k: (k[u"_source"][u"الترتيب"], k[u"_source"][u"الآية"]) )
  
    '''print surats 
    print time_citation  
    '''
    for w in allwords : 
        print w
        
        
    
    return  json.dumps( {"time_citation":time_citation  , "makia": makia, "madania": madania, "list_surats": surats, "data": allwords }, ensure_ascii=False);   #json.dumps(allwords, ensure_ascii=False) ;

       
if __name__ == '__main__':
    
    app.run(port="5000")
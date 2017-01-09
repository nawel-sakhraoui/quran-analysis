#!/usr/bin/env python
# -*- coding: utf-8 -*-

##############################################
#####
##### Natural Language processing 
##### Indexing dataset in ElasticSearch
##### 
#####Author Naouel Sakhraoui 
###############################################




from datetime import  datetime
from elasticsearch import Elasticsearch
import os 
import pandas.io.parsers as parser
import numpy as np
from _codecs import encode
import json 
parent_rep = os.path.abspath(os.path.join( os.path.abspath(os.path.join( os.path.dirname(os.path.realpath(__file__)), os.pardir)), os.pardir))
es = Elasticsearch()
#es.cluster.health(wait_for_status='yellow')

#indexing surats 


surat_rep = parent_rep+"/data/surat.csv"
surat_dict = []
list_keys = []
flag = True 

with open(surat_rep ) as f:
    for line in f:
        if flag :
            list_keys = line.split(',')
        else : 
            temp_f = line.split(",")
            temp_dict = {}
            for i in range (len(temp_f)) : 
                temp_dict.update({ list_keys[i]: temp_f[i]})
                
            surat_dict.append(temp_dict)
            
        flag = False

       
#put surat_dict in elasticsearch  
#read data set 
#for i in range (len(surat_dict)) : 
#    #res = es.index(index="quran-index", doc_type='surats', id=i+1 , body=surat_dict[i])




index_list = [  u"مكان_النزول", u"ترتيب_التنزيل" ,u"السورة" ,u"الآية",    u"الترتيب"  ]


tachkil =   [u'ٍ'  ,u'ِ',  u'ْ', u'ٌ' , u'ُ' , u'ً' , u'َ', u'ّ', u'ٰ',u'َّ' , u'ِّ',  u'ُّ',  u'ٰ']
#hurufatf = [u"و", u"ف", u"ثم", u"حتى", u"أو", u"أم", u"بل", u"لا", u"لكن", u"أ", u"ب", u"ل", u"ال"]
hamza = [ u"أ", u"إ", u"آ", u"ٱ" ]
#tachkil = tachkil+ hurufatf 

#tachkil2 = u"ًٌٌٍَُِْ"
#indexing ayat : 
dataset_rep = parent_rep+"/data/dataset.csv"
flag = True 
ix = 0 
colums = []
tachkil_dict = {}
hamza_dict = {}
with open(dataset_rep ) as f:
    for line in f:
        if flag : 
            columns = line.decode('string_escape').decode('UTF-8').split(",")
            flag = False 
            #print columns 
        else : 
     
    
            line = line.decode('string_escape').decode('UTF-8')
            temp_dict = {} 
            data = line.split(",")
            
            
            
            #print line  
            
            for i in range (len(data)) :
                
                if data[i]!= u"0" and  data[i]!= u"" and columns[i] !=u"" : 
                
                    c = columns[i].strip(' \t\n\r')
                    c2 = c
                    c3 = c2 
                    for t in tachkil : 
                        if t in c2 : 
                            c2= c2.replace (t, u"")
                    for h in hamza : 
                         if h in c2 : 
                             c3 = c2.replace(h, u"ا") 
                            
                    
                    d  = data[i].strip(' \t\n\r')
                    d2 = d 
                    d3=d2 
                    for t in tachkil : 
                        if t in d2 : 
                            d2= d2.replace (t, u"")
                    
                    for h in hamza : 
                         if h in d2 : 
                             d3 = d2.replace(h, u"ا") 
                    
                    
                    if  c in index_list :
                        temp_dict.update( {  c : d  } )
 
                        """try: 
                            int(d)
                            pass 
                        except :
                             
                            if d2 in tachkil_dict.keys(): 
                                if d not in tachkil_dict[d2] :
                                    tachkil_dict[d2].append( d)  
                            else : 
                                tachkil_dict.update({d2: [d]})
                        
                            if d3 !=d2 and d3 not in tachkil_dict : 
                                tachkil_dict[d2].append (d3)                        
                            pass """
                        
                    else :
                      
                        temp_dict.update( { d : c } )
                
                        if c2 in tachkil_dict.keys(): 
                            if c not in tachkil_dict[c2] :
                                tachkil_dict[c2].append( c )
                        else : 
                            tachkil_dict.update({c2: [c]})
                          
                        if c3 !=c2  and  c3 not in  tachkil_dict : 
                                tachkil_dict[c2].append (c3)
                                
                    '''print "______________"
                    print d
                    print d2 
                    print c
                    print c2 '''
                    
            
            
            #print temp_dict 
            #res = es.index(index="quran-index", doc_type='ayats', id= ix,  body=  temp_dict)
            ix +=1 
            
            
  
print  len (tachkil_dict.keys()) 
   
        
mapping ='''{
"settings": {
    "index.mapping.total_fields.limit": 100000,
    "number_of_shards": 1,
    "number_of_replicas": 0
  }
}'''
    
res = es.indices.create(index='words-index', body=mapping )    
    

ix = 0 
for p in tachkil_dict.keys() : 
    #print p 
    # print tachkil_dict[p]
    #print ix 
    #for i in tachkil_dict[p]:
    #    print i 
    tachkil_dict[p].append(p)
    res = es.index(index="words-index", doc_type='words',id = ix,  body= {ix:tachkil_dict[p]}, request_timeout=60)
    ix+=1  
    

print "done"
      

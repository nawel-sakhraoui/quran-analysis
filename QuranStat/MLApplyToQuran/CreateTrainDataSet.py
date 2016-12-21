#!/usr/bin/python
# -*- coding: utf-8 -*-

##############################################
#####
##### Natural Language processing 
##### Indexing dataset in ElasticSearch
##### 
#####Author Naouel Sakhraoui 
###############################################


import os 
import pandas.io.parsers as parser
import numpy as np 
import pandas as pd 
import json 





parent_rep = os.path.abspath(os.path.join( os.path.abspath(os.path.join( os.path.dirname(os.path.realpath(__file__)), os.pardir)), os.pardir))
print parent_rep

#surats  edit
surat_rep = parent_rep+"/data/surat.csv"

surat_data =  parser.read_csv(surat_rep, sep=",", encoding="utf8")#, header= 0, encoding="utf8")
                            
#words
unique_val_list  = []
#ayat_rep = parent_rep+"/data/ayat.csv"
ayat_rep = parent_rep+"/data/quran-simple-min.txt"
size = 0 
with open(ayat_rep ) as f:
    for line in f:
        line = line.decode('string_escape').decode('UTF-8')#encode('utf-8')
        #print line 
        #line = line.decode('UTF-8','ignore').encode("UTF-8")
        #print line 
        line = ''.join(line.splitlines())
         
        size +=1
        l = list(np.unique(np.array(np.array(line.split(" ")))) )
        l = [p.strip(' \t\n\r') for p in l  ]
        unique_val_list = list(np.unique(np.array(unique_val_list + l) ))
 
print "size of words  1"        
print len(unique_val_list)       
unique_val_list = [p for p in unique_val_list if p!=u"" and p!=u"\n" and p!=u"\t" ]         
print "size of words  " 
print len(unique_val_list)


#dataset 
data_set = pd.DataFrame("0", index = np.arange( size), columns = unique_val_list + [ u"مكان_النزول", u"ترتيب_التنزيل" ,u"السورة" ,u"الآية" ,u"الترتيب"], dtype='object')      
i = 0 
with open(ayat_rep ) as f:
    for line in f:
        line = line.decode('string_escape').decode('UTF-8')
        #line = line.decode('UTF-8','ignore').encode("UTF-8")
        line = ''.join(line.splitlines())
        
        j = 1
        for word in  line.split(" ") : 
            word = word.strip(' \t\n\r') 
            #word= word.decode('UTF-8','ignore').encode("UTF-8")
            data_set.ix[i][word]= j
            j+=1 
        i =i+1
        
        


all_words  = unique_val_list 
for p in  surat_data[u"السورة" ] : 
    if p not in all_words : 
        all_words.append(p)
    
    
print "all words :"
print len (all_words)

with open(parent_rep+"/data/allwords.json", 'w') as outfile:
    json.dump(all_words, outfile, encoding='utf-8' )

'''

surat_set = pd.DataFrame(0, index = np.arange(size), columns = [ "مكان_النزول" , "ترتيب_التنزيل" ,"السورة" ])
compteur = 0 
for  c in np.arange(114) : 
    i = surat_data.iloc[[c]]  
    ayat = 1 
    for p in range(compteur, (int(i["عدد_آياتها"])+compteur) ):
        
            
        surat_set["الآية"]  =  ayat 
        ayat = ayat + 1 
        surat_set["السورة"][p] =  int(i["الترتيب"])
        surat_set["ترتيب_التنزيل"][p] =  i["ترتيب_التنزيل"]
    
        if ( str("مكية")==str(i["مكان_النزول"])) :
            surat_set["مكان_النزول"][p] =   str(1)
        else :
            surat_set["مكان_النزول"][p] =   str(2)
         
    compteur = int(i["عدد_آياتها"])+compteur

dataframe = pd.concat([data_set, surat_set], axis=1)
print dataframe.shape
print "saving  traindataset  outputs"
dataframe.to_csv( parent_rep+"/data/traindataset.csv", sep =",", encoding='utf-8')
'''

#surat_set2 = pd.DataFrame("0", index = np.arange(size), columns = [ "مكان_النزول" , "ترتيب_التنزيل" ,"السورة" , "الترتيب"])
compteur = 0 
print data_set.shape 

for  c in np.arange(114) : 
    i = surat_data.iloc[[c]]  
    
    #print i 
    
    ayat = 0
    for p in range(compteur, (int(i.ix[c][u"عدد_آياتها"])+compteur) ):
        
         
      
        
        data_set.ix[p][u"الآية"]  =  ayat+1  
        
        data_set.ix[p][u"السورة"] =   i.ix[c][u"السورة"] 
        data_set.ix[p][u"الترتيب"]  =  i.ix[c][u"الترتيب"]
        data_set.ix[p][u"ترتيب_التنزيل"] =  i.ix[c][u"ترتيب_التنزيل"]
        data_set.ix[p][u"مكان_النزول"] = i.ix[c][u"مكان_النزول"]
       
      
        #print ayat 
        #print data_set.ix[p][u"الترتيب"] 
        ayat = ayat + 1
        
    
    compteur = int(i.ix[c][u"عدد_آياتها"])+compteur
    

print data_set.shape
#dataframe2 = pd.concat([data_set, surat_set2], axis=1)
#print dataframe2.shape
print "saving dataset  outputs"
data_set.to_csv( parent_rep+"/data/dataset.csv", sep =",", encoding='UTF-8')




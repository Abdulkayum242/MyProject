# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 23:27:01 2021

@author: HEMANT HATANKAR
"""
############################################################
#Import essential Python Libraries
############################################################
import re 
from nltk.tokenize import word_tokenize
from collections import Counter
import time
import orjson
import random

############################################################
#Reading probability files for Uni-gram, Bi-gram & Tri-gram
############################################################
with open('uni_dict_prob.json', 'r') as fp:
    uni_dict=orjson.loads( fp.read())
    
with open('bi_dict_prob.json', 'r') as fp:
    bi_dict=orjson.loads( fp.read())
    
with open('./data/ngrams/tri_dict_prob.json', 'r') as fp:
    tri_dict=orjson.loads( fp.read())

############################################################
#Text cleaning
############################################################
def extra_space(text):
    new_text= re.sub("\s+"," ",text)
    return new_text

def sp_charac(text):
    new_text=re.sub("[^0-9A-Za-z ]", "" , text)
    return new_text

def tokenize_text(text):
    new_text=word_tokenize(text)
    return new_text

############################################################
# Uni Prediction
# This function is used when we have only one preceding word
############################################################
def unipred(word): 
    if word not in uni_dict.keys(): 
        #if that word does not exist in our dictionary then we predict some random word
        preds=Counter(uni_dict[random.choice(list(uni_dict.keys()))]).most_common()[:3]
        for k,v in preds:
            print(k ,':' ,v)
    else:
        preds=Counter(uni_dict[word]).most_common()[:3]
        for k,v in preds:
            print(k ,':' ,v)
            
############################################################
# Bi Prediction
# This function is used when we have only 2 preceding words 
############################################################
def bipred(word1,word2):
    '''
    if that phrase does not exist in our dictionary then we move to the lower gram to make a prediction
    '''
    if word1+" "+word2 not in bi_dict.keys():
        unipred(word2)
    else:
        preds=Counter(bi_dict[word1+" "+word2]).most_common()[:3]
        for k,v in preds:
            print(k ,':' ,v)    
            
############################################################
# Tri Prediction
# This function is used when we have only 3 preceding words 
############################################################
def tripred(word1,word2,word3):
    '''
    if that phrase does not exist in our dictionary then we move to the lower gram to make a prediction
    '''
    if word1+" "+word2+" "+word3 not in tri_dict.keys():
        bipred(word2,word3)
    else:
        preds=Counter(tri_dict[word1+" "+word2+" "+word3]).most_common()[:3]
        for k,v in preds:
            print(k ,':' ,v)  
            
############################################################
# Multi Prediction
# This function is used when we have more hhan 3 preceding words 
############################################################
def multipred(tokens):
    '''
    We take the last 3 words as history for prediction
    '''
    last_3=tokens[-3:]
    tripred(last_3[0],last_3[1],last_3[2])

######################################################################################
# In this function we take input , clean that data and convert them into tokens.
# and then depending on the numer of tokens we have we use it to predict the next word.
######################################################################################
def predictNextWord():
    while(True):
        print("Enter Text to Test: ")
        text=input()
        start=time.time()
        cleaned_text=extra_space(text)
        cleaned_text=sp_charac(cleaned_text)
        tokenized=tokenize_text(cleaned_text)
        if len(tokenized)==1:
            unipred(tokenized[0])
        elif len (tokenized)==2:
            bipred(tokenized[0],tokenized[1])
        elif len(tokenized)==3:
            tripred(tokenized[0],tokenized[1],tokenized[2])
        else:
            multipred(tokenized)
        print('Time Taken: ',time.time()-start)
        
if __name__ == "__main__":
    predictNextWord()

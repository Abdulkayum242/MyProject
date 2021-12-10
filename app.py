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
with open('./data/ngrams/uni_dict_prob.json', 'r') as fp:
    uni_dict=orjson.loads( fp.read())
    
with open('./data/ngrams/bi_dict_prob.json', 'r') as fp:
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
def unipred(word,num_words): 
    if word not in uni_dict.keys(): 
        #if that word does not exist in our dictionary then we predict some random word
        preds=Counter(uni_dict[random.choice(list(uni_dict.keys()))]).most_common()[:num_words]
        # for k,v in preds:
        #     print(k ,':' ,v)
        return preds
    else:
        preds=Counter(uni_dict[word]).most_common()[:num_words]
        # for k,v in preds:
        #     print(k ,':' ,v)
        return preds
            
############################################################
# Bi Prediction
# This function is used when we have only 2 preceding words 
############################################################
def bipred(word1,word2,num_words):
    '''
    if that phrase does not exist in our dictionary then we move to the lower gram to make a prediction
    '''
    if word1+" "+word2 not in bi_dict.keys():
        unipred(word2,num_words)
    else:
        preds=Counter(bi_dict[word1+" "+word2]).most_common()[:num_words]
        # for k,v in preds:
        #     print(k ,':' ,v)
        return preds
            
############################################################
# Tri Prediction
# This function is used when we have only 3 preceding words 
############################################################
def tripred(word1,word2,word3,num_words):
    '''
    if that phrase does not exist in our dictionary then we move to the lower gram to make a prediction
    '''
    if word1+" "+word2+" "+word3 not in tri_dict.keys():
        bipred(word2,word3,num_words)
    else:
        preds=Counter(tri_dict[word1+" "+word2+" "+word3]).most_common()[:num_words]
        return preds
        # for k,v in preds:
        #     print(k ,':' ,v)  
            
############################################################
# Multi Prediction
# This function is used when we have more hhan 3 preceding words 
############################################################
def multipred(tokens,num_words):
    '''
    We take the last 3 words as history for prediction
    '''
    last_3=tokens[-3:]
    preds = tripred(last_3[0],last_3[1],last_3[2],num_words)
    return preds

######################################################################################
# In this function we take input , clean that data and convert them into tokens.
# and then depending on the numer of tokens we have we use it to predict the next word.
######################################################################################
def predictNextWord(text, num_words):
    print("Enter Text to Test: ")
    start=time.time()
    cleaned_text=extra_space(text)
    cleaned_text=sp_charac(cleaned_text)
    tokenized=tokenize_text(cleaned_text)
    if len(tokenized)==1:
        preds = unipred(tokenized[0],num_words)
    elif len (tokenized)==2:
        preds = bipred(tokenized[0],tokenized[1],num_words)
    elif len(tokenized)==3:
        preds = tripred(tokenized[0],tokenized[1],tokenized[2],num_words)
    else:
        preds = multipred(tokenized,num_words)
    print('Time Taken: ',time.time()-start)
    return preds
if __name__ == "__main__":
    predictNextWord()
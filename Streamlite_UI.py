# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 15:27:57 2021

@author: HEMANT HATANKAR
"""
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PredictNextWord import predictNextWord
###################################################################
# Code for Sidebar
###################################################################
image = Image.open('./images/Title1.jpg')
st.sidebar.image(image, caption='Machine Learning & NLP')
st.sidebar.title("Project Name:")
st.sidebar.write("Next Word Prediction using N-grams")

st.sidebar.title("Techniques Used:")
st.sidebar.write("1. NLP(Text Preprocessing & Ngrams)\n2. Python\n3. Probability Concept")

st.sidebar.title("Developed By:")
st.sidebar.write("1. Grishma Shah (11)\n2. Pratiksha Karanjkar (23)\n3. Hemant Hatankar (03)")

st.sidebar.title("Under Guidance of:")
st.sidebar.write("Prof. Cyrus Lentin")

###################################################################
# Code for Main Content
###################################################################
# st.title("Research Project: Next Word Prediction")

col1, col2 = st.columns(2)
###################################################################
# Code for COLUMN_1 Content
###################################################################
col1.markdown('**Please select the number of suggestions you want in Result:**.')
num_words = col1.slider("",0, 15, 5)
input_text = col1.text_input(label="Enter the Text:",autocomplete="on")
print("input_text: ",input_text)
if input_text != "":
    col1.markdown('**Input Text:**.')
    col1.markdown(input_text)
    col1.markdown('**Predicted Words:**.')
    predicted_word = predictNextWord(input_text, num_words)
    # predicted_word = [('solution', 1.1842822065546072e-05), ('big', 1.1842822065546072e-05), ('point', 1.1842822065546072e-05), ('best', 7.895214710364048e-06), ('old', 7.895214710364048e-06)]
    if predicted_word:
        predicted_word_dic = {}
        for k,v in predicted_word:
            predicted_word_dic[k] = v 
        print("predicted_word: ",predicted_word)
        # col1.markdown('**Input Text:**.')
        # col1.markdown(input_text)
        
        # col1.markdown('**Predicted Words:**.')
        
        for index,val in enumerate(list(predicted_word_dic.keys())):
            pred_words_str = "["+str(index)+"] "+str(val) 
            col1.markdown(pred_words_str)
        ###################################################################
        # Code for COLUMN_1 Content
        ###################################################################
        # col2.header("Select #words for suggestion:")
        
        
        col2.markdown('**Word Cloud for Predicted Words:**.')
        wordcloud = WordCloud(background_color = "white")
        wordcloud.generate_from_frequencies(frequencies=predicted_word_dic)
        wc = plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
        # fig, ax = plt.subplots()
        
        col2.pyplot(wc)
        
        col2.markdown('**Bar Chart for Predicted Words:**.')
        
        y = list(predicted_word_dic.keys())
         
        # getting values against each value of y
        x = list(predicted_word_dic.values())
        hor_bar_chart = plt.figure()
        plt.barh(y, x)
         
        # setting label of y-axis
        plt.ylabel("Predicted Words")
         
        # setting label of x-axis
        plt.xlabel("Probability of being next ")
        plt.title("Horizontal bar graph")
        plt.show()
        
        col2.pyplot(hor_bar_chart)
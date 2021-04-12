# tokenize.py

import pandas as pd
import nltk
from nltk.corpus import stopwords

def load_data(file_path):  
    return pd.read_csv (r'%s'%(file_path)).values[:,1:]
    
def processed_data(data):
    return data[:,0], [sents.lower() for sents in data[:,1]]

def tokenize_sen(sen):
    stop_words = set(stopwords.words("english"))
    return [word for word in [''.join(e for e in string if e.isalnum()) for string in [x.replace('\\n','') for x in nltk.word_tokenize(sen)]] if word not in stop_words and word != '']

'''
title,text = processed_data(load_data(file_path = "ICO1_filtered.csv"))
print(tokenize_sen(text[0]))
print(title[0])
'''
# tokenize.py

import pandas as pd
import nltk
from nltk.corpus import stopwords

def load_data(file_path):
    df = pd.read_csv (r'%s'%(file_path))
    data = df.values
    data = data[:,1:]
    return data
    
def processed_data(data):
    paper_titles = data[:,0]
    paper_texts = data[:,1]
    paper_texts = [sents.lower() for sents in paper_texts]
    return paper_titles, paper_texts

def tokenize_sen(sen):
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in [''.join(e for e in string if e.isalnum()) for string in [x.replace('\\n','') for x in nltk.word_tokenize(sen)]] if word not in stop_words and word != '']
    return tokens

'''
_,test = processed_data(load_data(file_path = "ICO1_filtered.csv"))
print(tokenize_sen(test[0]))
'''
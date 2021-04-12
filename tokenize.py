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

def token_count(sen):
    return {token:sen.count(token) for token in set(sen)}

def tf_idf(docs):
    docs_tokens={idx:token_count(doc) for idx,doc in enumerate(docs)}
    dtm=pd.DataFrame.from_dict(docs_tokens, orient="index" )
    dtm=dtm.fillna(0)
    dtm = dtm.sort_index(axis = 0)        
    tf=dtm.values
    doc_len=tf.sum(axis=1, keepdims=True)
    tf=np.divide(tf, doc_len)
    df=np.where(tf>0,1,0)
    smoothed_idf=np.log(np.divide(len(docs)+1, np.sum(df, axis=0)+1))+1    
    smoothed_tf_idf=tf*smoothed_idf
    return smoothed_tf_idf

'''
title,text = processed_data(load_data(file_path = "ICO1_filtered.csv"))
print(tokenize_sen(text[0]))
print(title[0])
title,text = processed_data(load_data(file_path = "D:\\main\\Education\\Stevens MBA (2020-2022)\\1_Spring 2021\\BIA 660\\Project\\Datasets\\ICO1_filtered.csv"))
tok_sens = [tokenize_sen(sen) for sen in text]
tfidf = tf_idf(tok_sens)
print(tfidf)
'''
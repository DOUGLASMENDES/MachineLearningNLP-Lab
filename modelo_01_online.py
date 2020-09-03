import json

import numpy as np 
import pandas as pd 
import re
import nltk 
import pickle

# Modelo Machine Learning NLP a disponibilizar como AWS lambda function:

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nltk.download('stopwords')
from nltk.corpus import stopwords

textoAClassificar = "@VirginAmerica this is great news!  America could start flights to Hawaii by end of year http://t.co/r8p2Zy3fe4 via @Pacificbiznews"
textoAClassificar = "@VirginAmerica  I flew from NYC to SFO last week and couldn't fully sit in my seat due to two large gentleman on either side of me. HELP!"
#textoAClassificar = "@VirginAmerica do you miss me? Don't worry we'll be together very soon."



def trata_texto(texto):
    # Remove todos os caracteres especiais
    processed_feature = re.sub(r'\W', ' ', texto)

    # remove todos os caracteres simples
    processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove caracteres simples do início
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substitui múltiplos espaços com espaço simples
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Remove 'b' prefixado
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converte para minúsculo
    processed_feature = processed_feature.lower()
    return processed_feature


def classifica_texto(texto):
    airline_tweets = pd.read_csv("dados/Tweets.csv")
    features = airline_tweets.iloc[:, 10].values
    labels = airline_tweets.iloc[:, 1].values

    processed_features = []

    for sentence in range(0, len(features)):
        processed_feature = trata_texto(str(features[sentence]))    
        processed_features.append(processed_feature)

    vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
    processed_features = vectorizer.fit_transform(processed_features).toarray()

    with open('modelo_classif_texto_lambda_01.pkl', 'rb') as training_model:
        model = pickle.load(training_model)

    #y_pred2 = model.predict(X_test)

    #print(confusion_matrix(y_test, y_pred2))
    #print(classification_report(y_test, y_pred2))
    #print(accuracy_score(y_test, y_pred2)) 

    tfidfconverter = TfidfTransformer()

    text = vectorizer.transform([texto]).toarray()
    text = tfidfconverter.fit_transform(text).toarray()

    label = model.predict(text)[0]

    return label


textoAClassificar = trata_texto(textoAClassificar)
tipo_retorno = classifica_texto(textoAClassificar)

print("RETORNO: " + tipo_retorno)
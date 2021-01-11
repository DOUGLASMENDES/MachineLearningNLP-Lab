import numpy as np 
import pandas as pd 
import re
import nltk 
import pickle
import requests

# Treinamento do modelo NLP

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

nltk.download('stopwords')
from nltk.corpus import stopwords

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


print('Downloading training data...')

csvUrl = 'https://github.com/DOUGLASMENDES/MachineLearningNLP-Lab/blob/master/dados/Tweets.csv?raw=true'
r = requests.get(csvUrl, allow_redirects=True)

open('Tweets.csv', 'wb').write(r.content)

print('Training data downloaded!')

airline_tweets = pd.read_csv("Tweets.csv")

features = airline_tweets.iloc[:, 10].values
print(features[:10])

labels = airline_tweets.iloc[:, 1].values
print(labels[:10])


processed_features = []

for sentence in range(0, len(features)):
    processed_feature = trata_texto(str(features[sentence]))
    
    processed_features.append(processed_feature)


vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)

predictions = text_classifier.predict(X_test)


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))


with open('modelo_classif_texto_lambda_01.pkl', 'wb') as picklefile:
    pickle.dump(text_classifier, picklefile)


with open('modelo_classif_texto_lambda_01.pkl', 'rb') as training_model:
    model = pickle.load(training_model)

y_pred2 = model.predict(X_test)

print(confusion_matrix(y_test, y_pred2))
print(classification_report(y_test, y_pred2))
print(accuracy_score(y_test, y_pred2)) 

tfidfconverter = TfidfTransformer()

# Testando:

# exemplo de sentimento positivo:
#text = "@VirginAmerica this is great news!  America could start flights to Hawaii by end of year http://t.co/r8p2Zy3fe4 via @Pacificbiznews"

# exemplo de sentimento negativo:
text = "@VirginAmerica  I flew from NYC to SFO last week and couldn't fully sit in my seat due to two large gentleman on either side of me. HELP!"

text = trata_texto(text)

text = vectorizer.transform([text]).toarray()
text = tfidfconverter.fit_transform(text).toarray()

label = model.predict(text)[0]

print(label)
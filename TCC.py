########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
import operator

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from string import punctuation
from collections import defaultdict
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from scipy.sparse import hstack
from scipy.special import logit, expit


import sys

df=pd.read_csv("Data/cleanwords.csv")
## cleaning text in datasets

print('cleaning text dataset')
# Regex to remove all Non-Alpha Numeric and space
splchar_removal=re.compile(r'[^?!.,:a-z\d ]',re.IGNORECASE)

# regex to replace all numerics
replace_no=re.compile(r'\d+',re.IGNORECASE)

def clean_text(text, remove_stopwords=False, stem_words=True, clean_wiki_tokens=True):
    # Clean the text, with the option to remove stopwords and to stem words.
    # dirty words
    text = text.lower()
    text = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", "", text)
    text = re.sub(r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}", "", text)

    if clean_wiki_tokens:
        # Drop the image
        text = re.sub(r"image:[a-zA-Z0-9]*\.jpg", " ", text)
        text = re.sub(r"image:[a-zA-Z0-9]*\.png", " ", text)
        text = re.sub(r"image:[a-zA-Z0-9]*\.gif", " ", text)
        text = re.sub(r"image:[a-zA-Z0-9]*\.bmp", " ", text)

        # Drop css
        text = re.sub(r"#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})", " ",text)
        text = re.sub(r"\{\|[^\}]*\|\}", " ", text)

        # Clean templates
        text = re.sub(r"\[?\[user:.*\]", " ", text)
        text = re.sub(r"\[?\[user:.*\|", " ", text)
        text = re.sub(r"\[?\[wikipedia:.*\]", " ", text)
        text = re.sub(r"\[?\[wikipedia:.*\|", " ", text)
        text = re.sub(r"\[?\[special:.*\]", " ", text)
        text = re.sub(r"\[?\[special:.*\|", " ", text)
        text = re.sub(r"\[?\[category:.*\]", " ", text)
        text = re.sub(r"\[?\[category:.*\|", " ", text)

    # data frame
    for i in range(len(df['type'])):
        text = re.sub(df['type'][i]+ " " ,df['correct'][i] + " ", text)

    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\!", " ! ", text)
    text = re.sub(r"\"", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = replace_no.sub(' ', text)
    text = splchar_removal.sub('',text)
    #splitting line to words and joining it back with single space to join words back and remove extra spaces.

    text = " ".join(text.split())



    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    return (text)


train_df = pd.read_csv('Data/train.csv')
test_df = pd.read_csv('Data/test.csv')
list_sentences_train = train_df["comment_text"].fillna("no comment").values
list_sentences_test = test_df["comment_text"].fillna("no comment").values

comments = [clean_text(text) for text in list_sentences_train]
test_comments=[clean_text(text) for text in list_sentences_test]

print("Cleaned.")

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
all_text = pd.concat([pd.Series(list_sentences_train),  pd.Series(list_sentences_test)])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=20000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(comments)
test_word_features = word_vectorizer.transform(test_comments)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 6),
    max_features=30000)
char_vectorizer.fit(all_text)

train_char_features = char_vectorizer.transform(comments)
test_char_features = char_vectorizer.transform(test_comments)

from sklearn.model_selection import cross_val_score, cross_val_predict

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])
# classifier = nltk.NaiveBayesClassifier.train(train_features)

# print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, test_features))*100)

losses = []
predictions = {'id': test_df['id']}
for class_name in class_names:
    train_target = train_df[class_name]
    classifier = LogisticRegression(solver='sag')
    classifier.fit(train_features, train_target)
    predictions[class_name] = classifier.predict_proba(test_features)[:, 1]

submission = pd.DataFrame.from_dict(predictions)
submission.to_csv('Data/Logistic-Submission2.csv', index=False)

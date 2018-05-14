# Setting up by importing all the relevant libraries
import os
import pandas as pd
import nltk
import re

# Setting parameters
dir = r'C:\Users\ankuarora\Desktop\Client\2017_05_CogEx\2017_07_R2Implementation\Sprint 7\1CreateTrainData'
features_file = 'features_annotated.csv'

# Read the features csv created in the previous step, including manual annotation
features = pd.read_csv(os.path.join(dir, features_file), quotechar='"', index_col = 'filename', encoding='utf-8')
features['body'] = [1 if type == 'Body' else 0 for type in features['type']]

# Clean the ocr text to prepare for creating text features
stop = set(nltk.corpus.stopwords.words('english'))
wnl = nltk.WordNetLemmatizer()
word_corpus = []
for _, row in features.iterrows():
    if pd.isnull(row['text']):
        continue
    tokens = nltk.word_tokenize(row['text']) # tokenize
    tokens = [i for i in tokens 
              if i not in stop 
              if i.isalnum()
              if len(i) > 2
             ] # stopword removal & text cleaning
    tokens_lemma = [wnl.lemmatize(t) for t in tokens] # lemmatization
    for word in tokens_lemma:
        word_corpus.append(word)

# Create a list of most common words
all_words = nltk.FreqDist(w for w in word_corpus)
word_features = [w[0] for w in all_words.most_common(50)]

# Create word count features for the most common words 
for key, row in features.iterrows():
    for word in word_features:
        if pd.isnull(row['text']):
            continue
        features.loc[key,('count({})'.format(word))] = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(word), row['text']))

# Dump the feature space as csv for model training
features.to_csv('feature_space.csv', encoding='utf-8')

print ("Feature space exported successfully")
# -*- coding: utf-8 -*-
"""NER.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ttDOuDHkrgiC2cHHtkgOIKhO8ba2tfGq

**Please make a copy of this notebook :))))**

How to Load the dataset?
"""

!pip install datasets
from datasets import load_dataset
dataset = load_dataset('conll2003')

"""[I] indicates the features taken into consideration in the SVM model
1. All caps?   [I]
2. First letter cap? [I]
3. mean position in the sentence?
4. Length of the word? [I]
5. Mean number of times preceded by another Named Entity
6. Noun? [If POS tag available]
7. Hyphen? [I]
8. Is a number? Is a date? [I]
9. Preceded by article?
10. Preceded or Succeeded by Preposition
11. Contains dot [I]
12. Is a stop word? [I]
13. Next 's?
14. Preceeded by Named Entity?
15. Suceeded by Named Entity?
16. Named Entity Encountered in the sentence far back?
17. Whole sentence in upper case?




"""

stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def get_vector(word,word_vectors):
  try:
    return word_vectors[word]
  except:
    vector=[]
    vector.append(1 if word.upper()==word else 0) # 1) All characters CAPS?
    vector.append(1 if word[0].upper()==word[0] else 0) # 2) First character in CAPS?
    vector.append(len(word)) # 4) Length of the word
    vector.append(1 if '-' in word else 0) # 7) Contains '-'
    vector.append(1 if word.isnumeric() else 0) # 8) Is a number? 
    vector.append(1 if '.' in word else 0) # 11) Contains '.' ?
    vector.append(1 if word in stopwords else 0) # 12) Is a stop word?

    word_vectors[word]=vector
    return word_vectors[word]

word_vectors={}
train_X=[]
train_y=[]

for sentence,ner_sentence in zip(dataset['train'].data['tokens'],dataset['train'].data['ner_tags']):
                                                                   for word,ner_word in zip(sentence,ner_sentence):
                                                                     train_X.append(get_vector(word.as_py(),word_vectors))
                                                                     train_y.append(0 if ner_word.as_py()==0 else 1)

from sklearn import svm
clf = svm.SVC()
clf.fit(train_X, train_y) # This will take 40 min to train :) 
                          # I have saved the model below using pickle just let me know if you want it
                          # This is how you load it: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

"""Saving Model"""

import pickle

filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))

"""Accuracy on training set"""

test_X=[]
test_y=[]

"""Some things which will make your life easier
1. As always the words are not easily available as strings it is in this form:
2. If you execute this--> dataset['train'].data['tokens'][1][1]
3. i.e. 2nd sentence 2nd word you get: 
<pyarrow.StringScalar: 'Blackburn'>
Here the data structure is pyarrow and data is of type StringScalar
The actual data is "BlackBurn"
4. To get the string out of it use  "dataset['train'].data['tokens'][1][1].as_py()"
"""

for test_sentence,test_ner_sentence in zip(dataset['train'].data['tokens'],dataset['train'].data['ner_tags']):
                                                                   for word,ner_word in zip(test_sentence,test_ner_sentence):
                                                                     test_X.append(get_vector(word.as_py(),word_vectors))
                                                                     test_y.append(0 if ner_word.as_py()==0 else 1)

svm_predictions=clf.predict(test_X)

import numpy as np
actual_predictions=np.array(test_y)

compare=(actual_predictions==svm_predictions)
sum(compare)/len(compare)

"""Accuracy on Test set"""

test_X=[]
test_y=[]

for test_sentence,test_ner_sentence in zip(dataset['test'].data['tokens'],dataset['test'].data['ner_tags']):
                                                                   for word,ner_word in zip(test_sentence,test_ner_sentence):
                                                                     test_X.append(get_vector(word.as_py(),word_vectors))
                                                                     test_y.append(0 if ner_word.as_py()==0 else 1)

svm_predictions=clf.predict(test_X)

import numpy as np
actual_predictions=np.array(test_y)

compare=(actual_predictions==svm_predictions)
sum(compare)/len(compare)

"""How many named enetities were detected correctly?"""

sum(actual_predictions*svm_predictions)/sum(actual_predictions)
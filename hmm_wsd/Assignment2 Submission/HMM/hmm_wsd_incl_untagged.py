# In the code comments synset ID is used interchangibly with tag
# Importing necessary libraries

import numpy as np
import itertools
import nltk
import random
import math
import statistics
import pandas as pd
from nltk.tree import Tree
from nltk.corpus import semcor


# Getting the brown corpus
print("Downloading files from NLTK please wait...")
nltk.download('all', quiet=True)
print("NLTK files downloaded!")



# Task: Tag words with synset ID
# Three types of tags in semcor
# 1) No tag
# 2) Tagged with a lemma of synset: A lemma is a data structure which contains synset along with its , hypernymy, hyponymy,... 
# 3) Tagged with lemma but the synset ID doesn't exist in wordnet
#    One such example is the 7th sentence in the semcor corpus
#    The grand jury commented on a number of other topics, among them the Atlanta and Fulton County purchasing departments which it said are 
#    well operated and follow generally accepted practices which inure to the best interest of both governments.
#                                       ^______^  "accepted" is tagged as "accepted.s.00" lemma and this doesn't have a synsetID in wordnet
#    All the three cases are handled by the code

# Each sentence has a pair of phrase and (representation of)synset
# Each Phrase has one or more words(e.g. "primary election" in first sentence of corpora)
# Each synset is respresented as a Tree e.g. for "primary election"
# Lemma('primary.n.01.primary_election')
#                 /\
#                /  \
#               /    \
#         primary   election
# This example will be used later to explain code
print("Preprocessing the semcor corpus...")
synCorpus=[]
current_index=0
for taggedSents in semcor.tagged_sents(tag='sem'): # for each sentence fetch the (phrase,synsetTree)
    synCorpus.append([])                           # create a list designating current sentence which will contain (phrase,synsetID)
    for phrase in taggedSents:
        if type(phrase)==list:                     # The phrase with no tag is a list in semcor [Note: CASE 1]
            #if its a list fetch the word and set -1 as synset ID which indicates no synset TAG
            synCorpus[current_index].append((phrase[0],-1))  # a tuple is formed
        else: # if synset is a Tree [Note Case 2 and 3]
            # Here when theres a synset not in wordnet, theres no synset id[Case 3], to handle this
            # except block is used
            # The try except will get the synset ID and in case no synsetID -2 is given to such a case
            try: 
                # handle Case 2
                synTag_for_corpus=phrase.label().synset().offset()
            except:
                # handle Case 3
                synTag_for_corpus=-2 # -2 indicates no synset tags for the specific POS but available for some POS
            if isinstance(phrase.label(),str) ==True:
                synText_for_corpus=phrase.label().split(".")[0]
            else:
                synText_for_corpus=phrase.label().name()
            synCorpus[current_index].append((synText_for_corpus,synTag_for_corpus)) # For case 2 and 3 form a tuple
    current_index+=1 # Keeps tract of the sentence index
no_sents=current_index #Saved total number of sentences
# The reason for storing as tuples is because the brown corpus stored it in a same way and this way we have to make minimal changed to HMM-Viterbi POS tag code if any :)
print("Semcor preprocessing finished")

# Sentences fetched and prefixed and suffixed by delimiters. If originally sentence contains a 
# delimiter then set it as ('<DEL>','X')
sentence_tag = synCorpus
modified_sentence_tag=[]
for sent in sentence_tag:
  for word,tag in sent:
    if word=='^^' or word=='$$':
      word='<DEL>'
      tag='X'
  sent.insert(0,('^^','^^'))         # Sentence starts with '^^'
  sent.append(('$$','$$'))           # Sentence ends with '$$'
  modified_sentence_tag.append(sent)

# Shuffle the whole corpus uniformly
random.shuffle(modified_sentence_tag)

# Divide corpus into 5 equal parts
sentences_set1=modified_sentence_tag[:math.floor(len(modified_sentence_tag)*1/5)]
sentences_set2=modified_sentence_tag[math.floor(len(modified_sentence_tag)*1/5):math.floor(len(modified_sentence_tag)*2/5)]
sentences_set3=modified_sentence_tag[math.floor(len(modified_sentence_tag)*2/5):math.floor(len(modified_sentence_tag)*3/5)]
sentences_set4=modified_sentence_tag[math.floor(len(modified_sentence_tag)*3/5):math.floor(len(modified_sentence_tag)*4/5)]
sentences_set5=modified_sentence_tag[math.floor(len(modified_sentence_tag)*4/5):]

train_sentences=[[],[],[],[],[]]
test_sentences=[[],[],[],[],[]]

# For 5 Fold Cross Validation Train and test set
# Set1 as test set
train_sentences[0]=sentences_set2+sentences_set3+sentences_set4+sentences_set5
test_sentences[0]=sentences_set1

# Set2 as test set
train_sentences[1]=sentences_set1+sentences_set3+sentences_set4+sentences_set5
test_sentences[1]=sentences_set2

# Set3 as test set
train_sentences[2]=sentences_set1+sentences_set2+sentences_set4+sentences_set5
test_sentences[2]=sentences_set3

# Set4 as test set
train_sentences[3]=sentences_set1+sentences_set2+sentences_set3+sentences_set5
test_sentences[3]=sentences_set4

# Set5 as test set
train_sentences[4]=sentences_set1+sentences_set2+sentences_set3+sentences_set4
test_sentences[4]=sentences_set5

precision_sets=[0]*5
recall_sets=[0]*5
F1_score_sets=[0]*5
F05_score_sets=[0]*5
F2_score_sets=[0]*5
pos_estimation_sets=[pd.DataFrame]*5
print("Running Viterbi over all 5 cross validation sets...") #Note this is just for showing progress, viterbi is executed further in the loop
for setno in range(5):
  # For each set calculate the transmission and emission probabilities on training set
  # And perform viterbi on test set. Later find per POS and overall estimation
  train_dataset = train_sentences[setno]
  test_dataset = test_sentences[setno]

  ## EMISSION PROBABILITY TABLE
  # Creation of a dictionary whose keys are tags and values contain words which have corresponding tag in the taining dataset
  # example:- 'SYNSET_ID':{word1: count(word1,'SYNSET_ID')} count(word1,'SYNSET_ID') means how many times the word is tagged as 'SYNSET_ID'
  train_word_tag = {}
  for sent in train_dataset:
    for (word,tag) in sent:
      word=word.lower()            # removing ambiguity from capital letters 
      try:
        try:
          train_word_tag[tag][word]+=1
        except:
          train_word_tag[tag][word]=1
      except:
          train_word_tag[tag]={word:1}

  #Calculation of emission probabilities using train_word_tag
  train_emission_prob={}
  for key in train_word_tag.keys():
    train_emission_prob[key]={}
    count = sum(train_word_tag[key].values())                           # count is total number of words tagged as a 'SYNSET_ID'
    for key2 in train_word_tag[key].keys():
      train_emission_prob[key][key2]=train_word_tag[key][key2]/count    
  #Emission probability is #times a word occured as 'SYNSET_ID' / total number of 'SYNSET_ID' words

  ## TRANSITION PROBABILITY TABLE
  #Estimating the bigrams of synset id tags to be used for calculation of transition probability 
  #Bigram Assumption is made, the current tag depends only on the previous tag
  bigram_tag_data = {}
  for sent in train_dataset:
    bi=list(nltk.bigrams(sent))
    for b1,b2 in bi:
      try:
        try:
          bigram_tag_data[b1[1]][b2[1]]+=1
        except:
          bigram_tag_data[b1[1]][b2[1]]=1
      except:
        bigram_tag_data[b1[1]]={b2[1]:1}

  #bigram_tag_data is storing the values for every synset id.
  #Every key is a synset id and value is tag followed for that key and corresponding counts.

  #Calculation of the probabilities of tag bigrams for transition probability
  #We already made a bigram assumption  
  #Also note that since we are also considering $, the $ row of transition probability matrix give us the initial probabilities as well
  bigram_tag_prob={}
  for key in bigram_tag_data.keys():
    bigram_tag_prob[key]={}
    count=sum(bigram_tag_data[key].values())              # count is total number of times a 'SYNSET_ID' has occured
    for key2 in bigram_tag_data[key].keys():
      bigram_tag_prob[key][key2]=bigram_tag_data[key][key2]/count
  #Transmission probability is #times a SYNSET_ID2 is preceded by SYNSET_ID1 / total number of times SYNSET_ID1 exists in dataset
  
  #Calculation the possible tags for each word in the train dataset
  tags_of_tokens = {}
  count=0
  for sent in train_dataset:
    for (word,tag) in sent:
      word=word.lower()
      try:
        if tag not in tags_of_tokens[word]:
          tags_of_tokens[word].append(tag)
      except:
        list_of_tags = []
        list_of_tags.append(tag)
        tags_of_tokens[word] = list_of_tags
  #Each word and its corresponding tags in the train dataset

  # Getting words and their corresponding tags from the test set
  # Seperating the test data into test words and test ids
  test_words=[]
  test_tags=[]
  for sent in test_dataset:
    temp_word=[]
    temp_tag=[]
    for (word,tag) in sent:
      temp_word.append(word.lower()) # words of a sentence in test dataset
      temp_tag.append(tag) # tags of a sentence in test dataset
    test_words.append(temp_word) # list with words of a sentence(tokenized sentence) appended to a list of list
    test_tags.append(temp_tag) # list with tags of a sentence(tokenized sentence) appended to a list of list

  #VITERBI ALGORITHM IMPLEMENTATION
# In the following tag refers to as synset id tag
# For each word in a sentence
# – The probability of best candidates for each tag at previous level is
#   multiplied with the emission probability and the transition
#   probability of possible tags based on current word
# – The best candidate for each tag at the current level is chosen and
#   the previous tag is kept a track of for backtracking
# – The result of forward propagation at each level looks like the
#   following
#   LEVEL_K:{SYNSET_ID1:{best_candidate_among_Tag1,probability_of the
#   best candidate} , {SYNSET_ID2:{previous_tag_of_best_candidate_
#   among_Tag2, probability_of the best candidate}, ...}
# – When backtracking start from the end to start choosing the
#   predicted tag based on LEVEL information and the predicted tag
#   that follows the word.
  predicted_tags = []                #Final list for prediction
  for i in range(len(test_words)):   # for each tokenized sentence in the test data (test_words is a list of lists)
    sent = test_words[i]
    #storing_values is a dictionary which stores the required values
    #ex: storing_values = {step_no.:{state1:[previous_best_state,value_of_the_state]}}                
    storing_values = {}              
    for q in range(len(sent)):
      step = sent[q]
      #for the starting word of the sentence
      if q == 1:                
        storing_values[q] = {}
        try:
          tags = tags_of_tokens[step]
        except:
          # print(step,test_tags_of_tokens[step])
          tags=[-1] #tags_of_unseen_tokens
        for t in tags:
          #this is applied since we do not know whether the word in the test data is present in train data or not
          try:
            storing_values[q][t] = ['^^',bigram_tag_prob['^^'][t]*train_emission_prob[t][step]]
          #if word is not present in the train data but present in test data we assign a very low probability of 0.0001
          except:
            storing_values[q][t] = ['^^',0.0001]
      
      #if the word is not at the start of the sentence
      if q>1:
        storing_values[q] = {}
        previous_states = list(storing_values[q-1].keys())   # loading the previous states
        try:
          current_states  = tags_of_tokens[step]               # loading the current states
        except:
          current_states = [-1]#tags_of_unseen_tokens
        #calculation of the best previous state for each current state and then storing
        #it in storing_values
        for t in current_states:                             
          temp = []
          for pt in previous_states:                         
            try:
              temp.append(storing_values[q-1][pt][1]*bigram_tag_prob[pt][t]*train_emission_prob[t][step]) # If seen word
            except:
              temp.append(storing_values[q-1][pt][1]*0.0001)
          max_temp_index = temp.index(max(temp))
          best_pt = previous_states[max_temp_index]
          storing_values[q][t]=[best_pt,max(temp)] #Store the best previous tag for each best candidate per tag and the maximum probability
 
    #Backtracing to extract the best possible synset id tags for the sentence
    # for each word looking the current word and the word and synset id next to it in the sentence backtrack
    # to get the id of current word
    pred_tags = [] #predicted ids by viterbi using backtracking
    total_steps_num = storing_values.keys()
    last_step_num = max(total_steps_num)     # Begin from the last word which will end the delimiter
    for bs in range(len(total_steps_num)):    
      step_num = last_step_num - bs          
      if step_num == last_step_num:
        pred_tags.append('$$')
        pred_tags.append(storing_values[step_num]['$$'][0]) 
      if step_num<last_step_num and step_num>0:
        pred_tags.append(storing_values[step_num][pred_tags[len(pred_tags)-1]][0]) #Looking into storing value fetch the best previous tag for the current word
    predicted_tags.append(list(reversed(pred_tags)))


#Now that the tags are predicted, get the actual and predicted ids so that analysis can be done
  tag_seq_act=[]
  tag_seq_pred=[]
  uniq_tag=set()
  uniq_tag_dict={}
  for li in test_tags:
      for tag in li:
          if(tag!="^^" and tag!="$$"): #Exclude delimiters
            tag_seq_act.append(tag)

  for li in predicted_tags:
      for tag in li:
          if(tag!="^^" and tag!="$$"): #Exclude delimiters
            tag_seq_pred.append(tag)
      
  for tag in tag_seq_act:
      uniq_tag.add(tag)
      
  for tag in tag_seq_pred:
      if tag not in uniq_tag:
      	uniq_tag.add(tag)
  uniq_tag=list(uniq_tag)
    
  for i in range(len(uniq_tag)):
      uniq_tag_dict[uniq_tag[i]]=i
              
  for i,tag in enumerate(tag_seq_act):
      tag_seq_act[i]=uniq_tag_dict[tag]

  for i,tag in enumerate(tag_seq_pred):
      tag_seq_pred[i]=uniq_tag_dict[tag]

# Calculate the precision, Recall and F-Scores by comparing the actual and predicted tags
  matched_tags=0
  for i in range(len(tag_seq_act)):
      if tag_seq_act[i]==tag_seq_pred[i]:
        matched_tags+=1
  # Estimations for the current set
  precision=matched_tags/(len(tag_seq_pred))
  recall=matched_tags/(len(tag_seq_act))
  F1_score=(2*precision*recall)/(precision+recall)
  F05_score=(1.25*precision*recall)/(0.25*precision+recall)
  F2_score=(5*precision*recall)/(4*precision+recall)

  #Store every set's estimations
  precision_sets[setno]=precision
  recall_sets[setno]=recall
  F1_score_sets[setno]=F1_score
  F05_score_sets[setno]=F05_score
  F2_score_sets[setno]=F2_score
  #pos_estimation_sets[setno]=pos_estimation
  print("Set ",setno+1,"✓")

# After getting every set's estimations combine them
# For k fold cross validation
# mean(each of the k set estimations) ± standard_error(each of the k set estimations)
# standard_error= standard_deviation/square_root(k)
# example Overall precision=(Σ precision_set i)/5 ± squareroot([Σ(precision_set_i-precision_mean)²]/5) here 5 for 5 fold cross validation
print("===================\nOVERALL ESTIMATIONS\n===================")
print("Overall Precision:", "{:.6f}".format(statistics.mean(precision_sets)),"±","{:.6f}".format(statistics.stdev(precision_sets)/math.sqrt(setno+1)))
print("Overall Recall:" ,"{:.6f}".format(statistics.mean(recall_sets)),"±","{:.6f}".format(statistics.stdev(recall_sets)/math.sqrt(setno+1)))
print("Overall F1 Score:", "{:.6f}".format(statistics.mean(F1_score_sets)),"±","{:.6f}".format(statistics.stdev(F1_score_sets)/math.sqrt(setno+1)))
print("Overall F0.5 Score:", "{:.6f}".format(statistics.mean(F1_score_sets)),"±","{:.6f}".format(statistics.stdev(F1_score_sets)/math.sqrt(setno+1)))
print("Overall F2 Score:", "{:.6f}".format(statistics.mean(F1_score_sets)),"±","{:.6f}".format(statistics.stdev(F1_score_sets)/math.sqrt(setno+1)))


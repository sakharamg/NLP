
import numpy as np
import nltk
from nltk.tree import Tree
from nltk.corpus import semcor
from nltk.corpus import wordnet as wn


# Getting the brown corpus
print("Downloading files from NLTK please wait...")
nltk.download('all', quiet=True)
print("NLTK files downloaded!")



# Task: Tag words with synset ID
# Three types of tags in semcor
# 1) No tag
# 2) Tagged with a lemma of synset: A lemma is a data structure which contains synset along with its synset_id, hypernymy, hyponymy,... 
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
            # synCorpus[current_index].append((phrase[0],-1))  # a tuple is formed
            pass
        else: # if synset is a Tree [Note Case 2 and 3]
            # Here when theres a synset not in wordnet, theres no synset id[Case 3], to handle this
            # except block is used
            # The try except will get the synset ID and in case no synsetID -2 is given to such a case
            try: 
                # handle Case 2
                synCorpus[current_index].append((phrase.label().name(),phrase.label().synset().name())) 
            except:
                # handle Case 3
                #synTag_for_corpus=-2 # -2 indicates no synset tags for the specific POS but available for some POS
                pass
    current_index+=1 # Keeps tract of the sentence index
no_sents=current_index #Saved total number of sentences
# The reason for storing as tuples is because the brown corpus stored it in a same way and this way we have to make minimal changed to HMM-Viterbi POS tag code if any :)
print("Semcor preprocessing finished")

print("Calculating WFS Baseline")
WFS=[]
current=0
for sent in synCorpus:
    WFS.append([])
    for (word,tag) in sent:
          WFS[current].append((word,wn.synsets(word)[0].name()))
    current+=1
total_count=0
correct_count=0
current_sent=0
for sent in synCorpus:
    current_word=0
    for (word,tag) in sent:
        if tag==WFS[current_sent][current_word][1]:
            correct_count+=1
        total_count+=1
        current_word+=1
    current_sent+=1    
print("WFS Accuracy: ",correct_count*100/total_count,'%')
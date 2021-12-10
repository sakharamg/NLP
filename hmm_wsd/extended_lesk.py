import numpy as np
import nltk
from nltk.tree import Tree
from nltk.corpus import semcor
from gensim.models import KeyedVectors
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
print("Preprocessing the semcor corpus...")
synCorpus=[]
current_index=0
for taggedSents in semcor.tagged_sents(tag='sem'): # for each sentence fetch the (phrase,synsetTree)
    synCorpus.append([])                           # create a list designating current sentence which will contain (phrase,synsetID)
    for phrase in taggedSents:
        if type(phrase)==list:                     # The phrase with no tag is a list in semcor [Note: CASE 1]
            pass
        else: 
            try: 
                # handle Case 2
                synCorpus[current_index].append((phrase.label().name(),phrase.label().synset().name())) 
            except:
                # handle Case 3, The tags with no synset id but a synset present
                pass
    current_index+=1 # Keeps track of the sentence index
no_sents=current_index #Saved total number of sentences
print("Semcor preprocessing finished")
#Load the word2vec pretrained model
print("Loading word2vec")
try:
	model_w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',binary=True)
except:
	print("Download pretrained word2vec from https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz and save in the same directory of the file\nExiting...")
	exit()
np.seterr(divide='ignore', invalid='ignore')
print("Testing and getting accuracy estimate")
synCorpus_words=[]
line=0

# Get all the sentences in corpus
for sent in synCorpus:
    synCorpus_words.append([])
    for (word,tag) in sent:
        synCorpus_words[line].append(word)
    line+=1

# Find context bag for each sentence, the ambiguous word is ignored when comparing with the ambigous word's sense
length_of_wordvec=len(model_w2v["the"])
context_bag_vector = []
for sent in synCorpus_words:
    context_count=0
    context_list=np.zeros((length_of_wordvec,))
    #for every sentence get the context bag
    single_sentence='_'.join(word for word in sent)
    single_sentence=single_sentence.replace('-','_')
    splits=single_sentence.split('_')
    for word in splits:
        try:
            context_list+=model_w2v[word]
            context_count+=1
        except:
            pass
    if context_count!=0:
        context_bag_vector.append(context_list/context_count)
    else:
        context_bag_vector.append(context_list)


# Find similarity with sense bag for each sense
overlapCorpus=[]
count=0
test=0
test1=0
for sent in synCorpus_words:
    overlapCorpus.append([])
    for word in sent:
        synsets_word=wn.synsets(word)
        curr_sim=-1
        curr_synset=synsets_word[0]
        for synset_word in synsets_word: # for each sense of a word
            amb_list=np.zeros((length_of_wordvec,))
            amb_count=0
            synset_gloss=synset_word.definition() # get the word's gloss
            splits=synset_gloss.split()
            for word1 in splits:
                try:
                    if(word1!=word):
                        amb_list+=model_w2v[word1]
                        amb_count+=1
                except:
                    pass
            if amb_count!=0: # for context bag wrt an ambiguous word
                amb_vec=(amb_list/amb_count)
            else:
                amb_vec=amb_list
            cos_similarity=model_w2v.cosine_similarities(context_bag_vector[count].T,[amb_vec.T])[0] # Check cosine similary
            if curr_sim<cos_similarity: # and update the sense to the most similar sense
                curr_sim=cos_similarity
                curr_synset=synset_word
        overlapCorpus[count].append((word,curr_synset.name()))
    count+=1

# Test the accuracy of overlap based
total_count=0
correct_count=0
current_sent=0
for sent in synCorpus:
    current_word=0
    for (word,tag) in sent:
        if tag==overlapCorpus[current_sent][current_word][1]:
            correct_count+=1
        total_count+=1
        current_word+=1
    current_sent+=1    
print("Accuracy of WSD with overlapping: ",correct_count*100/total_count,'%')

from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize




stop_words = stopwords.words('english')

# word_vectors = KeyedVectors.load_word2vec_format('/home/sandy/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True)


corpus = [[(),(),()],[(),(),()]]

#predictions = [[(),(),()],[(),(),()]] # going to predict

predictions = []

for sent in corpus:
	test_sentence = []
	sentence_predictions = []
	
	for wd in sent:
		test_sentence.append(wd[0])
	
	for wd in sent:
		tag_pred_sim = dict()
		# test_word = 'play'
		# test_tag = 'play.v.01'
		test_word = wd[0]
		test_tag = wd[1]

		test_sentence_set = {word for word in test_sentence if not word in stopwords}
		test_sentence_set = test_sentence_set - set([wd])

		test_pred_similarity = []
		for item in test_sentence_set:
			v_wd = word_vectors[wd]
			v_item = word_vectors[item]
			cos_sim = cosine_similarity([v_wd],[v_item])
			test_pred_similarity.append(cos_sim)
		test_max_similarity = max(test_pred_similarity)

		tag_pred_sim.update({test_tag:test_max_similarity})

		test_sentence_set = {word for word in test_sentence if not word in stopwords}
		test_sentence_set = test_sentence_set - set([wd])

		word_synsets = wordnet.synsets(test_word)

		tag_list = []
		definition_list = []

		for i in range(len(word_synsets)):
			tag_list.append(wordnet.synsets(test_word)[i].name())

		if len(tag_list) == 1:
			sentence_predictions.append((test_word,test_tag))
		else:
			for i in range(len(tag_list)):
				definition_list.append(wordnet.synset(tag_list[i]).definition())
				current_definition = word_tokenize(definition)
				current_definition_set = {word for word in current_definition if not word in stopwords}
				temp_similarity = []
				for item in current_definition_set:
					v_wd = word_vectors[wd]
					v_item = word_vectors[item]
					cos_sim = cosine_similarity([v_wd],[v_item])
					temp_similarity.append(cos_sim)
				temp_max_similarity = max(temp_similarity) 
				if test_tag == tag_list[i]:
					if tag_pred_sim[test_tag] < temp_max_similarity:
						tag_pred_sim.update({test_tag:temp_max_similarity})
				else:
					tag_pred_sim.update({tag_list[i]:temp_max_similarity})
		keymax = max(zip(tag_pred_sim.values(), tag_pred_sim.keys()))[1]
		sentence_predictions.append((test_word,keymax))

	predictions.append(sentence_predictions)


#you have corpus and you have predictions, we can calculate the precison, accuracy etc 
		



# v_apple = word_vectors['banana']
# v_mango = word_vectors['mango']
# print(cosine_similarity([v_apple],[v_mango]))
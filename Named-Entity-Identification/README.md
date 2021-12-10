# Named-Entity-Identification

'' Get dataset:

On commandline:
pip install datasets


On python:
from datasets import load_dataset
dataset = load_dataset('conll2003')

'' Access Dataset
print(dataset)
	shows the metadata of dataset
	
dataset['train'].data['tokens']
	Get the tokens
	
dataset['train'].data['ner_tags']
	Get ner tags
	

https://paperswithcode.com/dataset/conll-2003
https://huggingface.co/datasets/conll2003

dataset['train'].data['tokens'][i]
	ith sentence

dataset['train'].data['ner_tags'][i]
	named entity tag for words in ith sentence

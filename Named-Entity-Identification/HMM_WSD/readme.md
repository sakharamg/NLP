# Assignment 2: Part1 - HMM based WSD 

1) Our preprocessing had issues when processing the test

Words were spaced in between example: "hello" was converted to "h e l l o"

2) Also for unseen data when applying viterbi, we have used WSD and MFS as suggested to us in the assignment 2 discussion

## Tools:  

	1) Python (version 3.x)  
	2) Numpy (version 1.21.2)  
	3) Matplotlib (version 3.4.3)  
	4) NLTK (version 3.6.2)  
	5) Pandas (version 1.3.2) 
	6) Sklearn  

## How to run code

### For Linux:

1) Run the file *main.py* using linux terminal as follows:  
`python3 main.py`
2) Select the choice from options available
3) The Estimations will be shown within the terminal 

### For Windows

1) Run the file *main.py* using command prompt as follows:  
`python main.py`
2) Select the choice from options available
3) The Estimations will be shown within the command prompt

## References
 
1) *Pandas Dataframe used for storing per POS estimations*: **Pandas Dataframe by Pandas** https://pandas.pydata.org/pandas-docs/version/0.23.3/generated/pandas.DataFrame.html  
2) *Shuffling the corpus using random()*: **random.shffle by Python** https://docs.python.org/3/library/random.html#random.shuffle  
3) *Nested Dictionaries in python*: **Python Nested Dictionary by programiz** https://www.programiz.com/python-programming/nested-dictionary  
4) *Calculating estimations on k-fold cross validation:* **Black Box Machine Learning by Bloomberg ML EDU**  https://youtu.be/MsD28INtSv8?t=3139

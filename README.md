## Unigram, Bigram and Trigram Language Models and Smoothing
### Run me in Google Colab Notebook 

https://colab.research.google.com/drive/1AiOHCNk4knaGlbJ1Pj1Y6UDf0itCB7Zd 

### About the dataset 
There are three datasets provided (subsets of the One Billion Word Language Modeling Benchmark). 
Each line in a file contains a whitespace-tokenized sentence.

`1b_benchmark.train.tokens` -> data for training the language models

`1b_benchmark.dev.tokens` -> data for debugging and choosing the best hyperparameters (in smoothing)

`1b_benchmark.test.tokens` -> data for evaluating your language models

Note: The term `UNK` indicates words that appear less than three times in the dataset.
 `<start>` denotes the start token and `<stop>` denotes the stop token. 
 Those not included in the files and will be appended as required while processing the dataset.
 While computing probability of a sentence, any OOV (out of vocabulary) word will be converted to UNK.
 
 ### Implementing the models
The following steps were taken to create and evaluate the language models
 
1) Using train tokens. Create a function taking in a file of data and create uni, bi, trigrams from it. 
 Convert rarely appearing tokens (< 3 times) to UNKS. Calculate probability of occurence of each of the ngrams in the vocabulary. 

2) Using train tokens. Take the dataset on which probability of sentences will be calculated and convert OOVs to UNKS.
Parse the file sentence by sentence converting words in sentences to unigrams/bigrams/trigrams. Find probability of occurrence of each of the aforementioned ngrams in sentences based on findings from the probability maps (first part of the procedure). 

3) Using train tokens. Calculate perplexity based on findings in (2). 

4) When the models are trained implement smoothing, 
adjusting the parameters (to ensure lowest perplexity) using dev tokens. 

5) Run the perplexity on test tokens. 

### Structure 
The following folder contains two files `modelEvaluation.py` and `languageModels.py`
as well as the datasets, this README file and a `requirements.txt `. 
If the program doesn't work then run `pip install requirements.txt` to install lacking requirements.

### How to run & Output 
Run `modelEvaluation.py`. The `print_nicely()` function evaluates the 
models and prints perplexities. 

Output: 
1) Perplexities for train data: unigram, bigram, trigram respectively
2) Perplexities for dev data: unigram, bigram, trigram respectively
3) Perplexities for test data: unigram, bigram, trigram respectively
4) Smoothed perplexities using the best performing hyperparameters

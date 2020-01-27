from collections import Counter
import math
from nltk import ngrams

# Given a train file creates unigram, bigram and trigram models
# Returns maps of unigrams, bigrams, trigrams -> their probabilities in the vocabulary
def corpus_to_probabilities(path):

    # Initialize unigram dictionary and set UNK count to 0
    unigram_freq = dict({"UNK": 0})

    # Initialize lists of tokens for n-grams
    unigrams = list()
    bigrams = list()
    trigrams = list()

    # Sentence count
    sent_count = 0

    # Tokenize the file line by line
    f = open(path)
    while True:
        sentences = f.readlines()
        if not sentences:
            break
        else:
            for s in sentences:
                sent_count += 1
                tokens = s.split()
                unigrams += tokens + ["<stop>"]
                bigrams += ["<start>"] + tokens + ["<stop>"]
                trigrams += ["<start>"] + tokens + ["<stop>"]
    f.close()

    # Add UNKs to bigrams, trigrams
    freq = Counter(bigrams)
    tokens_w_unks = ["UNK" if freq[element] < 3 else element for element in bigrams]

    # Create lists of bigrams and trigrams
    bigrams = list(ngrams(tokens_w_unks, 2))
    trigrams = list(ngrams(tokens_w_unks, 3))

    # Create frequency dictionaries
    temp_uni = Counter(unigrams)
    bigram_freq = Counter(bigrams)
    trigram_freq = Counter(trigrams)

    for t in temp_uni:
        if temp_uni[t] < 3:
            unigram_freq["UNK"] += 1
        else:
            unigram_freq[t] = temp_uni[t]

    # How big is the vocabulary
    vocab = sum(unigram_freq.values())
    # Calculate probabilities
    unigram_prob = {k: float(v) / vocab for k, v in unigram_freq.items()}
    unigram_freq["<start>"] = sent_count
    bigram_prob = {k: float(v) / unigram_freq[k[0]] for k, v in bigram_freq.items()}
    trigram_prob = {k: float(v) / bigram_freq[k[:2]] for k, v in trigram_freq.items()}

    vocabulary = [u for u in unigram_prob.keys()]
    return unigram_prob, bigram_prob, trigram_prob, vocabulary

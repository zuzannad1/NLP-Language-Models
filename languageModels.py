from collections import Counter
import math

# Given a file creates unigram, bigram and trigram models
# Returns maps of unigrams, bigrams, trigrams -> their probabilities in the vocabulary
from nltk import ngrams


def corpus_to_probabilities(path):
    unigram_freq = dict({"UNK": 0})
    bigram_freq = dict({"UNK": 0})
    trigram_freq = dict({"UNK": 0})

    unigrams = list()
    bigrams = list()
    trigrams = list()

    sent_count = 0

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

    bigrams = list(ngrams(bigrams, 2))
    trigrams = list(ngrams(trigrams, 3))

    temp_uni = Counter(unigrams)
    temp_bi = Counter(bigrams)
    temp_tri = Counter(trigrams)

    for t in temp_uni:
        if temp_uni[t] < 3:
            unigram_freq["UNK"] += 1
        else:
            unigram_freq[t] = temp_uni[t]

    for t in temp_bi:
        if temp_bi[t] < 3:
            bigram_freq["UNK"] += 1
        else:
            bigram_freq[t] = temp_bi[t]

    for t in temp_tri:
        if temp_tri[t] < 3:
            trigram_freq["UNK"] += 1
        else:
            trigram_freq[t] = temp_tri[t]

    vocab = sum(unigram_freq.values())
    unigram_prob = {k: float(v) / vocab for k, v in unigram_freq.items()}
    ### EVERYTHING WORKS UP TO HERE
    unigram_freq["<start>"] = sent_count

    bigram_prob = {k: math.log((float(v) / unigram_freq[k[0]]), 2) for k, v in bigram_freq.items()}
    print(sum(bigram_freq.values()))

    bigram_freq[("<start>", "<start>")] = sent_count
    trigram_prob = {k: math.log(float(v) / bigram_prob[k[:2]], 2) if k != "UNK" else math.log(float(v) / bigram_prob["UNK"], 2) for k, v in trigram_freq.items()}

    return unigram_prob, bigram_prob, trigram_prob


corpus_to_probabilities("1b_benchmark.train.tokens")




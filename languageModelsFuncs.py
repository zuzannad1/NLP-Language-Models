from nltk import ngrams
import math

# Perplexity
def perplexity(map):
    perplexity = 1
    N = 0
    for word in map.keys():
        N += 1
        perplexity = perplexity * (1 / map[word])
        perplexity = pow(perplexity, 1 / float(N))
    return perplexity




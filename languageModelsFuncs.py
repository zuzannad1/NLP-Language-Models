from nltk import ngrams


# Read in the file line by line
# Transform into tokens
def file_to_tokens(path):
    all_tokens = []
    file = open(path)
    while True:
        lines = file.readlines()
        if not lines:
            break
        else:
            for line in lines:
                tokens = line.split()
                tokens.insert(0, "<start>")
                tokens.append("<stop>")
                all_tokens += tokens
    return all_tokens


# Transforms a file of sentences to bigrams line by line
def file_to_bigrams(path):
    bigrams = list()
    file = open(path)
    while True:
        lines = file.readlines()
        if not lines:
            break
        else:
            for line in lines:
                tokens = line.split()
                tokens.insert(0, "<start>")
                tokens.append("<stop>")
                bigrams += list(ngrams(tokens, 2))
    return bigrams


# Transforms a file of sentences to bigrams line by line
def file_to_trigrams(path):
    trigrams = list()
    file = open(path)
    while True:
        lines = file.readlines()
        if not lines:
            break
        else:
            for line in lines:
                tokens = line.split()
                tokens.insert(0, "<start>")
                tokens.append("<stop>")
                trigrams += list(ngrams(tokens, 3))
    return trigrams


# Perplexity
def perplexity(map):
    perplexity = 1
    N = 0
    for word in map.keys():
        N += 1
        perplexity = perplexity * (1 / map[word])
        print(perplexity)
    perplexity = pow(perplexity, 1 / float(N))
    return perplexity


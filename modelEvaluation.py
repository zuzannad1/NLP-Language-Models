import math
from languageModels import corpus_to_probabilities
import nltk

# Calculates sentence log probability for every sentence
# ngram_prob: map uni, bi, tri-grams -> probability
# n: length of "-gram" (1,2,3)
# sentences: the dataset. list of sentences, where each sentence is a list of tokens
# Returns: a list of scores where [n] -> score of nth sentence
def sentence_log_prob(ngram_prob, n, sentences):
    scores = []
    for sentence in sentences:
        sentence_score = 0
        if n == 1:
            tokens = sentence + ["<stop>"]
        elif n == 2:
            tokens = nltk.bigrams(["<start>"] + sentence + ["<stop>"])
        elif n == 3:
            tokens = nltk.trigrams(["<start>"] + sentence + ["<stop>"])
        else:
            raise ValueError('Parameter "n" has an invalid value %s' % n)
        for token in tokens:
            try:
                p = ngram_prob[token]
                sentence_score += math.log(p, 2)
            except KeyError:
                sentence_score += 0
        scores.append(sentence_score)
    return scores

# Reads in the data file and tokenizes it
# Transforming all OOVs to UNKS
# filename: name of the data file
# vocabulary: the vocabulary based on which OOVs are determined
# Returns: a list of sentences where each sentence is a list of tokens (no UNKS, starts, stops)
def file_to_unked_sentences(filename, vocabulary):
    f = open(filename, 'r')
    corpus = f.readlines()
    f.close()
    sentences = []
    for sentence in corpus:
        temp = sentence.split()
        temp_tokens = ["UNK" if t not in vocabulary else t for t in temp]
        sentences += [temp_tokens]
    return sentences

# Calculates perplexity on sentences from a data file
# filename: file on which sentences perplexity is calculated
# scores: a list of log probabilities calculated in sentence_log_prob
def perplexity(filename, scores):
    f = open(filename, 'r')
    sentences = f.readlines()
    f.close()
    M = 0
    for sentence in sentences:
        words = sentence.split()
        M += len(words) + 1

    perplexity = 0
    for score in scores:
        perplexity += score # assumes log probability

    perplexity /= M
    perplexity = 2 ** (-1 * perplexity)

    return "Perplexity is " + str(perplexity)

# Implements smoothing of trigram probability based on probabilities of uni, bi, tri-gram probabilities
# unigrams:
# bigrams:
# trigrams:
# sentences:
def smoothing(unigrams, bigrams, trigrams, sentences):
    scores = []
    lambda_ = 1.0 / 3
    for sentence in sentences:
        interpolated_score = 0
        for trigram in nltk.trigrams(["<start>"] + sentence + ["<stop>"]):
            try:
                p3 = trigrams[trigram]
            except KeyError:
                p3 = -1000
            try:
                p2 = bigrams[trigram[1:3]]
            except KeyError:
                p2 = -1000
            try:
                p1 = unigrams[trigram[2]]
            except KeyError:
                p1 = -1000
            interpolated_score += math.log(0.01 * (2 ** p3) + 0 * (2 ** p2) + 0.99 * (2 ** p1), 2)
        scores.append(interpolated_score)
    return scores

def print_nicely():
    train = "1b_benchmark.train.tokens"
    dev = "1b_benchmark.dev.tokens"
    test = "1b_benchmark.test.tokens"
    unigram_probs, bigram_probs, trigram_probs, vocabulary = corpus_to_probabilities(train)
    train_sentences = file_to_unked_sentences(train, vocabulary)
    dev_sentences = file_to_unked_sentences(dev, vocabulary)
    test_sentences = file_to_unked_sentences(test, vocabulary)

    print("PERPLEXITIES FOR TRAIN DATA")
    print("***************************")
    print("UNIGRAM's " + perplexity(train, (sentence_log_prob(unigram_probs, 1, train_sentences))))
    print("BIGRAM's " + perplexity(train, (sentence_log_prob(bigram_probs, 2, train_sentences))))
    print("TRIGRAM's " + perplexity(train, (sentence_log_prob(trigram_probs, 3, train_sentences))))
    print("\n\n")
    print("PERPLEXITIES FOR DEV DATA")
    print("**************************")
    print("UNIGRAM's " + perplexity(dev, (sentence_log_prob(unigram_probs, 1, dev_sentences))))
    print("BIGRAM's " + perplexity(dev, (sentence_log_prob(bigram_probs, 2, dev_sentences))))
    print("TRIGRAM's " + perplexity(dev, (sentence_log_prob(trigram_probs, 3, dev_sentences))))
    print("\n\n")
    print("PERPLEXITIES FOR TEST DATA")
    print("***************************")
    print("UNIGRAM's " + perplexity(test, (sentence_log_prob(unigram_probs, 1, test_sentences))))
    print("BIGRAM's " + perplexity(test, (sentence_log_prob(bigram_probs, 2, test_sentences))))
    print("TRIGRAM's " + perplexity(test, (sentence_log_prob(trigram_probs, 3, test_sentences))))

    print("SMOOTHED TRAIN: ")
    print(perplexity(train, smoothing(unigram_probs, bigram_probs,trigram_probs, train_sentences)))
    print("SMOOTHED DEV: ")
    print(perplexity(dev, smoothing(unigram_probs, bigram_probs,trigram_probs, dev_sentences)))
    print("SMOOTHED TEST: ")
    print(perplexity(test, smoothing(unigram_probs, bigram_probs,trigram_probs, dev_sentences)))
print_nicely()
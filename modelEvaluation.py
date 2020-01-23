import math
from languageModels import corpus_to_probabilities
import nltk

# Calculates sentence log probability for every sentence
# ngram_prob: map uni, bi, tri-grams -> probability
# n: length of "-gram" (1,2,3)
# filename: the dataset. list of sentences
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
            p = ngram_prob[token]
            sentence_score += math.log(p, 2)
        scores.append(sentence_score)
    return scores


# TOO SLOW
def file_to_unked_sentences(filename, unked_tokens):
    f = open(filename, 'r')
    corpus = f.readlines()
    f.close()
    sentences = []
    for sentence in corpus:
        temp = sentence.split()
        temp_tokens = ["UNK" if t in unked_tokens else t for t in temp]
        sentences += [temp_tokens]
    return sentences


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
        perplexity += score # assume log probability

    perplexity /= M
    perplexity = 2 ** (-1 * perplexity)

    return "perplexity is " + str(perplexity)

def print_nicely():
    train = "1b_benchmark.train.tokens"
    dev = "1b_benchmark.dev.tokens"
    test = "1b_benchmark.test.tokens"
    train_unked_tokens = corpus_to_probabilities(train)[3]
    # dev_unked_tokens = corpus_to_probabilities(dev)[3]
    # test_unked_tokens = corpus_to_probabilities(test)[3]
    sentences = file_to_unked_sentences(train, train_unked_tokens)

    print("PERPLEXITIES FOR TRAIN DATA")
    print("---------------------------")
    print("UNIGRAM's " + perplexity(train, (sentence_log_prob(corpus_to_probabilities(train)[0], 1, sentences))))
    print("BIGRAM's " + perplexity(train, (sentence_log_prob(corpus_to_probabilities(train)[1], 2, sentences))))
    print("TRIGRAM's " + perplexity(train, (sentence_log_prob(corpus_to_probabilities(train)[2], 3, sentences))))
    # print("PERPLEXITIES FOR DEV DATA")
    # print("-------------------------")
    # print("UNIGRAM's " + perplexity(dev, (sentence_log_prob(corpus_to_probabilities(dev)[0], 1, dev))))
    # print("BIGRAM's " +  perplexity(dev, (sentence_log_prob(corpus_to_probabilities(dev)[1], 2, dev))))
    # print("PERPLEXITIES FOR TEST DATA")
    # print("--------------------------")

print_nicely()
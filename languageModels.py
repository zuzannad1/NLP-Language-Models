import re
import math

# UNK token used for tokens occuring less than three times
UNK = None
# Sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"


# Read sentences from file
def read_from_file(path):
    with open(path, "r") as f:
        return [re.split("\s+", line.rstrip('\n')) for line in f]

# UNIGRAM LANGUAGE MODEL
class UnigramLanguageModel:

    # Constructor
    def __init__(self, sentences, smoothing=False):
        self.unigram_freq = dict()
        self.length = 0
        for sentence in sentences:
            for word in sentence:
                self.unigram_freq[word] = self.unigram_freq.get(word, 0) + 1
                if word != SENTENCE_START and word != SENTENCE_END:
                    self.length += 1
        # Subtract one because the frequency dictionary contains SENTENCE_STARTs
        self.unique_words = len(self.unigram_freq) - 1
        self.smoothing = smoothing


    def
# BIGRAM LANGUAGE MODEL

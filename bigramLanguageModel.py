from collections import Counter
import languageModelsFuncs


# Map: unique bigrams -> bigram frequency
# Convert low frequency tokens to UNKs
def bigram_frequency(bigrams):
    temp = Counter(bigrams)
    bigram_frequencies = dict({"UNK": 0})
    for t in temp:
        if temp[t] < 3:
           bigram_frequencies["UNK"] += 1
        else:
            bigram_frequencies[t] = temp[t]
    return bigram_frequencies


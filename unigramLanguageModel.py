import nltk
from collections import Counter

sentence = "Marc D. Seitles , the lawyer who represents David M. Packouz , the licensed massage therapist who is AEY " \
           "'s former vice president , said he had sent a letter to Congress saying Mr. Packouz would speak publicly " \
           "only if he was granted immunity from prosecution .In the annals of circus and variety history , " \
           "no family better deserved the description of the royal family of juggling than did the Brunns .One of the " \
           "defendants was found not guilty and the jury was unable to reach a verdict for four other men on trial . " \
           "Nursing Homes : Genworth 's 2009 Cost of Care Survey analyzed long term care costs in two regions in " \
           "Hawaii : Honolulu and the rest of the state . Players can take their performances to the next level by " \
           "turning their best 30 second clips into their own mini music videos using a selection of special editing " \
           "effects . Sabathia came in fifth . "

# Transform the sentence into tokens
tokens = nltk.word_tokenize(sentence)

# Initialise a frequency map to count
word_frequencies = dict({"UNK":0})

# Map tokens to token frequency
# Convert low frequency tokens to UNKs
for t in Counter(tokens):
    if Counter(tokens)[t] < 3:
        word_frequencies["UNK"] += 1
    else: word_frequencies[t] = Counter(tokens)[t]

# Count all words (all tokens that have freq < 3 are already UNKS)
total_words = sum(word_frequencies.values())

# Convert frequencies to probabilities
for word in word_frequencies:
    word_frequencies[word] = float(word_frequencies[word])/float(total_words)


# Calculates probability of a unigram in the whole vocabulary
def unigram_probability(token):
    num = word_frequencies.get(token,0)
    denom = total_words
    return float (num) / float (denom)


# Unigram perplexity
def perplexity(testset, model):
    perplexity = 1
    N = 0
    for word in testset:
        N += 1
        perplexity = perplexity * (1/model[word])
    perplexity = pow(perplexity, 1/float(N))
    return perplexity

print(perplexity(word_frequencies.keys(), word_frequencies))
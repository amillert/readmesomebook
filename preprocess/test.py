import numpy as np
from collections import Counter

seps = ["...", ".", "?", "!", ".\"", "!\"", "?\""]
text = open("/home/amillert/private/readmesomebook/data/books/the-governors-man.txt").read().replace("\n", " ")
for sep in seps:
    text = text.replace(sep, f"{sep}{{")
text = text.split("{")
print(f"{len(text)} qualified sentences after splitting")
tokens = [word.lower() for sentence in [sentence.split() for sentence in text] for word in sentence]
print(f"{len(tokens)} tokens")

english_stop_words = open("/home/amillert/private/readmesomebook/data/unique-stop-words/english-unique-sw.txt").read().split()
vocab = sorted(list(set(tokens)))
print(f"{len(vocab)} words in vocabulary")

def check(word, sw):
    for stop in sw:
        # if stop.startswith(word): return False
        if stop.startswith(word) and len(word) / len(stop) >= 0.6: return False
    return True

reduced_vocab = [word for word in vocab if check(word, english_stop_words)]
print(f"{len(reduced_vocab)} tokens in vocab after reducing")
print(f"Vocab reduced by {(1.0 - len(reduced_vocab) / len(vocab)) * 100.0}%")

reduced_tokens = [word for word in tokens if check(word, english_stop_words)]
print(f"{len(reduced_tokens)} tokens after reducing tokens")
print(f"Reduced tokens by {(1.0 - len(reduced_tokens) / len(tokens)) * 100.0}%")

word_cnts = Counter(tokens)

word_freqs = {word: word_cnts[word] / len(tokens) for word in reduced_vocab}

print(sorted(word_cnts.items(), key=lambda l: -l[1])[:20])
print(sorted(word_freqs.items(), key=lambda l: -l[1])[:20])

word2idx = {word: idx for (idx, word) in enumerate(reduced_vocab)}

FAKE_WORD = "fake"
reduced_vocab.append(FAKE_WORD)
word2idx.update({FAKE_WORD: len(word2idx)})
idx2word = {v: k for (k, v) in word2idx.items()}
FAKE = word2idx[FAKE_WORD]

target_context_pairs = []
window = 3

# TODO do something with paragraphs and stuff so padding can be used more often; not just one sentence in a sence

# for line in data:
for center_idx in range(len(reduced_tokens)):
    context_words = []
    for context in range(-window, window + 1):
        context_idx = center_idx + context
        # verify statements
        if context_idx < 0 or context_idx >= len(reduced_tokens):
            context_words.append(FAKE)
        elif context_idx != center_idx:
            context_words.append(word2idx[reduced_tokens[context_idx]])
    
    target_context_pairs.append(tuple([word2idx[reduced_tokens[center_idx]], context_words]))

X, y = zip(*target_context_pairs)

print(len(X))

for i, x in enumerate(target_context_pairs):
    if i < 10: print(x)
    else: break


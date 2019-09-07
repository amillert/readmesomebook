import numpy as np
from torch.utils.data import Dataset
from collections import Counter


class BookDataset(Dataset):
    def __init__(self, args):
        separators = ["...", ".", "?", "!", ".\"", "!\"", "?\""]
        english_stop_words = list(set([word.lower() for word in open(args.stopwords_path).read().split()]))
        text = open(args.input_path).read().replace("\n", "")

        for sep in separators:
            text = text.replace(sep, f"{sep}{{")

        def check(word, sw):
            for stop in sw:
                if word == stop:
                    # print(f"{word} odrzucone")
                    return False
                # if stop.startswith(word) and len(word) / len(stop) >= 0.6: return False
            return True

        text = text.split("{")
        print(f"{len(text)} qualified sentences after splitting")
        self.tokens = [word.lower() for sentence in [sentence.split() for sentence in text] for word in sentence if check(word.lower(), english_stop_words)]
        print(f"{len(self.tokens)} tokens")
        self.vocab = [word for word in sorted(list(set(self.tokens)))]
        print(f"vocab size: {len(self.vocab)}")

        word_cnts = Counter(self.tokens)
        word_freqs = {word: word_cnts[word] / len(self.tokens) for word in self.vocab}
        # print(sorted(word_cnts.items(), key=lambda l: -l[1])[:5])
        # print(sorted(word_freqs.items(), key=lambda l: -l[1])[:5])

        FAKE_WORD = "fake"
        self.vocab.append(FAKE_WORD)
        self.vocab_size = len(self.vocab)
        self.word2idx = {word: idx for (idx, word) in enumerate(self.vocab)}
        self.idx2word = {idx: word for (word, idx) in self.word2idx.items()}
        self.FAKE = self.word2idx[FAKE_WORD]

        for token in self.tokens:
            if token not in self.word2idx.keys(): print(token)

        target_context_pairs = []

        # TODO do something with paragraphs and stuff so padding can be used more often; not just one sentence in a sence

        # for line in data:
        for center_idx in range(len(self.tokens)):
            context_words = []
            for context in range(-args.window, args.window + 1):
                context_idx = center_idx + context
                # verify statements
                if context_idx < 0 or context_idx >= len(self.tokens):
                    context_words.append(self.FAKE)
                elif context_idx != center_idx:
                    context_words.append(self.word2idx[self.tokens[context_idx]])

            target_context_pairs.append(tuple([self.word2idx[self.tokens[center_idx]], np.array(context_words)]))

        self.X, self.y = zip(*target_context_pairs)
        assert len(self.X) == len(self.y)
        self.len = len(self.X)
        print(self.len)

        for i, x in enumerate(target_context_pairs):
            if i < 5: print(x)
            else: break

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.len


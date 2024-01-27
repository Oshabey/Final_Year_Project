import numpy as np 
import spacy
import random
import pandas as pd 
from tqdm import tqdm
from collections import Counter

df = pd.read_csv("data.csv")


def lda_model():
    alpha = 0.1
    beta = 0.1
    num_topics = 20
    sp = spacy.load("en_core_web_sm")

    np.random.seed(42)
    random.seed(42)

    def generate_frequencies(data, max_docs=10000):
        freqs = Counter()
        all_stopwords = sp.Defaults.stop_words
        nr_tokens = 0

        for doc in data[:max_docs]:
            tokens = sp.tokenizer(doc)

            for token in tokens:
                token_text = token.text.lower()

                if token_text not in all_stopwords and token.is_alpha:
                    nr_tokens += 1
                    freqs[token_text] += 1

        return freqs            

    def get_vocab(freqs, freq_threshold=3):
        vocab = {}
        vocab_idx_str = {}
        vocab_idx = 0

        for word in freqs:
            if freqs[word] >= freq_threshold:
                vocab[word] = vocab_idx
                vocab_idx_str[vocab_idx] = word
                vocab_idx += 1

        return vocab, vocab_idx_str        

    def tokenize_dataset(data, vocab, max_docs=10000):
        nr_tokens = 0
        nr_docs = 0
        docs = []

        for doc in data[:max_docs]:
            tokens = sp.tokenizer(doc)

            if len(tokens) > 1:
                doc = []
                for token in tokens:
                    token_text = token.text.lower()
                    if token_text in vocab:
                        doc.append(token_text)
                        nr_tokens += 1
                nr_docs += 1  
                docs.append(doc)

        print(f"Number of papers: {nr_docs}")
        print(f"Number of tokens: {nr_tokens}")

        corpus = []
        for doc in docs:
            corpus_d = []

            for token in doc:
                corpus_d.append(vocab[token])

            corpus.append(np.asarray(corpus_d))

        return docs, corpus                      

    data = df['title'].sample(frac=1.0, random_state=42).values
    freqs = generate_frequencies(data)
    vocab, vocab_idx_str = get_vocab(freqs)
    docs, corpus = tokenize_dataset(data, vocab)
    vocab_size = len(vocab)
    print(f"vocab size: {vocab_size}")

    def lda_collapsed_gibbs(corpus, num_iter=200):
        z = []
        num_docs = len(corpus)

        for _, doc in enumerate(corpus):
            zd = np.random.randint(low=0, high=num_topics, size = (len(doc)))
            z.append(zd)

        ndk = np.zeros((num_docs, num_topics))
        for d in range(num_docs):
            for k in range(num_topics):
                ndk[d, k] = np.sum(z[d] == k)

        nkw = np.zeros((num_topics, vocab_size))
        for doc_idx, doc in enumerate(corpus):
            for i, word in enumerate(doc):
                topic = z[doc_idx][i]
                nkw[topic, word] += 1

        nk = np.sum(nkw, axis=1)
        topic_list = [i for i in range(num_topics)]
        
        for _ in tqdm(range(num_iter)):
            for doc_idx, doc in enumerate(corpus):
                for i in range(len(doc)):
                    word = doc[i]
                    topic = z[doc_idx][i]

                    ndk[doc_idx, topic] -= 1
                    nkw[topic, word] -= 1
                    nk[topic] -= 1

                    p_z = (ndk[doc_idx, :] + alpha) * (nkw[:, word] + beta) / (nk[:] + beta*vocab_size)
                    topic = random.choices(topic_list, weights=p_z, k=1)[0]

                    z[doc_idx][i] = topic
                    ndk[doc_idx, topic] += 1
                    nkw[topic, word] += 1
                    nk[topic] += 1

        return z, ndk, nkw, nk

    z, ndk, nkw, nk = lda_collapsed_gibbs(corpus)            

    phi = nkw / nk.reshape(num_topics, 1)

    num_words = 10
    for k in range(num_topics):
        most_commn_words = np.argsort(phi[k])[::-1][:num_words]
        print(f"Topic {k} most common words: ")

        for word in most_commn_words:
            print(vocab_idx_str[word])

        print("\n")

lda_model()
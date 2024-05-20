from collections import Counter
from math import log
from collections import defaultdict
import numpy as np

def calculate_base_bm25(tokenized_words, k1=1.5, b=0.75):

    tokenized_text = tokenized_words

    term_freq = Counter(tokenized_text)

    avgdl = sum(term_freq.values()) / len(term_freq)


    bm25_dict = {}


    for term, freq in term_freq.items():
        idf = log((len(tokenized_text) - freq + 0.5) / (freq + 0.5) + 1)
        bm25 = idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (len(tokenized_text) / avgdl)))
        bm25_dict[term] = bm25

    bm25_dict = dict(sorted(bm25_dict.items(), key=lambda item: item[1], reverse=True))

    return bm25_dict

def bm25_compare_with_tokens(tokens1, tokens2, k=1.5, b=0.75):

    word_counts_doc1 = defaultdict(int)
    word_counts_doc2 = defaultdict(int)
    for word in tokens1:
        word_counts_doc1[word] += 1
    for word in tokens2:
        word_counts_doc2[word] += 1


    all_words = set(word_counts_doc1.keys()) | set(word_counts_doc2.keys())


    doc_freq = defaultdict(int)
    for word in all_words:
        if word in word_counts_doc1:
            doc_freq[word] += 1
        if word in word_counts_doc2:
            doc_freq[word] += 1

    num_docs = 2  
    idf = {word: np.log((num_docs - doc_freq[word] + 0.5) / (doc_freq[word] + 0.5) + 1) for word in all_words}

   
    bm25_values = np.zeros((2, len(all_words)))

    for i, word in enumerate(all_words):
        
        word_freq_doc1 = word_counts_doc1[word]
        word_freq_doc2 = word_counts_doc2[word]

        
        for j, word_freq in enumerate([word_freq_doc1, word_freq_doc2]):
            doc_length = len(tokens1) if j == 0 else len(tokens2)
            bm25_values[j][i] = (idf[word] * word_freq * (k + 1)) / (word_freq + k * (1 - b + b * (doc_length / len(all_words))))

    
    normalized_bm25_values = bm25_values / np.linalg.norm(bm25_values, axis=1, keepdims=True)

    
    bm25_dicts = [{word: normalized_bm25_values[0][i] for i, word in enumerate(all_words)},
                  {word: normalized_bm25_values[1][i] for i, word in enumerate(all_words)}]

    
    cosine_similarity_bm25 = np.dot(normalized_bm25_values[0], normalized_bm25_values[1])

    return bm25_dicts, cosine_similarity_bm25



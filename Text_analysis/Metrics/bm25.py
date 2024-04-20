from collections import Counter
from math import log
from collections import defaultdict
import numpy as np

def calculate_base_bm25(text, tokenized_words, k1=1.5, b=0.75):
    # Предобработка текста и токенизация
    tokenized_text = tokenized_words

    # Словарь частот термов в документе
    term_freq = Counter(tokenized_text)

    # Вычисляем среднюю длину документа
    avgdl = sum(term_freq.values()) / len(term_freq)

    # Инициализация переменной для хранения суммы BM25 для каждого терма в тексте
    bm25_dict = {}

    # Рассчитываем BM25 для каждого терма
    for term, freq in term_freq.items():
        idf = log((len(tokenized_text) - freq + 0.5) / (freq + 0.5) + 1)  # +1 для избежания деления на ноль
        bm25 = idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (len(tokenized_text) / avgdl)))
        bm25_dict[term] = bm25

    # Сортируем по убыванию BM25
    bm25_dict = dict(sorted(bm25_dict.items(), key=lambda item: item[1], reverse=True))

    return bm25_dict

def bm25_compare_with_tokens(tokens1, tokens2, k=1.5, b=0.75):
    # Подсчет частот слов в каждом документе
    word_counts_doc1 = defaultdict(int)
    word_counts_doc2 = defaultdict(int)
    for word in tokens1:
        word_counts_doc1[word] += 1
    for word in tokens2:
        word_counts_doc2[word] += 1

    # Получение всех уникальных слов
    all_words = set(word_counts_doc1.keys()) | set(word_counts_doc2.keys())

    # Подсчет количества документов, содержащих каждое слово
    doc_freq = defaultdict(int)
    for word in all_words:
        if word in word_counts_doc1:
            doc_freq[word] += 1
        if word in word_counts_doc2:
            doc_freq[word] += 1

    # Вычисление IDF для каждого слова
    num_docs = 2  # У нас два документа
    idf = {word: np.log((num_docs - doc_freq[word] + 0.5) / (doc_freq[word] + 0.5) + 1) for word in all_words}

    # Подготовка данных для вычисления BM25
    bm25_values = np.zeros((2, len(all_words)))

    for i, word in enumerate(all_words):
        # Подсчет вхождений слова в каждый документ
        word_freq_doc1 = word_counts_doc1[word]
        word_freq_doc2 = word_counts_doc2[word]

        # Вычисление BM25 для каждого документа
        for j, word_freq in enumerate([word_freq_doc1, word_freq_doc2]):
            doc_length = len(tokens1) if j == 0 else len(tokens2)
            bm25_values[j][i] = (idf[word] * word_freq * (k + 1)) / (word_freq + k * (1 - b + b * (doc_length / len(all_words))))

    # Нормализация BM25 векторов
    normalized_bm25_values = bm25_values / np.linalg.norm(bm25_values, axis=1, keepdims=True)

    # Создание словарей BM25 для каждого документа
    bm25_dicts = [{word: normalized_bm25_values[0][i] for i, word in enumerate(all_words)},
                  {word: normalized_bm25_values[1][i] for i, word in enumerate(all_words)}]

    # Вычисление косинусного расстояния между BM25 векторами документов
    cosine_similarity_bm25 = np.dot(normalized_bm25_values[0], normalized_bm25_values[1])

    return bm25_dicts, cosine_similarity_bm25



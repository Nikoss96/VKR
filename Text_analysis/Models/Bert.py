from sentence_transformers import SentenceTransformer, util
from scipy.spatial.distance import cosine

def compare_texts_bert(text1, text2):
    # Загрузка модели
    model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

    # Получение векторных представлений для обоих текстов
    embeddings = model.encode([text1, text2], convert_to_tensor=True)

    # Вычисление косинусного расстояния между векторами
    similarity = 1 - cosine(embeddings[0], embeddings[1])

    return similarity
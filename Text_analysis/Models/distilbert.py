from transformers import DistilBertTokenizer, DistilBertModel
import torch

# Загрузка предобученной модели DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Пример текстов для сравнения
text1 = "собака ест"
text2 = "кошка сидит на стуле"

# Токенизация текстов
inputs1 = tokenizer(text1, return_tensors='pt')
inputs2 = tokenizer(text2, return_tensors='pt')

# Получение скрытых представлений для каждого текста
with torch.no_grad():
    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)

# Используем скрытое состояние CLS токена для сравнения текстов
hidden_states1 = outputs1.last_hidden_state
hidden_states2 = outputs2.last_hidden_state

cls_embedding1 = hidden_states1[:, 0, :]
cls_embedding2 = hidden_states2[:, 0, :]

# Рассчитываем косинусное расстояние между скрытыми представлениями текстов
similarity = torch.nn.functional.cosine_similarity(cls_embedding1, cls_embedding2, dim=1)

print("Similarity between the two texts:", similarity.item())
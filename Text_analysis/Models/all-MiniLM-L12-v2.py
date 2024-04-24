from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from Metrics.Similarity import get_cosine_similarity 
from Preprocessing.preprocess_text import *


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)




sentences = ['я'+preprocess_text('я вышел гулять',language = "russian"),'я'+preprocess_text('я вышел на пробежку',language = "russian")]


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')


encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')


with torch.no_grad():
    model_output = model(**encoded_input)


sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])


sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print(f"Сходство текстов: {get_cosine_similarity(sentence_embeddings[0],sentence_embeddings[1])*100}")
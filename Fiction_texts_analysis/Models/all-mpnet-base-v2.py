from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from Metrics.Similarity import get_cosine_similarity 
from Preprocessing.preprocess_text import *
#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

file2 = "C:/Users/nikch/Text_analysis/Texts/1_Blow-Up.docx"
file1 = "C:/Users/nikch/Text_analysis/Texts/Blow_internet.docx"
# C:/Users/nikch/Text_analysis/Texts/1_Blow-Up.docx
# C:/Users/nikch/Text_analysis/Texts/10_End_of_the_Game.docx
dicted_file1 = read_docx(file_path = file1, language = 'english')
dicted_file2 = read_docx(file_path = file2, language = 'english')
preprocessed_text1 = preprocess_text(text = dicted_file1['text'], language = "english")
preprocessed_text2 = preprocess_text(text = dicted_file2['text'], language = "english")

#sentences = [preprocessed_text1,preprocessed_text2]
# Sentences we want sentence embeddings for
sentences = [dicted_file1['text'],dicted_file2['text']]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print(f"Сходство текстов: {get_cosine_similarity(sentence_embeddings[0],sentence_embeddings[1])*100}")
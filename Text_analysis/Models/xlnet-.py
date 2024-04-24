from transformers import XLNetTokenizer, XLNetForSequenceClassification, XLNetModel
import torch

train_args = {
'learning_rate': 3e-5,
'num_train_epochs': 3, ###
'max_seq_length': 384,
'doc_stride': 384,
'max_query_length': 64,
'max_answer_length':100,
'n_best_size' :3,
'early_stopping_consider_epochs': True,
'overwrite_output_dir': False, #####
'reprocess_input_data': False,
'gradient_accumulation_steps': 8,
'use_early_stopping': True,
'evaluate_during_traing': True,
'save_eval_checkpoints' : True,
'save_model_every_epoch': True,
'save_steps': 2000,
'n_gpu': 2, ###
'train_batch_size': 4,
'dataloader_num_worker': 8, ###
'use_early_stopping': True,
'early_stopping_delta': 0.01,
'early_stopping_metric': 'eval_loss',
'early_stopping_metric_minimize': True,
'early_stopping_patience': 3,
'evaluate_during_training_steps': 1000,
'mem_len': 1024, ### Xlnet

}

cuda_available = torch.cuda.is_available()

#model_name = 'xlnet-base-cased'
model_name = 'xlnet-large-cased'
tokenizer = XLNetTokenizer.from_pretrained(model_name)
model = XLNetModel.from_pretrained(model_name)
model1 = XLNetForSequenceClassification.from_pretrained(model_name)

def compare_texts_xlnet(text1, text2):
    
    inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
        
        outputs3 = model1(**inputs1)
        outputs4 = model1(**inputs2)
    
    print(outputs3.hidden_states)
    #distance1 = torch.cdist(outputs3.hidden_states.mean(dim=1), outputs4.hidden_states.mean(dim=1))
    
    distance = torch.cdist(outputs1.last_hidden_state.mean(dim=1), outputs2.last_hidden_state.mean(dim=1))
    percent = 100 - distance.item()
    #print(distance)
    return distance.item(),percent
    #print (outputs1.keys())
    #print(outputs1['logits'])
    #print(outputs1['mems'])


text1 = "собака"
text2 = "собака"
compare_texts_xlnet(text1, text2)
distance,percent = compare_texts_xlnet(text1, text2)
print(f"{distance}")
print(f"Процент сходства: {percent}")
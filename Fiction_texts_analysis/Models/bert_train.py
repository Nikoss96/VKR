import pandas as pd
import torch
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm

frame = pd.read_parquet('..\\Texts\\train.parquet', engine='pyarrow')
data = frame[["text_1", "text_2", "class"]]

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
max_length = 128

def preprocess_data(row):
    encoding = tokenizer(row['text_1'], row['text_2'], truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
    similarity = int(row['class'])
    return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'labels': torch.tensor(similarity + 1, dtype=torch.long)
    }

processed_data = data.apply(preprocess_data, axis=1)

train_size = int(0.8 * len(processed_data))
train_data = processed_data[:train_size]
val_data = processed_data[train_size:]

batch_size = 8
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    logging_dir='./logs'
)

model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=3)

data_collator = DataCollatorWithPadding(tokenizer)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data.to_list(),
    eval_dataset=val_data.to_list(),
    data_collator=data_collator
)

def show_loss(worst, results):
    global pbar
    pbar.set_description(results)
    return worst

with tqdm(total=training_args.num_train_epochs) as pbar:
    trainer.callback = lambda worst, results: show_loss(worst, results)
    trainer.train()
    pbar.update()

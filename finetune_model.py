import pandas as pd
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, pipeline
from sklearn.model_selection import train_test_split

excel_file = "/data[extracted]_4350_headersandsummaries.xlsx"
csv_file = "/data/csv_versions/[extracted]_4350_headersandsummaries.csv"

df = pd.read_excel(excel_file)
df = df[['summary', 'header']]

df.to_csv(csv_file, index=False)

train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)


train_csv_path = "/data/csv_versions/train.csv"
validation_csv_path = "/data/csv_versions/validation.csv"
train_df.to_csv(train_csv_path, index=False)
validation_df.to_csv(validation_csv_path, index=False)

print(f"Excel dosyasından {csv_file} dosyasına dönüştürüldü.")

dataset = load_dataset('csv', data_files={
    'train': train_csv_path,  
    'validation': validation_csv_path  
})


#model_name = "ytu-ce-cosmos/turkish-gpt2-large"
#model_name = "ytu-ce-cosmos/Turkish-Llama-8b-DPO-v0.1"
model_name = "ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 'pad_token' ayarı
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):

    # Task tipi başlıktan özete olduğu için header to summary yaptık
    inputs = tokenizer(examples['header'], padding="max_length", truncation=True, max_length=128)
    outputs = tokenizer(examples['summary'], padding="max_length", truncation=True, max_length=128)

    # Labels, hedef metnin tokenize edilmiş input_ids değeridir
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_dir='./logs',
    logging_steps=100,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer
)

trainer.train()

fine_tuned_model_path = "./drive/MyDrive/nlp_final_projesi/cosmos750-task2/fine_tuned_gpt2"

model.save_pretrained(fine_tuned_model_path)
tokenizer.save_pretrained(fine_tuned_model_path)
print("Fine-tuned model başarıyla kaydedildi.")

# Deneme
fine_tuned_model = GPT2LMHeadModel.from_pretrained(fine_tuned_model_path)
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(fine_tuned_model_path)
generator = pipeline("text-generation", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)

# Test girdisi
test_input = "Özet: Türkiye ekonomisi toparlanıyor."
output = generator(test_input, max_length=30, num_return_sequences=1)
print("Üretilen metin:")
print(output[0]["generated_text"])

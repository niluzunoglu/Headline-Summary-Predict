import os
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

FINE_TUNED_MODEL_DIR = "./drive/MyDrive/nlp_final_projesi/cosmos750-task2/fine_tuned_gpt2"
MODEL_NAME = FINE_TUNED_MODEL_DIR

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", load_in_4bit=True).to("cuda")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def run_model_batch(input_texts):
    print(f"[INFO] run_model_batch başlatılıyor: {len(input_texts)} metin için...")
    start_time = time.time()
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to("cuda")

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=30, 
        temperature=1.0
    )

    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    print(f"[INFO] run_model_batch tamamlandı: {len(input_texts)} metin için {time.time() - start_time:.2f} saniye.")
    return generated_texts

def process_batch(batch, batch_size):
    if TASK_TYPE == "ozettenbasliga":
        prompts = ["Verilen özet metnine göre anlamlı bir başlık oluştur. İşte özet:" + row["summary"] for _, row in batch.iterrows()]
    elif TASK_TYPE == "basliktanozete":
        prompts = ["Verilen başlığa göre anlamlı bir haber metni oluştur. İşte başlık:" + row["header"] for _, row in batch.iterrows()]

    start_time = time.time()
    generated_contents = run_model_batch(prompts)
    end_time = time.time()

    batch_results = []
    for idx, (row, generated_content) in enumerate(zip(batch.itertuples(index=False), generated_contents)):
        if TASK_TYPE == "ozettenbasliga":
            batch_results.append({
                "summary": row.summary,
                "real_header": row.header,
                "generated_header": generated_content,
                "model": MODEL_NAME,
                "time": (end_time - start_time) / batch_size,
                "prompt": prompts[idx]
            })
        elif TASK_TYPE == "basliktanozete":
            batch_results.append({
                "baslik": row.header,
                "real_summary": row.summary,
                "generated_summary": generated_content,
                "model": MODEL_NAME,
                "time": (end_time - start_time) / batch_size,
                "prompt": prompts[idx]
            })
        print(f"[INFO] İşlendi: {idx + 1}/{len(batch)} | Üretilen: {generated_content[:30]}...")
    return batch_results

# Batch Bölme
def split_into_batches(data, batch_size):
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

# Paralel İşleme
def parallel_process(data, batch_size=10):
    batches = split_into_batches(data, batch_size)
    print(f"[INFO] {len(batches)} adet batch işlenecek.")
    results = []
    for i, batch in enumerate(batches):
        results.extend(process_batch(batch, batch_size))  
        print(f"[INFO] Batch {i + 1}/{len(batches)} tamamlandı.")
    return results

if __name__ == "__main__":
    
    file_path = "/data/[evaluation]_550_headersandsummaries.xlsx"
    data = pd.read_excel(file_path)

    start_time = time.time()
    batch_size = 32
    TASK_TYPE = "ozettenbasliga"  # veya "basliktanozete"
    print("[INFO] İşlem başlıyor...")

    model_results = parallel_process(data, batch_size=batch_size)

    print("[INFO] Tüm işlemler tamamlandı. Sonuçlar derleniyor...")
    results_df = pd.DataFrame(model_results)

    if TASK_TYPE == "ozettenbasliga":
        output_path = "/content/finetuned_cosmos750_task1_ozettenbasliga_tahminleri_son.xlsx"
    elif TASK_TYPE == "basliktanozete":
        output_path = "/content/finetuned_cosmos750_task2_basliktanozete_tahminleri_son.xlsx"

    if os.path.exists(output_path):
        existing_results = pd.read_excel(output_path)
        combined_results = pd.concat([existing_results, results_df], ignore_index=True)
    else:
        combined_results = results_df

    combined_results.to_excel(output_path, index=False)
    print(f"[INFO] Sonuçlar {output_path} dosyasına kaydedildi.")
    print(f"[INFO] Toplam Süre: {time.time() - start_time:.2f} saniye.")

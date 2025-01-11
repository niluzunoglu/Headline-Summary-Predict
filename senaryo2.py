import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os

from evaluator import evaluate

#MODEL_NAME = "ytu-ce-cosmos/turkish-gpt2-large"
#MODEL_NAME = "ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1"
#MODEL_NAME = "TURKCELL/Turkcell-LLM-7b-v1"
#MODEL_NAME = "Orbina/Orbita-v0.1"
MODEL_NAME = "NovusResearch/Novus-7b-tr_v1"

#TASK_TYPE = "ozettenbasliga"
TASK_TYPE = "basliktanozete"

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
        max_new_tokens=30,  # Başlık uzunluğuna uygun değer
        temperature=1.0
    )

    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    print("Generated texts : ", generated_texts)
    print("Input texts: ",input_texts)
    print(f"[INFO] run_model_batch tamamlandı: {len(input_texts)} metin için {time.time() - start_time:.2f} saniye.")
    return generated_texts

# Batch İşleme
def process_batch(batch, batch_size):


    threeshot = """
    Özet: Mustafa Sarıgül, Büyükşehir Yasası'yla Şişli'den alınan mahallelerin geri alınması için Anayasa Mahkemesi'ne başvuracak. Başlık: sarıgül anayasa_mahkemesi ne gidiyor
    Özet: Pegasus Havayolları, 12 milyar dolarlık yatırımla, Türk sivil havacılık tarihindeki en büyük sipariş olan 100 Airbus A320neo ailesi uçağını satın aldı. Başlık: pegasus tan 100 uçaklık sipariş
    Özet: KOBİ'lerin bilançolarının yetersizliği nedeniyle krediye erişimde yaşadıkları zorluklar ve bankaların yüksek karlılığına rağmen reel ekonomiye yeterince kaynak aktarmamaları, Türkiye'nin sanayi üretimini olumsuz etkiliyor. Başlık: kobi ler bankalardan yana dertli
    """

    if TASK_TYPE == "ozettenbasliga":
      prompt = threeshot + "Özet:"
      prompts = [prompt + row["summary"] + "Başlık:" for _, row in batch.iterrows()]

    elif TASK_TYPE == "basliktanozete":
      prompts = [threeshot + "Başlık:" + row["header"] + "Özet:" for _, row in batch.iterrows()]

    start_time = time.time()
    generated_contents = run_model_batch(prompts)
    end_time = time.time()

    batch_results = []
    for idx, (row, generated_content) in enumerate(zip(batch.itertuples(index=False), generated_contents)):

        if TASK_TYPE == "ozettenbasliga":

          real_header = row.header
          batch_results.append({
              "summary": row.summary,
              "real_header": row.header,
              "generated_header": generated_content,
              "model":MODEL_NAME,
              "time":(end_time-start_time)/batch_size,
              "prompt": prompts[idx]
          })

        elif TASK_TYPE == "basliktanozete":

          real_summary = row.summary
          batch_results.append({
              "baslik":row.header,
              "real_summary": row.summary,
              "generated_summary": generated_content,
              "model":MODEL_NAME,
              "time":(end_time-start_time)/batch_size,
              "prompt": prompts[idx]
          })

        print(f"[INFO] İşlendi: {idx + 1}/{len(batch)} | Başlık: {generated_content[:30]}...")
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
        results.extend(process_batch(batch, batch_size))  # Paralellik yerine sırayla işleniyor
        print(f"[INFO] Batch {i + 1}/{len(batches)} tamamlandı.")
    return results

if __name__ == "__main__":
    # Veri Yükleme
    file_path_550 = "/data/[evaluation]_550_headersandsummaries.xlsx"
    data_550 = pd.read_excel(file_path_550)

    start_time = time.time()
    batch_size = 32
    print("[INFO] İşlem başlıyor...")

    # Model Sonuçları
    model_results = parallel_process(data_550, batch_size=batch_size)

    print("[INFO] Tüm işlemler tamamlandı. Sonuçlar derleniyor...")
    results_df = pd.DataFrame(model_results)
    print(results_df)
    end_time = time.time()
    print("total_time:", end_time-start_time)

    # Buraya evaluate metodu eklenebilir
    results_df = evaluate(results_df, TASK_TYPE)

    # Excel'e Kaydetme

    if TASK_TYPE == "ozettenbasliga":
      output_path = "/senaryo2/task1_ozettenbasliga_tahminleri_son.xlsx"
    elif TASK_TYPE == "basliktanozete":
      output_path = "/senaryo2/task2_basliktanozete_tahminleri_son.xlsx"

    new_results = results_df

    if os.path.exists(output_path):
        existing_results = pd.read_excel(output_path)
        combined_results = pd.concat([existing_results, new_results], ignore_index=True)
    else:
        combined_results = new_results

    # Son veriyi dosyaya yaz
    combined_results.to_excel(output_path, index=False)
    print(f"[INFO] Sonuçlar {output_path} dosyasına kaydedildi.")
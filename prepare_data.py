
import pandas as pd
import requests
import re
import os
import time
from utils import clean_response_data, upload_data

from datasets import load_dataset

def generate_prompt(task_type, text):

  if task_type == "summary":
    prompt = """Aşağıdaki metni al ve özetle. Metnin içerisinde başlığı da bulunuyor.
                Anlamsal olarak düşündüğünde özetleyebileceksin.
                Özet kısa ve öz olmalı. En fazla 1 cümle olmalı. Haber metni ile kesinlikle aynı olmamalı.
                Sadece özeti döndür. Başka hiçbir şey döndürme. """ + f"""Metin: {text}"""

  elif task_type == "header":
    prompt = """Aşağıdaki metni al ve ilk kelimelerinden başlığını çıkar.
                Anlamsal olarak düşündüğünde başlığı metinin içerisinden ayırabileceksin.
                Başlık kesme işareti veya tırnak işareti içermemeli. Onların yerine boşluk koy.
                Başlık mutlaka metinin içinden çıkarılmalı, kendin oluşturma.
                Sadece başlığı döndür. Başka hiçbir şey döndürme.""" +  f"""Metin: {text}"""

  return prompt

def make_api_call(prompt, api_key):

  print("[*] making api call...")
  url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
  payload = {"contents": [{"parts": [{"text": prompt}]}]}
  headers = {"Content-Type": "application/json"}

  return requests.post(url, json=payload, headers=headers)

def connect_gemini(data, task_type, api_keys):

    results = []
    print("[*] connected gemini - making api calls...")
    number_of_keys = len(api_keys)

    for index, row in data.iterrows():
        print(f"[*] {index}th step -")
        prompt = generate_prompt(task_type,row["text"])
        response = make_api_call(prompt, api_keys[index%number_of_keys])

        if response.status_code == 200:
          print("[*] Response 200 ")
          response_content = clean_response_data(response.json()["candidates"][0]["content"]["parts"][0]["text"])
          print("Response content:",response_content)
          results.append({"text":row["text"],"baslik":response_content})

        else:
          print("ERROR! Code: ",response.status_code)
          if response.status_code == 429:
            time.sleep(15)
            response = make_api_call(prompt, api_keys[index%number_of_keys])
            results.append({"text":row["text"],"baslik":response_content})

    return results


if __name__ == "__main__":
    
    data_4900 = upload
    results = connect_gemini(data_4900, task_type="summary", api_keys=API_KEYS)
    # Sonuçları bir pandas DataFrame'e dönüştür ve Excel'e kaydet
    df = pd.DataFrame(results)
    output_file = "4900_generated_summaries.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")
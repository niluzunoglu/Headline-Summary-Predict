from datasets import load_dataset

def read_data(file_path):
    try:
        data = pd.read_excel(file_path)
        return data
    except Exception as e:
        print(e)
        return None

def upload_data(dataset_name):
    try:
        data = load_dataset(dataset_name)
        return data
    except Exception as e:
        print(e)
        return None

def clean_response_data(content):
    content = content.replace("json","").replace("\n","").replace("\"","'").strip()
    print("cleaned response content:", content)
    return content
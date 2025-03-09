import json

def extract_text_by_name(file_path, target_name):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    extracted_texts = [entry["text"] for entry in data["conversation"] if entry["name"] == target_name]
    return extracted_texts

json_file_path = "alex_mia_conversation.json" 
target_name = "Mia"

texts = extract_text_by_name(json_file_path, target_name)
for text in texts:
    print(text)

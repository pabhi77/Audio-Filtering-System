import os
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification

upload_directory = 'uploads'
modified_upload_directory = 'modified_upload'
# Paths for resources
model_path = 'model'
tokenizer_path = 'tokenizer'


# Make sure the output directory exists
os.makedirs(modified_upload_directory, exist_ok=True)

# Load the fine-tuned model and tokenizer
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertForSequenceClassification.from_pretrained(model_path)

def read_json_file(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def is_offensive(word):
    inputs = tokenizer.encode_plus(word, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(probs).item()
    return predicted_label == 1

def process_segments(segments):
    for segment in segments:
        for word in segment['words']:
            text = word['text'].lower()
            if is_offensive(text):
                word['is_offensive'] = 1
                word['text'] = "****"  # Replace the offensive word
            else:
                word['is_offensive'] = 0
    return segments

def write_json_file(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def main():
    # Process each file in the upload directory
    for filename in os.listdir(upload_directory):
        if filename.endswith('.json'):
            json_file_path = os.path.join(upload_directory, filename)
            output_json_file_path = os.path.join(modified_upload_directory, filename)

            original_data = read_json_file(json_file_path)

            modified_segments = process_segments(original_data['segments'])
            original_data['segments'] = modified_segments

            write_json_file(original_data, output_json_file_path)

            print(f"Processed {filename} and saved modified transcription in {modified_upload_directory}")

if __name__ == "__main__":
    main()

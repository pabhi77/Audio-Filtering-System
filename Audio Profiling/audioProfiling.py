import os
import json
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Assuming you have your model and tokenizer saved locally
model_path = "model"
tokenizer_path = "tokenizer"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Ensure the model is in evaluation mode
model.eval()

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def detect_offensive_words(segments):
    print("Offensive word detection:")
    offensive_words = []

    for segment in segments:
        for word in segment['words']:
            text = word['text']
            inputs = tokenizer.encode_plus(text, return_tensors='pt', padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)

            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(probs).item()

            if predicted_label == 1:  # Assuming 1 corresponds to the offensive class
                offensive_words.append(text)

    return offensive_words


def process_files(directory, output_directory):
    # Create the output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                data = json.load(file)

                offensive_words = detect_offensive_words(data['segments'])

                if offensive_words:
                    # Save the offensive words to a CSV file within the specified output directory
                    df = pd.DataFrame(offensive_words, columns=['Offensive Word'])
                    csv_filename = os.path.join(output_directory, filename.split('.')[0] + '.csv')
                    df.to_csv(csv_filename, index=False)
                    print(f"Offensive words from {filename} saved to {csv_filename}")


if __name__ == "__main__":
    trans_directory = "trans"
    output_directory = "offensive_word_list_files"
    process_files(trans_directory, output_directory)

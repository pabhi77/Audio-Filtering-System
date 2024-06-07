import os
import json
import pandas as pd

upload_directory = 'uploads'
modified_upload_directory = 'modified_upload'
output_excel_path = 'comparison_transcriptions.xlsx'


def read_json_file(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def extract_sentences(data):
    sentences = []
    for segment in data['segments']:
        sentence = ' '.join(word['text'] for word in segment['words'])
        sentences.append(sentence)
    return ' '.join(sentences)


def main():
    original_texts = []
    modified_texts = []

    # Ensure both directories have the same files and are sorted
    original_files = sorted(os.listdir(upload_directory))
    modified_files = sorted(os.listdir(modified_upload_directory))

    for filename in original_files:
        if filename.endswith('.json'):
            # Paths for the original and modified JSON files
            original_file_path = os.path.join(upload_directory, filename)
            modified_file_path = os.path.join(modified_upload_directory, filename)

            # Reading files and extracting sentences
            original_data = read_json_file(original_file_path)
            modified_data = read_json_file(modified_file_path)

            original_sentence = extract_sentences(original_data)
            modified_sentence = extract_sentences(modified_data)

            original_texts.append(original_sentence)
            modified_texts.append(modified_sentence)

    # Create a DataFrame and save to Excel
    df = pd.DataFrame({
        'Original Transcription': original_texts,
        'Modified Transcription': modified_texts
    })

    df.to_excel(output_excel_path, index=False)
    print(f"Excel file has been saved to {output_excel_path}")


if __name__ == "__main__":
    main()

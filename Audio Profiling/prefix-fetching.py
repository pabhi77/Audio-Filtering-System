import os
import json
import pandas as pd

modified_upload_directory = 'modified_upload'
output_excel_path = 'prefixes_of_offensive_words.xlsx'

def read_json_file(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def extract_prefixes(data):
    prefixes = []
    for segment in data['segments']:
        for i in range(1, len(segment['words'])):
            # Check if the current word is offensive
            if segment['words'][i]['is_offensive'] == 1:
                # Append the previous word's text to the list
                prefixes.append(segment['words'][i - 1]['text'])
    return prefixes

def main():
    all_prefixes = []
    filenames = sorted(os.listdir(modified_upload_directory))

    for filename in filenames:
        if filename.endswith('.json'):
            modified_file_path = os.path.join(modified_upload_directory, filename)
            modified_data = read_json_file(modified_file_path)
            prefixes = extract_prefixes(modified_data)
            all_prefixes.extend(prefixes)

    # Create a DataFrame and save to Excel
    df = pd.DataFrame({
        'Prefixes': all_prefixes
    })

    df.to_excel(output_excel_path, index=False)
    print(f"Excel file with prefixes has been saved to {output_excel_path}")

if __name__ == "__main__":
    main()

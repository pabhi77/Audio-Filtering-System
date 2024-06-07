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

    # Create a DataFrame for prefixes
    prefix_df = pd.DataFrame({
        'Prefixes': all_prefixes
    })

    # Calculate the prefix counts and percentages
    prefix_count = pd.Series(all_prefixes).value_counts()
    total_prefixes = prefix_count.sum()
    percentages = (prefix_count / total_prefixes) * 100  # Calculate percentage

    # New DataFrame with percentage data
    percentage_df = pd.DataFrame({
        'Prefix Word': prefix_count.index,
        'Percentage': percentages
    })

    # Read existing Excel if it exists or initialize it if not
    if os.path.exists(output_excel_path):
        existing_df = pd.read_excel(output_excel_path)
        combined_df = pd.concat([existing_df, percentage_df], axis=1)
    else:
        combined_df = percentage_df

    # Save updated DataFrame to Excel
    combined_df.to_excel(output_excel_path, index=False)
    print(f"Excel file has been updated and saved to {output_excel_path}")

if __name__ == "__main__":
    main()

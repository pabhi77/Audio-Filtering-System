import pandas as pd
import os

# Directory containing the CSV files
input_directory = "offensive_word_list_files"
# Output Excel file
output_excel_file = "offensive_words_analysis.xlsx"


def classify_offensiveness(count):
    if count < 2:
        return "Less Offensive"
    elif 2 <= count <= 4:
        return "Medium"
    else:
        return "Highly Offensive"


# List to hold the data for the final DataFrame
data = []

# Iterate over each file in the directory
for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_directory, filename)
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Number of offensive words
        num_offensive_words = len(df)

        # Level of offense based on the number of offensive words
        level_of_offense = classify_offensiveness(num_offensive_words)

        # Offensive words in the file
        offensive_words = ", ".join(df['Offensive Word'])

        # Append the information to the data list
        data.append([filename, num_offensive_words, level_of_offense, offensive_words])

# Create a DataFrame with the compiled data
analysis_df = pd.DataFrame(data, columns=['File Name', 'Number of Offensive Words', 'Level of Offense',
                                          'Offensive Words in the File'])

# Save the DataFrame to an Excel file
analysis_df.to_excel(output_excel_file, index=False)

print(f"Analysis completed. Results are saved in '{output_excel_file}'")

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Path to the Excel file generated previously
excel_file = "offensive_words_analysis.xlsx"

# Read the Excel file
df = pd.read_excel(excel_file)

# Combine all the offensive words into a single list
all_offensive_words = []
for words in df['Offensive Words in the File'].dropna():
    all_offensive_words.extend(words.split(', '))

# Count the frequency of each offensive word
word_counts = Counter(all_offensive_words)

# Convert the counter object to a DataFrame for easier plotting
words_df = pd.DataFrame(word_counts.items(), columns=['Offensive Word', 'Frequency']).sort_values(by='Frequency', ascending=False)

# Plotting the distribution of the level of offense
plt.figure(figsize=(10, 6))
df['Level of Offense'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Level of Offense Across Files')
plt.xlabel('Level of Offense')
plt.ylabel('Number of Files')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('level_of_offense_distribution.png')
plt.show()

# Plotting the pie chart for proportion of offense levels
plt.figure(figsize=(8, 8))
df['Level of Offense'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['tomato', 'gold', 'lightgreen'])
plt.title('Proportion of Offense Levels')
plt.ylabel('')  # Hide the y-label as it's unnecessary for pie charts
plt.tight_layout()
plt.savefig('offense_level_proportion.png')
plt.show()

# Plotting the frequency of each offensive word
plt.figure(figsize=(12, 8))
plt.bar(words_df['Offensive Word'], words_df['Frequency'], color='purple')
plt.title('Frequency of Each Offensive Word')
plt.xlabel('Offensive Words')
plt.ylabel('Frequency')
plt.xticks(rotation=90)  # Rotate the x-axis labels to make them readable
plt.tight_layout()
plt.savefig('offensive_word_frequency.png')
plt.show()

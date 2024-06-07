import pandas as pd
import matplotlib.pyplot as plt

# Path to the Excel file generated previously
excel_file = "offensive_words_analysis.xlsx"

# Read the Excel file
df = pd.read_excel(excel_file)

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

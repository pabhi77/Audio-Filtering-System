from flask import Flask, request, render_template, redirect, url_for
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
from collections import Counter
import json
import shutil
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['STATIC_FOLDER'] = 'static'
app.config['ALLOWED_EXTENSIONS'] = {'json'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit uploads to 16MB per file

model_path = "model"
tokenizer_path = "tokenizer"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

devanagari_font_path = 'NotoSansDevanagari.ttf'
devanagari_font = FontProperties(fname=devanagari_font_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    files = request.files.getlist('file')
    if not files:
        return redirect(request.url)
    for file in files:
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
    process_files(app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER'])
    return redirect(url_for('results'))

@app.route('/results')
def results():
    output_excel_file = os.path.join(app.config['PROCESSED_FOLDER'], "offensive_words_analysis.xlsx")
    compile_results(app.config['PROCESSED_FOLDER'], output_excel_file)
    df = pd.read_excel(output_excel_file)

    # Convert DataFrame to list of dictionaries for easier template processing
    data = df.to_dict(orient='records')
    visualize_data(output_excel_file)
    return render_template('results.html', data=data,images=['level_of_offense_distribution.png', 'offense_level_proportion.png', 'offensive_word_frequency.png'])



def visualize_data(output_excel_file):
    df = pd.read_excel(output_excel_file)

    # Combine and count offensive words
    all_offensive_words = [word for words in df['Offensive Words in the File'].dropna() for word in words.split(', ')]
    word_counts = Counter(all_offensive_words)
    words_df = pd.DataFrame(word_counts.items(), columns=['Offensive Word', 'Frequency']).sort_values(by='Frequency', ascending=False)

    # Visualize data
    plot_distribution(df, 'level_of_offense_distribution.png')
    plot_pie_chart(df, 'offense_level_proportion.png')
    plot_frequency(words_df, 'offensive_word_frequency.png')


def plot_distribution(df, filename):
    plt.figure(figsize=(10, 6))
    ax = df['Level of Offense'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Bar chart')
    plt.xlabel('Level of Offense')
    plt.ylabel('Number of Files')
    plt.xticks(rotation=0)

    # Ensure y-axis has integer ticks
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(app.config['STATIC_FOLDER'], filename))
    plt.close()

def plot_pie_chart(df, filename):
    plt.figure(figsize=(8, 8))
    df['Level of Offense'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['tomato', 'gold', 'lightgreen'])
    plt.title('Proportion of Offense Levels')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(app.config['STATIC_FOLDER'], filename))
    plt.close()

def plot_frequency(words_df, filename):
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.sans-serif'] = ['Nirmala UI']
    plt.bar(words_df['Offensive Word'], words_df['Frequency'], color='purple')
    plt.title('Frequency of Each Offensive Word', fontname='Nirmala UI')
    plt.xlabel('Offensive Words', fontname='Nirmala UI')
    plt.ylabel('Frequency', fontname='Nirmala UI')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(app.config['STATIC_FOLDER'], filename))
    plt.close()


def save_plot(figure, filename):
    output_path = os.path.join(app.config['STATIC_FOLDER'], filename)
    figure.savefig(output_path)
    plt.close(figure)

    # Move file to static directory
    static_path = os.path.join('static', filename)
    shutil.move(output_path, static_path)
def compile_results(input_directory, output_excel_file):
    data = []
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_directory, filename)
            df = pd.read_csv(file_path)
            num_offensive_words = len(df)
            level_of_offense = classify_offensiveness(num_offensive_words)
            offensive_words = ", ".join(df['Offensive Word'])
            data.append([filename, num_offensive_words, level_of_offense, offensive_words])
    analysis_df = pd.DataFrame(data, columns=['File Name', 'Number of Offensive Words', 'Level of Offense', 'Offensive Words in the File'])
    analysis_df.to_excel(output_excel_file, index=False)
    print(f"Analysis completed. Results are saved in '{output_excel_file}'.")

def classify_offensiveness(count):
    if count < 2:
        return "Less"
    elif 2 <= count <= 4:
        return "Medium"
    else:
        return "High"

def detect_offensive_words(segments):
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
            if predicted_label == 1:  # Assuming 1 corresponds to offensive
                offensive_words.append(text)
    return offensive_words

def process_files(directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                data = json.load(file)
                offensive_words = detect_offensive_words(data['segments'])
                if offensive_words:
                    df = pd.DataFrame(offensive_words, columns=['Offensive Word'])
                    csv_filename = os.path.join(output_directory, filename.split('.')[0] + '.csv')
                    df.to_csv(csv_filename, index=False)
    return "Processing completed."

if __name__ == '__main__':
    app.run(debug=True)

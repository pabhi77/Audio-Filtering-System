import speech_recognition as sr
import nltk
import string
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from pydub import AudioSegment
from gtts import gTTS
import os

# Load the fine-tuned model and tokenizer
model_path = 'model'
tokenizer_path = 'tokenizer'

tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Download the Punkt tokenizer for Marathi
nltk.download('punkt')

# Path to the beep sound file (1-second long beep)
beep_sound_path = 'beep.mp3'
beep_sound = AudioSegment.from_mp3(beep_sound_path)


def tokenize_marathi_sentence(sentence):
    words = nltk.word_tokenize(sentence)
    words = [word for word in words if word not in string.punctuation]
    return words


def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data, language="mr-IN")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Speech recognition request failed: {e}")
        return None


def censor_offensive_words(text):
    marathi_words = tokenize_marathi_sentence(text)
    censored_text = text
    for word in marathi_words:
        inputs = tokenizer.encode_plus(word, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(probs).item()

        if predicted_label == 1:  # Assuming 1 corresponds to the offensive class
            censored_text = censored_text.replace(word, "[BEEP]", 1)

    return censored_text


def generate_censored_audio(censored_text, output_file_path):
    tts = gTTS(censored_text.replace("[BEEP]", " "), lang='mr')
    tts.save("temp_audio.mp3")
    audio = AudioSegment.from_mp3("temp_audio.mp3")

    output_audio = AudioSegment.empty()
    start = 0
    for word in censored_text.split():
        if word == "[BEEP]":
            output_audio += beep_sound
        else:
            word_duration = len(word) / len(censored_text.replace("[BEEP]", " ")) * len(audio)
            end = start + word_duration
            output_audio += audio[start:end]
            start = end

    output_audio.export(output_file_path, format="mp3")
    os.remove("temp_audio.mp3")


if __name__ == "__main__":
    input_audio_file = "voice.wav"  # Specify your input audio file path
    output_audio_file = "audio.mp3"  # Specify your output audio file path

    text = transcribe_audio(input_audio_file)
    if text:
        censored_text = censor_offensive_words(text)
        generate_censored_audio(censored_text, output_audio_file)
        print("Censored audio file has been generated.")

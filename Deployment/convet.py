import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from pydub import AudioSegment

# Load the fine-tuned model and tokenizer
model_path = 'model'
tokenizer_path = 'tokenizer'

tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Load the beep sound
beep_sound_path = 'beep.mp3'
beep_sound = AudioSegment.from_mp3(beep_sound_path)


def read_transcribed_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data['segments']


def detect_offensive_words(segments):
    offensive_segments = []
    for segment in segments:
        for word in segment['words']:
            text = word['text']
            inputs = tokenizer.encode_plus(text, return_tensors='pt', padding=True, truncation=True)
            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)

            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(probs).item()

            if predicted_label == 1:  # Assuming 1 corresponds to the offensive class
                offensive_segments.append((word['start'] * 1000, word['end'] * 1000))  # Convert to milliseconds

    return offensive_segments


def overlay_beeps_on_audio(input_audio_path, offensive_segments, output_audio_path):
    original_audio = AudioSegment.from_file(input_audio_path)

    for start_ms, end_ms in offensive_segments:
        original_audio = original_audio[:start_ms] + beep_sound + original_audio[end_ms:]

    original_audio.export(output_audio_path, format="mp3")
    print("Censored audio file has been generated.")


if __name__ == "__main__":
    json_file_path = 'transcribed_audio.json'
    input_audio_path = 'voice.wav'
    output_audio_path = 'audio.mp3'

    segments = read_transcribed_json(json_file_path)
    offensive_segments = detect_offensive_words(segments)
    overlay_beeps_on_audio(input_audio_path, offensive_segments, output_audio_path)

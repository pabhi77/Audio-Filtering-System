import os
from pydub import AudioSegment
import json

# Paths for directories and beep sound
input_audio_directory = 'input_audio_files'
modified_upload_directory = 'modified_upload'
output_audio_directory = 'output_audio_files'
beep_sound_path = 'beep.mp3'

# Ensure output directory exists
os.makedirs(output_audio_directory, exist_ok=True)

# Load beep sound
beep_sound = AudioSegment.from_mp3(beep_sound_path)

def read_json_file(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def process_audio(original_audio, segments):
    modified_audio = original_audio
    for segment in segments:
        for word in segment['words']:
            if word['is_offensive']:
                start_ms = int(word['start'] * 1000)
                end_ms = int(word['end'] * 1000)
                modified_audio = modified_audio.overlay(beep_sound, position=start_ms, gain_during_overlay=-100)
    return modified_audio

def main():
    print("Starting processing...")
    for json_filename in os.listdir(modified_upload_directory):
        if json_filename.endswith('.json'):
            base_filename = json_filename.replace('.opus--edited.json', '')
            audio_filename = f"{base_filename}.wav"  # Adjust to WAV format
            audio_file_path = os.path.join(input_audio_directory, audio_filename)
            json_file_path = os.path.join(modified_upload_directory, json_filename)
            output_audio_path = os.path.join(output_audio_directory, f"{base_filename}_censored.wav")  # Output as WAV

            print(f"Processing {audio_filename}...")

            if os.path.exists(audio_file_path):
                print(f"Loading audio from {audio_file_path}")
                original_audio = AudioSegment.from_file(audio_file_path, format="wav")
                transcription_data = read_json_file(json_file_path)
                modified_audio = process_audio(original_audio, transcription_data['segments'])
                modified_audio.export(output_audio_path, format="wav")  # Export as WAV
                print(f"Censored audio file has been generated: {output_audio_path}")
            else:
                print(f"Audio file does not exist for {json_filename}, skipping...")

if __name__ == "__main__":
    main()

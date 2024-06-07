## Audio Filtering System Project

### Introduction
The "Audio Filtering System" project addresses the challenge of moderating and filtering offensive or unwanted words within audio streams using advanced speech recognition and audio processing techniques. It aims to create a positive audio environment across various communication platforms by identifying and removing undesired content in real-time.

### Overview and Problem Statement
The system processes audio input, converts speech to text, and identifies offensive words from a predefined blacklist. These words are then replaced with beep sounds, providing uninterrupted, moderated audio content.

### Motivation
The project is motivated by the increasing demand for respectful and safe communication platforms. It leverages advanced technology to foster positive and inclusive interactions, promoting constructive dialogue across diverse platforms.

### Key Features
1. **Accessibility**: Designed to be user-friendly and accessible to individuals of all technical backgrounds.
2. **Timesaving**: Automates the content moderation process, saving users from manual intervention.
3. **Customization**: Allows users to define their own blacklist of unwanted words, tailoring the system to their specific needs.

### Objectives
1. **Develop a Classification Model**: Create a machine learning model to identify offensive words.
2. **Word Replacement or Censoring**: Replace offensive words with beeps or softer alternatives.
3. **Audio File Generation**: Modify audio files to reflect the censored content.
4. **Predictive Censoring**: Estimate the probability of offensive words before they are spoken for proactive moderation.

### Application Areas
1. **Media and Entertainment**: Censor offensive language in live and recorded content.
2. **Customer Service and Call Centers**: Maintain professional communication standards.
3. **Educational Platforms**: Ensure safe and respectful learning environments.
4. **Social Media**: Efficiently moderate live and recorded content.
5. **Public Safety**: Monitor audio for aggressive or threatening speech.

### System Analysis and Design
**Technology Stack**:
- Python
- SpeechRecognition Library
- Pydub
- PyCharm
- TensorFlow
- PyTorch
- Flask

**Hardware Requirements**:
- Microphone
- Speakers or Headphones

### Implementation
1. **Training the Model**: Use BERT for offensive word detection in Marathi text, fine-tuning it for accuracy.
2. **Word Replacement**: Replace detected offensive words in audio with beeps or alternatives.
3. **Audio Modification**: Adjust the original audio to reflect the censored content.
4. **Predictive Analysis**: Analyze prefixes in transcribed audio data to understand patterns leading to offensive language.

### Conclusion
The Audio Filtering System project provides an innovative and comprehensive solution for moderating offensive content in audio streams, promoting a safer and more respectful communication environment.




![image](https://github.com/pratikr10/Real-Time-Audio-Filter/assets/109615455/d09df4fb-fd18-4b93-afdb-9e9b31ce4721) 

Table 1. Sample data

![image](https://github.com/pratikr10/Real-Time-Audio-Filter/assets/109615455/a2352b5d-0d2f-4630-b891-f0d66bcd6fe2)

Fig. 1. Accuracy of model.

![image](https://github.com/pratikr10/Real-Time-Audio-Filter/assets/109615455/69d079bc-bf5e-4645-9e0a-40d71042cf6f)

Screenshort while running system locally.

![image](https://github.com/pratikr10/Real-Time-Audio-Filter/assets/109615455/8be71d96-0d55-48e1-87bc-e8dd1c7645f8)
table 2. Less offensive word substitution.

![image](https://github.com/pratikr10/Real-Time-Audio-Filter/assets/109615455/c6e51c6f-a88f-4836-b00a-93578822d643)

Fig. 2. Distribution of Prefix words followed by Offensive words 


# Real-Time-Audio-Filter
Developed a system that could detect and censor offensive words in real-time during live conversations. I was motivated to work on this due to the increasing need for respectful and safe communication in digital interactions.

The primary objective was to create a system that could automatically detect offensive language and replace it with a beep sound during live audio streams, ensuring a more respectful conversation environment.

To achieve this, I fine-tuned a pre-trained BERT model for offensive word detection. I chose BERT for its effectiveness in understanding the context in language. The project was developed using TensorFlow, PyTorch, and Python, with PyCharm as the IDE.

I manually created a dataset of offensive words of marathi language and trained the model on this dataset. The training process was challenging but ultimately successful, achieving an accuracy of 96.56%.

The system successfully detects and censors offensive words in real-time, which can be particularly useful in various digital communication platforms.

Screenshort while running system locally.

![image](https://github.com/pratikr10/Real-Time-Audio-Filter/assets/109615455/69d079bc-bf5e-4645-9e0a-40d71042cf6f)

You can create your own regional language dataset and train the model on that dataset.

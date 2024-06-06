# Real-Time-Audio-Filter
Developed a system that could detect and censor offensive words in real-time during live conversations. I was motivated to work on this due to the increasing need for respectful and safe communication in digital interactions.

The primary objective was to create a system that could automatically detect offensive language and replace it with a beep sound during live audio streams, ensuring a more respectful conversation environment.

To achieve this, I fine-tuned a pre-trained BERT model for offensive word detection. I chose BERT for its effectiveness in understanding the context in language. The project was developed using TensorFlow, PyTorch, and Python, with PyCharm as the IDE.

I manually created a dataset of offensive words of marathi language and trained the model on this dataset. The training process was challenging but ultimately successful, achieving an accuracy of 96.56%.

The system successfully detects and censors offensive words in real-time, which can be particularly useful in various digital communication platforms.


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


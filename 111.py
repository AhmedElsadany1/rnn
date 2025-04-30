import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

text = "the cat sits on the mat and the dog barks at the moon"

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
word_index = tokenizer.word_index
total_words = len(word_index)

sequences = []
words = text.split()
for i in range(len(words) - 3):
    sequences.append(words[i:i+4])

encoded_sequences = []
for seq in sequences:
    encoded_sequences.append([word_index[word] for word in seq])

encoded_sequences = np.array(encoded_sequences)

X = encoded_sequences[:, :-1]  
y = encoded_sequences[:, -1]   

model = Sequential([
    Embedding(input_dim=total_words + 1, output_dim=10, input_length=3),
    SimpleRNN(50, return_sequences=False),
    Dense(total_words + 1, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=200, verbose=2)

def predict_next_word(input_text, model, tokenizer):
    input_seq = tokenizer.texts_to_sequences([input_text])[0]
    input_seq = pad_sequences([input_seq], maxlen=3, truncating='pre')
    predicted_word_index = np.argmax(model.predict(input_seq), axis=-1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

input_text = "the cat sits"
predicted_word = predict_next_word(input_text, model, tokenizer)
print(f"The predicted 4th word is: {predicted_word}")

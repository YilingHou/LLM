import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.stats import entropy
from tensorflow.keras.callbacks import EarlyStopping
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.corpus import brown
nltk.download('brown')
nltk.download('stopwords')
# Sample text from the Brown corpus
text = brown.words(categories='news')[:1000] # Using first 1000 words of Brown corpus
# Lowercase all letters in each word
text_lower = [word.lower() for word in text]
# Combine all words into a single string
text = ' '.join(text_lower)

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1
#print(f'total words: {total_words}') gives 416

# Create input sequences
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Split input sequences into predictors and labels
X = input_sequences[:, :-1]
y = input_sequences[:, -1]

# Convert labels to one-hot encoding
from keras.utils import to_categorical
y = to_categorical(y, num_classes=total_words)

# Define the model
embedding_dim = 10
model = Sequential()
model.add(Embedding(total_words, embedding_dim, input_length=max_sequence_len-1))
model.add(LSTM(50))
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
for epoch in range(500):  # Training for 500 epochs
    print(f'Epoch {epoch+1}/500')
    # Prevent overfitting with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # Train the model with early stopping
    history = model.fit(X, y, epochs=1, batch_size=64, verbose=1, validation_split=0.2, callbacks=[early_stopping])  # Train for 1 epoch
    
    # Print loss and accuracy
    loss = history.history['loss'][0]
    accuracy = history.history['accuracy'][0]
    print(f'Loss: {loss}, Accuracy: {accuracy}')

# Save the model
model.save('my_model.keras')

# Quantitative Evaluation
# Make predictions on the validation dataset
y_pred_probs = model.predict(X)
# Calculate entropy for each predicted probability distribution
entropies = [entropy(pred_probs) for pred_probs in y_pred_probs]
# Calculate the average entropy across all samples
average_entropy = np.mean(entropies)
print(f'Average Entropy on Validation Dataset: {average_entropy}')


# Split the text into words
words = text.split()
seed_text = ' '.join(words[:30])
print(f'seed:{seed_text}')
# Tokenize the seed text
seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
# Define the maximum number of words to generate
max_words = 20
# Generate words using the model
generated_words = []
for _ in range(max_words):
    # Pad the seed sequence
    padded_sequence = pad_sequences([seed_sequence], maxlen=max_sequence_len-1, padding='pre')
    # Predict the next word
    predicted_probs = model.predict(padded_sequence)[0]
    predicted_word_index = np.argmax(predicted_probs)
    predicted_word = tokenizer.index_word[predicted_word_index]
    # Append the predicted word to the seed sequence
    seed_sequence.append(predicted_word_index)
    # Limit the seed sequence to the maximum length
    seed_sequence = seed_sequence[-(max_sequence_len-1):]
    # Add the predicted word to the generated words list
    generated_words.append(predicted_word)

# Join the generated words into a sentence
generated_sentence = ' '.join(generated_words)

# Print the generated sentence
print("Generated Sentence:", generated_sentence)

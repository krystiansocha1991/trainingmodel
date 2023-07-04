# trainingmodel
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.models import load_model, save_model
# Ścieżka do folderu z artykułami
articles_folder = ""
# Ścieżka do zapisanego modelu
model_path = ''
# Wczytanie modelu
model = load_model(model_path)
# Inicjalizacja tokenizer'a
tokenizer = keras.preprocessing.text.Tokenizer()

# Wczytanie zawartości artykułów
articles = []
for filename in os.listdir(articles_folder):
    filepath = os.path.join(articles_folder, filename)
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
        articles.append(content)

# Tokenizacja tekstu
tokenizer.fit_on_texts(articles)
vocab_size = len(tokenizer.word_index) + 1
print("vocab_size:", vocab_size)
# Usunięcie stop words
stop_words = set(stopwords.words(""))
filtered_articles = []
for article in articles:
    tokens = word_tokenize(article)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    filtered_article = " ".join(filtered_tokens)
    filtered_articles.append(filtered_article)

# Podział tekstu na sekwencje
input_sequences = tokenizer.texts_to_sequences(filtered_articles)
output_sequences = [seq[1:] for seq in input_sequences]
input_sequences = [seq[:-1] for seq in input_sequences]

# Ograniczenie długości sekwencji
max_length = 512  # Długość docelowa sekwencji
input_sequences = pad_sequences(input_sequences, maxlen=max_length, truncating='post')
output_sequences = pad_sequences(output_sequences, maxlen=max_length, truncating='post')

# Przygotowanie danych treningowych
X = np.array(input_sequences)
y = np.array(output_sequences)

# Sprawdzenie rozmiarów sekwencji
if X.shape[0] == y.shape[0]:
    print("Rozmiary sekwencji wejściowych i wyjściowych po skalowaniu są takie same.")
else:
    print("Rozmiary sekwencji wejściowych i wyjściowych po skalowaniu są różne.")

logits_shape = X.shape
labels_shape = y.shape

print("Kształt logitów:", logits_shape)
print("Kształt etykiet:", labels_shape)

# Definicja modelu
input_layer = Input(shape=(max_length,))
embedding_layer = Embedding(vocab_size, 100)(input_layer)
lstm_layer = LSTM(128)(embedding_layer)
output_layer = Dense(512, activation="softmax")(lstm_layer)
model = Model(input_layer, output_layer)

# Kompilacja modelu
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Callback do zapisu modelu
checkpoint = ModelCheckpoint("/home/krystian/architektura/modele/model.h5", monitor="loss", save_best_only=True)

# Trening modelu
batch_size = 1
num_epochs = 100
for epoch in range(num_epochs):
    # Zmiana rozmiaru wsadu (batch size) co epokę
    if epoch > 0 and epoch % 2 == 0:
        batch_size *= 2
    history = model.fit(X, y, epochs=1, batch_size=batch_size, callbacks=[checkpoint])
    print(f"Epoch {epoch+1} - Loss: {history.history['loss'][0]} - Accuracy: {history.history['accuracy'][0]}")

# Zapisanie wytrenowanego modelu
model.save("")

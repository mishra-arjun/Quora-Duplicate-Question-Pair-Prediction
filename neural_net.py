import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Merge
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text
from keras.layers import SpatialDropout1D

data = pd.read_csv('train.csv')
y = data.is_duplicate.values

tk = text.Tokenizer(num_words=200000)

max_len = 40
tk.fit_on_texts(list(data.question1.values) + list(data.question2.values.astype(str)))
x1 = tk.texts_to_sequences(data.question1.values)
x1 = sequence.pad_sequences(x1, maxlen=max_len)

x2 = tk.texts_to_sequences(data.question2.values.astype(str))
x2 = sequence.pad_sequences(x2, maxlen=max_len)

# Dividing the datasets into train and test splits
from sklearn.model_selection import train_test_split

x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size=0.2)
x2_train, x2_test, _, _ = train_test_split(x2, y, test_size=0.2)


word_index = tk.word_index

ytrain_enc = np_utils.to_categorical(y) # Do not know why this is defined - not used anywhere
# We might need to do this on our y dataset

embeddings_index = {}
f = open('glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

max_features = 200000
filter_length = 5
nb_filter = 64
pool_length = 4

model = Sequential()
print('Build model...')

# Building convolutional models
model1 = Sequential()
model1.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))
model1.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
model1.add(Dropout(0.2))

model1.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))

model1.add(GlobalMaxPooling1D())
model1.add(Dropout(0.2))

model1.add(Dense(300))
model1.add(Dropout(0.2))
model1.add(BatchNormalization())


model2 = Sequential()
model2.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))
model2.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))
model2.add(Dropout(0.2))

model2.add(Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1))

model2.add(GlobalMaxPooling1D())
model2.add(Dropout(0.2))

model2.add(Dense(300))
model2.add(Dropout(0.2))
model2.add(BatchNormalization())

# Building the LSTM models
model3 = Sequential()
model3.add(Embedding(len(word_index) + 1, 300, input_length=40))
model3.add(SpatialDropout1D(0.2))
model3.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))


model4 = Sequential()
model4.add(Embedding(len(word_index) + 1, 300, input_length=40))
model4.add(SpatialDropout1D(0.2))
model4.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))

# Merging the 4 models
merged_model = Sequential()
merged_model.add(Merge([model1, model2, model3, model4], mode='concat'))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())


merged_model.add(Dense(1))
merged_model.add(Activation('sigmoid'))

merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


merged_model.fit([x1_train, x2_train, x1_train, x2_train], y=y_train, batch_size=384, epochs=50,
                 verbose=1, validation_split=0.1, shuffle=True)

# Saving the model so that it can be used later on
merged_model.save("merged_model.h5")
merged_model.save_weights('merged_model1.h5')

# Using the model to predict on the test set
y_pred = merged_model.predict([x1_test, x2_test, x1_test, x2_test])

# Making the confusion matrix and getting the accuracy score for our predictions
from sklearn.metrics import confusion_matrix, accuracy_score

print("The confusion matrix for prediction:\n", confusion_matrix(y_test, y_pred))
print("The accuracy score for prediction:\n", accuracy_score(y_test, y_pred))

# This architecture of the model gives us an accuracy of 0.838.
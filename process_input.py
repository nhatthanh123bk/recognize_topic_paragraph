import pandas as pd 
import numpy as np 
import keras
import os
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import SimpleRNN, Embedding, Dense, LSTM
from keras.models import Sequential
#read_data
path1 = "Train_Full//"
text_train = []
topic_train = []
list_file = os.listdir(path1)
print(list_file)
i = 0
dem = 0
for j in list_file:
	list_file1 = os.listdir(path1+j+'//')
	for k in list_file1:
		f = open(path1+ j +'//'+k,'r',encoding='utf-16')
		str = f.read()
		if dem>499:
			break
		else:
			dem = dem + 1
		text_train.append(str)
		topic_train.append(i)
	i = i+1
	dem = 0

topic_train = np.asarray(topic_train)



training_samples = int(5000 * .8)
validation_samples = int(5000 - training_samples)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_train)
sequences = tokenizer.texts_to_sequences(text_train)
word_index = tokenizer.word_index
data_train = pad_sequences(sequences,1000)


np.random.seed(42)
# shuffle data
indices = np.arange(data_train.shape[0])
np.random.shuffle(indices)
data_train = data_train[indices]
topic_train = topic_train[indices]

texts_train = data_train[:training_samples]
y_train = topic_train[:training_samples]
texts_test = data_train[training_samples:]
y_test = topic_train[training_samples:]


model = Sequential()
model.add(Embedding(len(word_index) + 1, 64))
model.add(LSTM(64))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
history_lstm = model.fit(texts_train, y_train, epochs=10, batch_size=60, validation_split=0.2)

#predict
score = model.evaluate(texts_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


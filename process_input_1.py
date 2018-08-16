import pandas as pd
import numpy as np
import keras
import os
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import SimpleRNN, Embedding, Dense, LSTM
from keras.models import Sequential
from keras.utils import to_categorical

#read_data
path1 = "Train_Full//"
text_train = []
topic_train = []
list_file = os.listdir(path1)
print("danh sach cac chu de:\n")
i = 0
dem = 0
for j in list_file:
	list_file1 = os.listdir(path1+j+'//')
	for k in list_file1:
		f = open(path1+ j +'//'+k,'r',encoding='utf-16')
		str = f.read()
		text_train.append(str)
		topic_train.append(i)
	i = i+1
	print(i-1,": ",j)
	print("\n")

topic_train = np.asarray(topic_train)
topic_train = to_categorical(topic_train)
training_samples = int(len(text_train) * .8)
validation_samples = int(len(text_train) - training_samples)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_train)
sequences = tokenizer.texts_to_sequences(text_train)
word_index = tokenizer.word_index
data_train = pad_sequences(sequences,300)

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
model.add(Embedding(input_dim = len(word_index) + 1, output_dim = 100, input_length = 300))
model.add(LSTM(64))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.summary()
history_lstm = model.fit(texts_train, y_train, epochs=10, batch_size=60, validation_split=0.2)
#predict

pred = model.predict_classes(texts_test)
acc = model.evaluate(texts_test, y_test)
proba_ltsm = model.predict_proba(texts_test)
from sklearn.metrics import confusion_matrix
print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc[0],acc[1]))

path_sample = "//home//thanh//Desktop//AI//recognize_topic//Train_Full//Chinh tri Xa hoi//XH_NLD_ (3673).txt"
f = open(path_sample,'r',encoding='utf-16')
str = f.read()
Xtext = [str]
Xtext = tokenizer.texts_to_sequences(Xtext)
Xtext = pad_sequences(Xtext,300)
pred = model.predict_classes(Xtext)
print("doan van thuoc linh vuc:",pred)

#https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/
import numpy as np
from numpy import array
import os; os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding

import sys

'''def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
	# generate a fixed number of words
    
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# pre-pad sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
	return in_text'''

def generate(model, x, n_words):
    res = x
    for _ in range(n_words):
        padded = pad_sequences([res], maxlen=2, padding='pre')
        res += [ model.predict_classes(padded)[0] ]
    return res

data = open('big.txt').read()[:500000]
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]
vocab_size = len(tokenizer.word_index) + 1
sequences = list()
for i in range(2, len(encoded)):
	sequence = encoded[i-2:i+1]
	sequences.append(sequence)
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)

# Download glove.6B.100d.txt (Wikipedia, 100d) from https://nlp.stanford.edu/projects/glove/
embeddings_index = {}
f = open(os.path.join('./', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

max_words = vocab_size
word_index = tokenizer.word_index
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
 
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length-1))
model.add(Flatten())
model.add(Dense(embedding_dim, activation='relu')) # 32
model.add(Dense(vocab_size, activation='softmax'))

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('# fit network', file=sys.stderr)
model.fit(X, y, epochs=20, verbose=2)

revIndex = dict( [ (value, key) for key, value in tokenizer.word_index.items() ])

def view(index): return ' '.join( [revIndex.get(i, '?') for i in index])

print('# evaluate model\n', 'I want:', view( generate(model, tokenizer.texts_to_sequences(['I want'])[0], 5) ))

'''print(generate_seq(model, tokenizer, max_length-1, 'Jack and', 5))
print(generate_seq(model, tokenizer, max_length-1, 'And Jill', 3))
print(generate_seq(model, tokenizer, max_length-1, 'fell down', 5))
print(generate_seq(model, tokenizer, max_length-1, 'pail of', 5))'''


''' model.add(Embedding(vocab_size, 10, input_length=max_length-1))
    model.add(LSTM(50))
    model.add(Dense(vocab_size, activation='softmax'))'''

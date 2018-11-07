#https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/
from numpy import array
import os; os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding

import sys

'''def generate_seq(model, tokenizer, revIndex, inwords, n_words):
	for _ in range(n_words):
		encoded = tokenizer.texts_to_sequences([inwords])[0]
		encoded = pad_sequences([encoded], maxlen=2, padding='pre')
		yhat = model.predict_classes(encoded)[0]
		inwords += ' ' + revIndex[yhat]
	return inwords'''
    
def generate(model, x, n_words):
    res = x
    for _ in range(n_words):
        padded = pad_sequences([res], maxlen=2, padding='pre')
        res += [ model.predict_classes(padded)[0] ]
    return res

 
data = open('big.txt').read()[:100000] #0]
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]
vocab_size = len(tokenizer.word_index) + 1
sequences = [ encoded[i-2:i+1] for i in range(2, len(encoded)) ]
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=2)) #max_length-1))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('# fit network', file=sys.stderr)
model.fit(X, y, epochs=20, verbose=2)

revIndex = dict( [ (value, key) for key, value in tokenizer.word_index.items() ])

def view(index): return ' '.join( [revIndex.get(i, '?') for i in index])

print( 'I want:', view( generate(model, tokenizer.texts_to_sequences(['I want'])[0], 5) ))

padded = pad_sequences([tokenizer.texts_to_sequences(['I want'])[0]], maxlen=2, padding='pre')
L = sorted([ (iword, prob) for iword, prob in enumerate(model.predict(padded)[0]) ], key=lambda x: -x[1])
print ('I want:', end='\t')
for iword, prob in L[:3]: 
    print ('%s (%s)'%(revIndex[iword], prob), end=',  ')
print ()


#print(generate_seq(model, tokenizer, revIndex, 'And Jill', 3))
#print(generate_seq(model, tokenizer, revIndex, 'fell down', 5))
#print(generate_seq(model, tokenizer, revIndex, 'pail of', 5))


''' model.add(Embedding(vocab_size, 10, input_length=max_length-1))
    model.add(LSTM(50))
    model.add(Dense(vocab_size, activation='softmax'))'''

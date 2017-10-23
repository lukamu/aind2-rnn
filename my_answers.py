import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    series_length = np.size(series)
    X = [series[n: n+window_size] for n in range(series_length-window_size) ]
    y = [series[n+window_size] for n in range(series_length-window_size) ]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    
    # create my LSTM network using Keras
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size ,1)))
    model.add(Dense(1))
    # Apply RMSprop optimizer as suggest in Keras documentation
    rmsProp = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=rmsProp)

    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    english_alphabet = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ")

    filter_map = punctuation + english_alphabet
    # remove chars not present in filter_map
    for char in text:
        if char not in filter_map:
            text = text.replace(char, ' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    text_length = len(text)
    inputs = [ text[n: n+window_size] for n in range(0, text_length-window_size, step_size) ]
    outputs =[ text[n+window_size] for n in range(0, text_length-window_size, step_size) ]
    
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    # create my LSTM model using Keras
    model = Sequential()
    # layer 1 should be an LSTM module with 200 hidden units --> note this should have 
    # input_shape = (window_size,len(chars)) where len(chars) = number of unique characters 
    # in your cleaned text
    model.add(LSTM(200, input_shape=(window_size, num_chars)))

    # layer 2 and 3 should be a linear module, fully connected, with len(chars) hidden units --> where len(chars) = number of unique characters in your cleaned text
    # using a softmax activation ( since we are solving a multiclass classification)
    model.add(Dense(num_chars, activation='softmax'))

    # Apply RMSprop optimizer as suggest in Keras documentation
    rmsProp = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=rmsProp)

    return model

import numpy as np
from td_utils import *
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
# import time


from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

# from train import *

# Use 1101 for 2sec input audio
Tx = 5511  # The number of time steps input to the model from the spectrogram
n_freq = 101  # Number of frequencies input to the model at each time step of the spectrogram


# Use 272 for 2sec input audio
Ty = 1375# The number of time steps in the output of our model


def model(input_shape):
    """
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    X_input = Input(shape=input_shape)

    ### START CODE HERE ###

    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)  # CONV1D
    X = BatchNormalization()(X)  # Batch normalization
    X = Activation('relu')(X)  # ReLu activation
    X = Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization

    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization
    X = Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 4: Time-distributed dense layer (≈1 line)
    X = TimeDistributed(Dense(1, activation="sigmoid"))(X)  # time distributed  (sigmoid)

    ### END CODE HERE ###

    model = Model(inputs=X_input, outputs=X)

    return model


model = model(input_shape = (Tx, n_freq))
# model.summary()

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

model = load_model('tr_model.h5')
print("Model Loaded!")

# print("Try to train")
# activates, negatives, backgrounds = load_raw_audio()
# x, y = create_training_example(backgrounds[0], activates, negatives)
# x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
# model.fit(x, y, batch_size = 5, epochs=1)
# print("Fit!")
# print("Save model")
# time_string = time.strftime("%m-%d-%Y-%H-%M-%S", time.localtime())
# newModalName = time_string + '.h5'
# model.save(newModalName)


# model.save("saveModelTest.h5") to save an model

def detect_triggerword_spectrum(x):
    """
    Function to predict the location of the trigger word.

    Argument:
    x -- spectrum of shape (freqs, Tx)
    i.e. (Number of frequencies, The number time steps)

    Returns:
    predictions -- flattened numpy array to shape (number of output time steps)
    """
    # the spectogram outputs  and we want (Tx, freqs) to input into the model
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    return predictions.reshape(-1)


def has_new_triggerword(predictions, chunk_duration, feed_duration, threshold=0.5):
    """
    Function to detect new trigger word in the latest chunk of input audio.
    It is looking for the rising edge of the predictions data belongs to the
    last/latest chunk.

    Argument:
    predictions -- predicted labels from model
    chunk_duration -- time in second of a chunk
    feed_duration -- time in second of the input to model
    threshold -- threshold for probability above a certain to be considered positive

    Returns:
    True if new trigger word detected in the latest chunk
    """
    predictions = predictions > threshold
    chunk_predictions_samples = int(len(predictions) * chunk_duration / feed_duration)
    chunk_predictions = predictions[-chunk_predictions_samples:]
    level = chunk_predictions[0]
    for pred in chunk_predictions:
        if pred > level:
            return True
        else:
            level = pred
    return False


chunk_duration = 0.5 # Each read length in seconds from mic.
fs = 44100 # sampling rate for mic
chunk_samples = int(fs * chunk_duration) # Each read length in number of samples.

# Each model input data duration in seconds, need to be an integer numbers of chunk_duration
feed_duration = 10
feed_samples = int(fs * feed_duration)

assert feed_duration/chunk_duration == int(feed_duration/chunk_duration)


def get_spectrogram(data):
    """
    Function to compute a spectrogram.

    Argument:
    predictions -- one channel / dual channel audio data as numpy array

    Returns:
    pxx -- spectrogram, 2-D array, columns are the periodograms of successive segments.
    """
    nfft = 200  # Length of each window segment
    fs = 8000  # Sampling frequencies
    noverlap = 120  # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, _, _ = mlab.specgram(data, nfft, fs, noverlap=noverlap)
    elif nchannels == 2:
        pxx, _, _ = mlab.specgram(data[:, 0], nfft, fs, noverlap=noverlap)
    return pxx


def plt_spectrogram(data):
    """
    Function to compute and plot a spectrogram.

    Argument:
    predictions -- one channel / dual channel audio data as numpy array

    Returns:
    pxx -- spectrogram, 2-D array, columns are the periodograms of successive segments.
    """
    nfft = 200  # Length of each window segment
    fs = 8000  # Sampling frequencies
    noverlap = 120  # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, _, _, _ = plt.specgram(data, nfft, fs, noverlap=noverlap)
    elif nchannels == 2:
        pxx, _, _, _ = plt.specgram(data[:, 0], nfft, fs, noverlap=noverlap)
    return pxx



def get_audio_input_stream(callback):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=fs,
        input=True,
        frames_per_buffer=chunk_samples,
        input_device_index=0,
        stream_callback=callback)
    return stream


import pyaudio
from queue import Queue
from threading import Thread
import sys
import time

# Queue to communiate between the audio callback and main thread
q = Queue()

run = True

silence_threshold = 100

# Run the demo for a timeout seconds
timeout = time.time() + 0.5 * 60  # 0.5 minutes from now

# Data buffer for the input wavform
data = np.zeros(feed_samples, dtype='int16')


def callback(in_data, frame_count, time_info, status):
    global run, timeout, data, silence_threshold
    if time.time() > timeout:
        run = False
    data0 = np.frombuffer(in_data, dtype='int16')
    if np.abs(data0).mean() < silence_threshold:
        # sys.stdout.write('-')
        print('-')
        return (in_data, pyaudio.paContinue)
    else:
        # sys.stdout.write('.')
        print('.')
    data = np.append(data, data0)
    if len(data) > feed_samples:
        data = data[-feed_samples:]
        # Process data async by sending a queue.
        q.put(data)
    return (in_data, pyaudio.paContinue)


stream = get_audio_input_stream(callback)
stream.start_stream()

try:
    while run:
        data = q.get()
        spectrum = get_spectrogram(data)
        preds = detect_triggerword_spectrum(spectrum)
        new_trigger = has_new_triggerword(preds, chunk_duration, feed_duration)
        if new_trigger:
            # sys.stdout.write('1')
            print('activate')
except (KeyboardInterrupt, SystemExit):
    stream.stop_stream()
    stream.close()
    timeout = time.time()
    run = False

stream.stop_stream()
stream.close()
#!/usr/bin/env python3

from scipy.fftpack import rfft, irfft, fftfreq
from scipy.io import wavfile
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.structure import BiasUnit
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.utilities import percentError

import matplotlib.pyplot as plt
import numpy as np

import sys

def buildNetwork(N, data):
    dimension = len(data)
    inLayer = LinearLayer(dimension)
    hiddenLayer = SigmoidLayer(N)
    outLayer = SigmoidLayer(dimension)
    bias = BiasUnit(name='bias')
    in_to_hidden = FullConnection(inLayer, hiddenLayer)
    hidden_to_out = FullConnection(hiddenLayer, outLayer)
    bias_to_out = FullConnection(bias, outLayer)
    bias_to_hidden = FullConnection(bias, hiddenLayer)

    net = FeedForwardNetwork()
    net.addModule(bias)
    net.addInputModule(inLayer)
    net.addModule(hiddenLayer)
    net.addOutputModule(outLayer)
    net.addConnection(in_to_hidden)
    net.addConnection(hidden_to_out)
    net.addConnection(bias_to_hidden)
    net.addConnection(bias_to_out)
    net.sortModules()
    return net

def trainNetwork(net, data_):
    dimension = len(data_)
    data = np.copy(data_)
    data.shape = (1, dimension)
    ds = SupervisedDataSet(dimension, dimension)
    ds.appendLinked(data, data)
    trainer = BackpropTrainer(net, dataset=ds)
    trainer.trainEpochs(1000)

if len(sys.argv) != 4:
    print("{} NUM_HIDDEN INPUT_WAV OUTPUT_WAV".format(sys.argv[0]))
    sys.exit(1)

sample_rate, data = wavfile.read(sys.argv[2])
#left_channel = data.T[0]
left_channel = data

num_samples = min(len(left_channel), int(sample_rate*2))
sample = left_channel[:num_samples]
normalized = np.array(sample)/np.iinfo(sample.dtype).max
#transformed = rfft(normalized)
transformed = normalized

NUM_HIDDEN = int(sys.argv[1])
net = buildNetwork(NUM_HIDDEN, transformed)
trainNetwork(net, transformed)
transformed_result = net.activate(transformed)

denormalized_result = (transformed_result + 1) * np.iinfo(sample.dtype).max
#result = irfft(denormalized_result).astype(sample.dtype)
result = denormalized_result

wavfile.write(sys.argv[3], sample_rate, result)

freq = fftfreq(len(transformed), 1. / sample_rate)
t = np.arange(len(tranformed))
plt.plot(t, transformed_result, 'r')
plt.plot(t, transformed, 'b')
#plt.plot(freq, np.abs(transformed_result-transformed), 'r+')
plt.show()

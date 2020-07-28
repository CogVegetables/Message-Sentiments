import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
from collections import defaultdict, UserDict
import mygrad as mg
from pathlib import Path
import pandas as pd
import re, string
from mynn.layers.conv import conv
from mynn.layers.dense import dense
from mynn.activations.relu import relu
from mygrad.nnet.layers import max_pool
from mynn.activations.sigmoid import sigmoid
from mynn.initializers.glorot_normal import glorot_normal
from mynn.optimizers.adam import Adam

from noggin import create_plot


unzipped_folder = "glove.twitter.27B/" # ENTER THE PATH TO THE UNZIPPED `glove.twitter.27B` HERE

# use glove2word2vec to convert GloVe vectors in text format into the word2vec text format:
if not Path('gensim_glove_vectors_200.txt').exists():
    
    # assumes you've downloaded and extracted the glove stuff
    glove2word2vec(glove_input_file= unzipped_folder + "glove.twitter.27B.200d.txt", 
               word2vec_output_file="gensim_glove_vectors_200.txt")

# read the word2vec txt to a gensim model using KeyedVectors
glove = KeyedVectors.load_word2vec_format("gensim_glove_vectors_200.txt", binary=False)


class Model:
    def __init__(self):
        """ Initializes model layers and weights. """
        # <COGINST>
        init_kwargs = {'gain': np.sqrt(2)}
        self.conv1 = conv(200, 250, 2, stride = 1, weight_initializer = glorot_normal, weight_kwargs = init_kwargs)
        self.dense1 = dense(250, 250, weight_initializer = glorot_normal, weight_kwargs = init_kwargs)
        self.dense2 = dense(250,1, weight_initializer = glorot_normal, weight_kwargs = init_kwargs)
        # </COGINST>
    
    
    def __call__(self, x):
        """ Forward data through the network.
        
        This allows us to conveniently initialize a model `m` and then send data through it
        to be classified by calling `m(x)`.
        
        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(N, D, S)
            The data to forward through the network.
            
        Returns
        -------
        mygrad.Tensor, shape=(N, 1)
            The model outputs.
        
        Notes
        -----
        N = batch size
        D = embedding size
        S = sentence length
        """
        # <COGINST>
        # (N, D, S) with D = 200 and S = 77
        x = self.conv1(x) # conv output shape (N, F, S') with F = 250 and S' = 75
        x = relu(x)
        x = max_pool(x, (x.shape[-1],), 1) # global pool output shape (N, F, S') with F = 250, S' = 1
        x = x.reshape(x.shape[0], -1)  # (N, F, 1) -> (N, F)
        x = self.dense1(x) # (N, F) @ (F, D1) = (N, D1)
        x = relu(x) 
        x = self.dense2(x) # (N, D1) @ (D1, 1) = (N, 1)
        x = sigmoid(x)
        return x # output shape (N, 1)
        # </COGINST>
        
    def classify_sentiment(self, text):
        text = batch_gen((text,))
        pol = self(text)
        if pol > 0.5:
            return "Sentiment is positive."
        else:
            return "Sentiment is negative."
        
    def save_parameters(self, path):
        """Path to .npz file where model parameters will be saved."""
        with open(path, "wb") as f:
            np.savez(f, *(x.data for x in self.parameters))
            
    def load_parameters(self, path):
         with open(path, "rb") as f:
            for param, (name, array) in zip(self.parameters, np.load(f).items()):
                param.data[:] = array
    
    @property
    def parameters(self, load = None):
        """ A convenience function for getting all the parameters of our model. """
        return self.conv1.parameters + self.dense1.parameters + self.dense2.parameters # <COGLINE>
    
    
def get_embedding(text):
    """ Returns the word embedding for a given word, reshaping the word embedding array. """
    out = []
    for word in text:
        if word not in glove:
            continue
        else:
            out.append(glove.get_vector(word))
    while len(out) < 80:
        out.append([0]*200)
    return np.array(out).T


punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
def strip_punc(text):
    return punc_regex.sub('', text).lower()


def accuracy(pred, truth):
    """ Calculates the accuracy of the predicted sentiments.
    
    Parameters
    ----------
    pred: Union[numpy.ndarry, mygrad.Tensor]
        The prediction scores of sentiments of the tweets (as a float from 0 to 1)
    
    truth: numpy.ndarry
        The true tweet sentiment (0 or 1)
    
    Returns
    -------
    float
        The accuracy of the predictions
    """
    # <COGINST>
    if isinstance(pred, mg.Tensor):
        pred = pred.data
    return np.mean(np.round(pred) == truth)
    # </COGINST>
    
    
def binary_cross_entropy(y_pred, y_truth):
    """ Calculates the binary cross entropy loss for a given set of predictions.
    
    Parameters
    ----------
    y_pred: mg.Tensor, shape=
        The Tensor of class scores output from the model
    
    y_truth: mg.Tensor, shape=
        A constant Tensor or a NumPy array that contains the truth values for each prediction
    
    Returns
    -------
    mg.Tensor, shape=()
        A zero-dimensional tensor that is the loss
    """
    return -mg.mean(y_truth * mg.log(y_pred + 1e-08) + (1 - y_truth) * mg.log(1 - y_pred + 1e-08)) # <COGLINE>


def batch_gen(text):
    out = []
    for each in text:
        each=strip_punc(each).split()
        out.append(get_embedding(each))
    return np.array(out)
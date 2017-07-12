# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.insert(0, "/Users/yoshinarifujinuma/work/cross_lingual_embed")
sys.path.insert(0, "/Users/yoshinarifujinuma/work/crisisNLP_LRL/python/projection")
from io_ import embeddings_and_vocab
from linear_projection import cos_sim


# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
parser = argparse.ArgumentParser()
parser.add_argument('--embedding_file', type=str, default='/Users/yoshinarifujinuma/work/cross_lingual_embed/eacl_data/ennl.mono.dim=50.bin')
parser.add_argument('--bwesg_embedding_file', type=str, default='/Users/yoshinarifujinuma/work/cross_lingual_embed/eacl_data/ennl.bwesg.dim=50.window=100.bin')
parser.add_argument('--training_data', type=str, default='/Users/yoshinarifujinuma/work/cross_lingual_embed/eacl_data/lex.filtered.train80-20.txt')
args = parser.parse_args()
vocab_S, vocab_T, _, Embs_S, Embs_T, _ = embeddings_and_vocab(args.embedding_file)
word2id_source = {w: i for i, w in enumerate(vocab_S)}
word2id_target = {w: i for i, w in enumerate(vocab_T)}

# Reading the bilingual training examples
source_vecs = []
target_vecs = []
for line in open(args.training_data):
    source, target = line.strip().split("\t")
    source_id = word2id_source[source]
    target_id = word2id_target[target]
    source_vecs.append(Embs_S[source_id])
    target_vecs.append(Embs_T[target_id])
source_vecs = np.array(source_vecs)
target_vecs = np.array(target_vecs)

# Parameters
learning_rate = 0.01
training_epochs = 30
batch_size = 16 
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 32 # 1st layer num features
n_hidden_2 = 16 # 2nd layer num features
#n_input = len(source_vecs) # word vectors
n_input = 50 # word vectors
dim_input = 50 # word vectors

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
X_target = tf.placeholder("float", [None, n_input])
#X = tf.placeholder("float", [dim_input, 1])
#X = tf.placeholder("float", [None, dim_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
#y_true = X
y_true = X_target

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    #total_batch = int(mnist.train.num_examples/batch_size)
    total_batch = int(len(source_vecs)/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            #batch_xs, _ = mnist.train.next_batch(batch_size)
            batch_xs = source_vecs[i:i+batch_size]
            batch_ys = target_vecs[i:i+batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, X_target: batch_ys})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    batch_xs_test = source_vecs[0:examples_to_show] # first five
    encode_decode = sess.run(
        y_pred, feed_dict={X: batch_xs_test})
    print(encode_decode)
    for i in range(examples_to_show):
        print(np.linalg.norm(encode_decode[i] - target_vecs[i])**2)
        print(cos_sim(encode_decode[i], target_vecs[i]))
    # Compare original images with their reconstructions
    #f, a = plt.subplots(2, 10, figsize=(10, 2))
    #for i in range(examples_to_show):
    #    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    #    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    #f.show()
    #plt.draw()
    #plt.waitforbuttonpress()

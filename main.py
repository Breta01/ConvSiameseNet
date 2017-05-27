# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
import time

# Importing dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

class Dataset:
    """
    Store data and provide interface for batches training and testing
    Create positive and negative pairs with ratio 1:1
    Ratio of pairs per lavel depends on labels_equal

    Requires: data, labels (with one hot encoding), number of labels, (eqal)
    """
    def __init__(self, data, labels, n_labels, labels_equal=True, max_pairs=-1):
        self.n_labels = n_labels
        self.label_indices = [np.where(np.argmax(labels, 1) == i)[0]
                              for i in range(n_labels)]
        self.data = data
        self.epoch = 0
        self.labels_equal = labels_equal
        self.max_pairs = max_pairs
        self.pos_pairs = self.generatePosPairs()
        self.neg_pairs = self.generateNegPairs()
        self.length = len(self.pos_pairs)
        self.index = 0

    def generatePosPairs(self):
        """ Returns positive pairs created from data set """
        pairs = []
        labels_len = [len(self.label_indices[d])
                      for d in range(self.n_labels)]

        start_time = time.time() # DEBUG

        if self.labels_equal or self.max_pairs != -1:
            # Number of pairs depends on smallest label dataset
            n = min(labels_len)

            lab = 0
            idx = 0
            pad = 1

            while len(pairs) < self.max_pairs and pad < n:
                pairs += [[self.data[self.label_indices[lab][idx]],
                           self.data[self.label_indices[lab][idx + pad]]]]

                lab = (lab + 1) % self.n_labels
                if lab == 0:
                    idx += 1
                    if (idx + pad) >= n:
                        idx = 0
                        pad += 1

        else:
            # Create maximum number of pairs
            for lab in range(self.n_labels):
                n = labels_len[lab]
                for i in range(n-1):
                    for ii in range(i+1, n):
                        pairs += [[self.data[self.label_indices[lab][i]],
                                    self.data[self.label_indices[lab][ii]]]]

        print("Positive pairs generated in", time.time() - start_time) # DEBUG
        return np.array(pairs)

    def generateNegPairs(self):
        """ Retruns random negative pairs same length as positive pairs """
        pairs = []
        chosen = []
        i = 0
        start_time = time.time() # DEBUG
        while len(pairs) < len(self.pos_pairs):
            ii = (i + random.randrange(1, self.n_labels)) % self.n_labels
            choice = [random.choice(self.label_indices[i]),
                      random.choice(self.label_indices[ii])]
            if choice not in chosen:
                chosen += [choice]
                pairs += [[self.data[choice[0]], self.data[choice[1]]]]
            i = (i + 1) % self.n_labels

        print("Negative pairs generated in", time.time() - start_time) # DEBUG
        return np.array(pairs)

    def get_epoch(self):
        """ Get current dataset epoch """
        return self.epoch

    def get_length(self):
        """ Get positive pairs length """
        return self.length

    def next_batch(self, batch_size):
        """
        Returns batch of images and labels of given length
        Requires: even batch size
        """
        start = self.index
        l_size = int(batch_size / 2)
        self.index += l_size

        if self.index > self.length:
            # Shuffle the data
            perm = np.arange(self.length)
            np.random.shuffle(perm)
            self.pos_pairs = self.pos_pairs[perm]
            self.neg_pairs = self.generateNegPairs()
            # Start next epoch
            start = 0
            self.epoch += 1
            self.index = l_size

        end = self.index
        return (np.append(self.pos_pairs[start:end],
                          self.neg_pairs[start:end], 0),
                np.append(np.ones((l_size, 1)),
                          np.zeros((l_size, 1)), 0))

    def random_batch(self, batch_size):
        """
        Returns random randomly shuffled batch - for testing
        *** Maybe not neccesary ***
        """
        pass


# Layers for CNN
def conv2d(input_, W_shape, name):
    """
    name - layer name for variable scope W_shape - [height, width, input_layers, output_layers]
    """
    with tf.variable_scope(name):
        W_conv = tf.get_variable('W_conv', shape=W_shape,
                                 initializer=tf.contrib.layers.xavier_initializer())
        b_conv = tf.Variable(tf.constant(0.1, shape=[W_shape[3]]), name="b_conv")

        return tf.nn.relu(tf.nn.conv2d(input_,
                                       W_conv,
                                       strides=[1, 1, 1, 1],
                                       padding='SAME') + b_conv)

def max_pool_2x2(input_):
    """ Perform max pool with 2x2 kelner"""
    return tf.nn.max_pool(input_, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def fc(input_, input_dim, output_dim, name):
    """ Fully connected layer with Sigmoid activation """
    with tf.variable_scope(name):
        W_fc = tf.get_variable('W_fc', shape=[input_dim, output_dim],
                        initializer=tf.contrib.layers.xavier_initializer())
        b_fc = tf.Variable(tf.constant(0.1, shape=[output_dim]), name="b_fc")

        return tf.nn.sigmoid(tf.matmul(input_, W_fc) + b_fc)

# Model creator
def convnet(image):
    """
    Input size: 784
    Image initial size: 28x28x1
    After 5_conv size:  7x7x256
    Output vector of 4096 values
    """
    x_image = tf.reshape(image, [-1, 28, 28, 1])
    conv_1 = conv2d(x_image, [10, 10, 1, 64], "1_conv")
    pool_2 = max_pool_2x2(conv_1)
    conv_3 = conv2d(pool_2, [7, 7, 64, 128], "3_conv")
    pool_4 = max_pool_2x2(conv_3)
    conv_5 = conv2d(pool_4, [4, 4, 128, 256], "5_conv")
    flat_6 = tf.reshape(conv_5, [-1, 7*7*256])
    full_7 = fc(flat_6, 7*7*256, 4096, "7_full")
    return full_7


X_train = mnist.train.images
y_train = mnist.train.labels
X_test = mnist.test.images
y_test = mnist.test.labels

tr_data = Dataset(X_train, y_train, 10, max_pairs=5000)
te_data = Dataset(X_test, y_test, 10, max_pairs=5000)


### MODEL
images_L = tf.placeholder(tf.float32,shape=([None,784]),name='images_L')
images_R = tf.placeholder(tf.float32,shape=([None,784]),name='images_R')
labels = tf.placeholder(tf.float32,shape=([None,1]), name='labels')

with tf.variable_scope("ConvSiameseNet") as scope:
    model_L = convnet(images_L)
    scope.reuse_variables()
    model_R = convnet(images_R)

# Combine two outputs by L1 distance
distance = tf.abs(tf.subtract(model_L, model_R))

# Final layer with sigmoid
W_out = tf.get_variable('W_out', shape=[4096, 1],
                        initializer=tf.contrib.layers.xavier_initializer())
b_out = tf.Variable(tf.constant(0.1, shape=[1]), name="b_out")

# Output - result of sigmoid - for future use
# Prediction - rounded sigmoid to 0 or 1
output = tf.nn.sigmoid(tf.matmul(distance, W_out) + b_out)
prediction = tf.round(output)

# Using cross entropy for sigmoid as loss
# @TODO add regularization
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                              logits=output))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss)

# Measuring accuracy of model
correct_prediction = tf.equal(prediction, labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


### TRAINING
batch_size = 64 # 128

with tf.Session() as sess:
    print("Starting training")
    tf.global_variables_initializer().run()

    # Training cycle
    for epoch in range(30):
        avg_loss = 0.
        avg_acc = 0.
        total_batch = 20            # Not accurate
        start_time = time.time()

        # Loop over all batches
        for i in range(total_batch):
            # Fit training using batch data
            tr_input, y = tr_data.next_batch(batch_size)
            _, loss_value, acc = sess.run([optimizer, loss, accuracy],
                                          feed_dict={images_L: tr_input[:,0],
                                                     images_R: tr_input[:,1],
                                                     labels: y})
            avg_loss += loss_value
            avg_acc += acc * 100

        duration = time.time() - start_time
        print('epoch %d  time: %f loss %0.5f acc %0.2f' % (epoch,
                                                           duration,
                                                           avg_loss/total_batch,
                                                           avg_acc/total_batch))
        te_pairs, te_y = te_data.next_batch(1000)
        te_acc = accuracy.eval(feed_dict={images_L: te_pairs[:,0],
                                          images_R: te_pairs[:,1],
                                          labels: te_y})
        print('Accuracy on test set %0.2f' % (100 * te_acc))


    # Final Testing
    tr_pairs, tr_y = te_data.next_batch(1000)
    tr_acc = accuracy.eval(feed_dict={images_L: tr_pairs[:,0],
                                      images_R: tr_pairs[:,1],
                                      labels: tr_y})
    print('Accuract training set %0.2f' % (100 * tr_acc))

    te_pairs, te_y = te_data.next_batch(1000)
    te_acc = accuracy.eval(feed_dict={images_L: te_pairs[:,0],
                                      images_R: te_pairs[:,1],
                                      labels: te_y})
    print('Accuract test set %0.2f' % (100 * te_acc))


    # TODO Predicting correct label based accuracy on sample of labeled data.

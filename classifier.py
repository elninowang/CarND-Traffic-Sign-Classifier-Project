# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data
training_file = 'traffic-signs-data/train.p'
validation_file = 'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np
import collections

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(collections.Counter(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

import matplotlib.pyplot as plt

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)
X_test, y_test = shuffle(X_test, y_test)

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
X_train_gray = np.sum(X_train/3, axis=3, keepdims=True)
X_valid_gray = np.sum(X_valid/3, axis=3, keepdims=True)
X_test_gray = np.sum(X_test/3, axis=3, keepdims=True)
print('X_train_gray shape:', X_train_gray.shape)

## Normalize the train and test datasets to (-1,1)
X_train_normalized = (X_train_gray - 128)/128
X_valid_normalized = (X_valid_gray - 128)/128
X_test_normalized = (X_test_gray - 128)/128

X_train = X_train_normalized
X_valid = X_valid_normalized
X_test = X_test_normalized

### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding="VALID") + conv1_b
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)
    layer1 = conv1
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer2 = conv1

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding="VALID") + conv2_b
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)
    layer3 = conv2
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer4 = conv2

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits, layer1, layer2, layer3, layer4

def LeNet2(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding="VALID") + conv1_b
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)
    layer1 = conv1
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer2 = conv1

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma), name="W2")
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding="VALID") + conv2_b
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)
    layer3 = conv2
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer4 = conv2

    # TODO: Layer 3: Convolutional. Output = 1x1x400.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 400), mean=mu, stddev=sigma))
    conv3_b = tf.Variable(tf.zeros(400), name="b3")
    conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID')
    conv3 = tf.nn.bias_add(conv3, conv3_b)
    # TODO: Activation.
    conv3 = tf.nn.relu(conv3)

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)
    print("layer2flat shape:", fc0.get_shape())

    # Flatten x. Input = 1x1x400. Output = 400.
    fc1 = flatten(conv3)
    print("xflat shape:", fc1.get_shape())
    # Concat layer2flat and x. Input = 400 + 400. Output = 800
    fc2 = tf.concat([fc0, fc1], 1)
    print("fc2 shape:", fc2.get_shape())
    # Dropout
    fc2 = tf.nn.dropout(fc2, keep_prob)

    # TODO: Layer 4: Fully Connected. Input = 800. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(800, 43), mean=mu, stddev=sigma), name="W4")
    fc3_b = tf.Variable(tf.zeros(43), name="b4")
    logits = tf.add(tf.matmul(fc2, fc3_W), fc3_b)

    return logits, layer1, layer2, layer3, layer4

#parameters
BATCH_SIZE = 128
EPOCHS = int(n_train/BATCH_SIZE) + 1  #use all the train data
rate = 0.001
dropout = 0.5

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)

logits, layer1, layer2, layer3, layer4 = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

saver = tf.train.Saver()
### Training, if you what to training, change the Flase to True
if False:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        print("Training...")
        print()
        for i in range(EPOCHS):
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

            validation_accuracy = evaluate(X_valid, y_valid)
            print("EPOCH {} / {}: Validation Accuracy = {:.3f}".format(i + 1, EPOCHS, validation_accuracy))
            print()

        saver.save(sess, './lenet')
        print("Model saved")

#Testing, if you what to Testing, change the Flase to True
if False:
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./lenet")
        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))
        test_accuracy = evaluate(X_train, y_train)
        print("Train Accuracy = {:.3f}".format(test_accuracy))
        test_accuracy = evaluate(X_valid, y_valid)
        print("Valid Accuracy = {:.3f}".format(test_accuracy))

### Load the images and plot them here.
### Feel free to use as many code cells as needed.
#reading in an image
import glob
import matplotlib.image as mpimg
import cv2

saver = tf.train.Saver()

my_images = []
for i, img in enumerate(glob.glob('./my-found-traffic-signs/*x.png')):
    image = cv2.imread(img)
    # axs[i].axis('off')
    # axs[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    my_images.append(image)

my_images = np.asarray(my_images)
my_images_gry = np.sum(my_images/3, axis=3, keepdims=True)
my_images_normalized = (my_images_gry - 128)/128
print(my_images_normalized.shape)

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

my_labels = [3, 11, 1, 12, 38, 34, 18, 25]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # saver3 = tf.train.import_meta_graph('./lenet.meta')
    # saver3.restore(sess, "./lenet")
    saver.restore(sess, "./lenet")
    y_predict = sess.run(tf.argmax(logits, 1), feed_dict={x: my_images_normalized, keep_prob: dropout})
    print("predict result is: ", y_predict)
    correct_prediction = tf.equal(tf.constant(my_labels), tf.constant(y_predict.astype(np.int32)))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    my_accuracy = sess.run(accuracy_operation)
    print("Test Set Accuracy = {:.3f}".format(my_accuracy))

import pandas as pd
k = 5
softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=k)
my_top_k = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./lenet")
    my_softmax_logits = sess.run(softmax_logits, feed_dict={x: my_images_normalized, keep_prob: 0.5})
    my_top_k = sess.run(top_k, feed_dict={x: my_images_normalized, keep_prob: 0.5})
    print(my_top_k)
# show the pictures
# fig, axs = plt.subplots(len(my_images),(k+1), figsize=(12, 20))
# axs = axs.ravel()
#
# for i, image in enumerate(my_images):
#     axs[(k+1)*i].axis('off')
#     axs[(k+1)*i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     axs[(k+1)*i].set_title('input')
#     for j in range(k):
#         guess = my_top_k[1][i][j]
#         index = np.argwhere(y_train == guess)[0]
#         axs[(k+1)*i+j+1].axis('off')
#         axs[(k+1)*i+j+1].imshow(X_train[index].squeeze(), cmap='gray')
#         axs[(k+1)*i+j+1].set_title('guest {}: ({:.0f}%)'.format(guess, 100*my_top_k[0][i][j]))
# plt.show()


### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
    plt.show()

with tf.Session() as sess:
    saver.restore(sess, "./lenet")
    print("Feature Map Stage 1:")
    outputFeatureMap(my_images_normalized, layer1, plt_num=1)

with tf.Session() as sess:
    saver.restore(sess, "./lenet")
    print("Feature Map Stage 2:")
    outputFeatureMap(my_images_normalized, layer2, plt_num=2)

with tf.Session() as sess:
    saver.restore(sess, "./lenet")
    print("Feature Map Stage 3:")
    outputFeatureMap(my_images_normalized, layer3, plt_num=3)

with tf.Session() as sess:
    saver.restore(sess, "./lenet")
    print("Feature Map Stage 4:")
    outputFeatureMap(my_images_normalized, layer4, plt_num=4)
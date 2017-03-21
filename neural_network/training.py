import glob
import os
import math
import numpy as np
from PIL import Image
import tensorflow as tf

IMAGE_RESULT = 12
IMAGE_SIZE = 100
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# input X: 100 grayscale images, the first dimension will index the images in the mini-batch
grayscale_images = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1])
# correct answers will go here / my label
label_images = tf.placeholder(tf.float32, [None, IMAGE_RESULT])
# variable learning rate
learning_rate = tf.placeholder(tf.float32)

# three convolutional layers with their channel counts and a fully connected layer
first_convolution = 4
second_convolution = 8
third_convolution = 12
fully_connected_layer = 200

weights_1 = tf.Variable(tf.truncated_normal([5, 5, 1, first_convolution], stddev=0.1))  # 5x5 patch, 1 input channel, first_convolution output channels
bias_1 = tf.Variable(tf.ones([first_convolution])/IMAGE_RESULT)
weights_2 = tf.Variable(tf.truncated_normal([5, 5, first_convolution, second_convolution], stddev=0.1))
bias_2 = tf.Variable(tf.ones([second_convolution])/IMAGE_RESULT)
weights_3 = tf.Variable(tf.truncated_normal([4, 4, second_convolution, third_convolution], stddev=0.1))
bias_3 = tf.Variable(tf.ones([third_convolution])/IMAGE_RESULT)

weights_4 = tf.Variable(tf.truncated_normal([25 * 25 * third_convolution, fully_connected_layer], stddev=0.1))
bias_4 = tf.Variable(tf.ones([fully_connected_layer])/IMAGE_RESULT)
weights_5 = tf.Variable(tf.truncated_normal([fully_connected_layer, IMAGE_RESULT], stddev=0.1))
bias_5 = tf.Variable(tf.ones([IMAGE_RESULT])/IMAGE_RESULT)

# The model
stride = 1  # output is 100*100
layer_1 = tf.nn.relu(tf.nn.conv2d(grayscale_images, weights_1, strides=[1, stride, stride, 1], padding='SAME') + bias_1)
stride = 2  # output is 50*50
layer_2 = tf.nn.relu(tf.nn.conv2d(layer_1, weights_2, strides=[1, stride, stride, 1], padding='SAME') + bias_2)
stride = 2  # output is 25x25
layer_3 = tf.nn.relu(tf.nn.conv2d(layer_2, weights_3, strides=[1, stride, stride, 1], padding='SAME') + bias_3)

# reshape the output from the third convolution for the fully connected layer
layer_convert_full = tf.reshape(layer_3, shape=[-1, 25 * 25 * third_convolution])
layer_4 = tf.nn.relu(tf.matmul(layer_convert_full, weights_4) + bias_4)

# Label predict by neural network
label_full = tf.matmul(layer_4, weights_5) + bias_5
label_prediction = tf.nn.softmax(label_full)

# Add to the Graph the Ops for loss calculation.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=label_full, labels=label_images)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
is_correct_prediction = tf.equal(tf.argmax(label_prediction, 1), tf.argmax(label_images, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Create a session for running Ops on the Graph.
initialize = tf.global_variables_initializer()
session = tf.Session()
session.run(initialize)
saver = tf.train.Saver()


def get_test_images(path):
    images = []
    for filename in glob.iglob(path, recursive=True):
        # Open file and convert to grayscale
        image = Image.open(filename).convert('1')
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        images.append(np.array(image))
    images = np.array(images)
    images = images.reshape(len(images), IMAGE_PIXELS)
    return images


def get_all_image_training(path):
    images_dictionary = {}

    for filename in glob.iglob(path, recursive=True):
        directory_name = os.path.basename(os.path.dirname(filename))
        if directory_name not in images_dictionary:
            images_dictionary[directory_name] = []

        images_dictionary[directory_name].append(filename)
    return images_dictionary


def get_training_images_and_labels(path):
    train_images = []
    train_labels = []

    for label, filenames in get_all_image_training(path).items():
        for filename in filenames:
            # Open file and convert to grayscale
            image = Image.open(filename).convert('1')
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            train_images.append(np.array(image))
            train_labels.append(int(label))

    train_images = np.array(train_images)
    train_images = train_images.reshape(len(train_images), IMAGE_PIXELS)

    train_labels = np.array(train_labels)
    zero_labels = np.zeros((len(train_images), IMAGE_RESULT))

    for i, _ in enumerate(train_images):
        zero_labels[i][train_labels[i]] = 1

    return train_images, zero_labels


def get_pebble_count(predictions):
    return np.argmax(predictions[0])


def restore_session():
    try:
        saver.restore(session, './saved_graphs/awale')
        print("Restored graph file")
        return True
    except Exception:
        print("Cannot restore graph file", "Run 'make run -- -t'")
        return False


def get_learning_rate(index_training):
    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    return min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-index_training/decay_speed)


def run_training():
    train_images, train_labels = get_training_images_and_labels('images/**/*.png')
    train_images = np.reshape(train_images, (len(train_images), IMAGE_SIZE, IMAGE_SIZE, 1))

    # Train
    for i in range(100):
        # learning rate
        lr = get_learning_rate(i)

        a, c = session.run([accuracy, cross_entropy], {grayscale_images: train_images, label_images: train_labels})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(lr) + ")")

        # the backpropagation training step
        session.run(train_step, {grayscale_images: train_images, label_images: train_labels, learning_rate: lr})

    saver.save(session, './saved_graphs/awale')
    print("Saving session graph")


def display_count_pebble():
    test_images = get_test_images('board_images/*.png')
    test_images = np.reshape(test_images, (len(test_images), IMAGE_SIZE, IMAGE_SIZE, 1))

    success = restore_session()
    if not success:
        return

    # Run trained model
    predictions = session.run(label_full, {grayscale_images: test_images})
    number_pebble = get_pebble_count(predictions)
    print(number_pebble)


def display_accuracy():
    train_images, train_labels = get_training_images_and_labels('images/**/*.png')
    train_images = np.reshape(train_images, (len(train_images), IMAGE_SIZE, IMAGE_SIZE, 1))

    success = restore_session()
    if not success:
        return

    # Test trained model
    print(session.run(accuracy, {grayscale_images: train_images, label_images: train_labels}))

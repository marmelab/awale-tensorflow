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
X = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1])
# correct answers will go here / my label
Y_ = tf.placeholder(tf.float32, [None, IMAGE_RESULT])
# variable learning rate
lr = tf.placeholder(tf.float32)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 16 softmax neurons)
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/IMAGE_RESULT)
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/IMAGE_RESULT)
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/IMAGE_RESULT)

W4 = tf.Variable(tf.truncated_normal([25 * 25 * M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/IMAGE_RESULT)
W5 = tf.Variable(tf.truncated_normal([N, IMAGE_RESULT], stddev=0.1))
B5 = tf.Variable(tf.ones([IMAGE_RESULT])/IMAGE_RESULT)

# The model
stride = 1  # output is 100*100
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 50*50
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 25x25
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 25 * 25 * M])
Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)

# Label predict by neural network
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

# Add to the Graph the Ops for loss calculation.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
is_correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# Create a session for running Ops on the Graph.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()


def get_test_images(path):
    images = []
    for filename in glob.iglob(path, recursive=True):
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
        saver.restore(sess, './saved_graphs/awale')
        print("Restored graph file")
        return True
    except Exception:
        print("Cannot restore graph file", "Run 'make run -- -t'")
        return False


def run_training():
    train_images, train_labels = get_training_images_and_labels('images/**/*.png')
    train_images = np.reshape(train_images, (len(train_images), IMAGE_SIZE, IMAGE_SIZE, 1))

    # Train
    for i in range(100):
        # learning rate decay
        max_learning_rate = 0.003
        min_learning_rate = 0.0001
        decay_speed = 2000.0
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

        a, c = sess.run([accuracy, cross_entropy], {X: train_images, Y_: train_labels})
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")

        # the backpropagation training step
        sess.run(train_step, {X: train_images, Y_: train_labels, lr: learning_rate})

    saver.save(sess, './saved_graphs/awale')
    print("Saving session graph")


def display_count_pebble():
    test_images = get_test_images('board_images/*.png')
    test_images = np.reshape(test_images, (len(test_images), IMAGE_SIZE, IMAGE_SIZE, 1))

    success = restore_session()
    if not success:
        return

    # Run trained model
    a = sess.run(Ylogits, {X: test_images})
    number_pebble = get_pebble_count(a)
    print(number_pebble)


def display_accuracy():
    train_images, train_labels = get_training_images_and_labels('images/**/*.png')
    train_images = np.reshape(train_images, (len(train_images), IMAGE_SIZE, IMAGE_SIZE, 1))

    success = restore_session()
    if not success:
        return

    # Test trained model
    print(sess.run(accuracy, {X: train_images, Y_: train_labels}))

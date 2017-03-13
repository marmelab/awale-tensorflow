import glob
import os
import math
import numpy as np
from PIL import Image
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 4, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

IMAGE_RESULT = 16
IMAGE_SIZE = 100
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 3


def get_all_image_training():
    dico = {}

    for filename in glob.iglob('images/**/*.png', recursive=True):
        directory_name = os.path.basename(os.path.dirname(filename))
        if directory_name != '1':
            continue

        if directory_name not in dico:
            dico[directory_name] = []

        dico[directory_name].append(filename)
    return dico


def get_training_image_label():
    train_images = []
    train_labels = []

    for label, filenames in get_all_image_training().items():
        for filename in filenames:
            image = Image.open(filename)
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


def run_training():
    train_images, train_labels = get_training_image_label()
    print(len(train_images), len(train_labels))
    print(len(train_images[0]), len(train_labels[0]))

    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
    W = tf.Variable(tf.zeros([IMAGE_PIXELS, IMAGE_RESULT]))
    b = tf.Variable(tf.zeros(IMAGE_RESULT))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, IMAGE_RESULT])

    print("x", x)
    print("W", W)
    print("b", b)
    print("y", y)
    print("y_", y_)

    # Construct model
    # logits = multilayer_network(train_images,
    #                             FLAGS.hidden1, FLAGS.hidden2)

    # Add to the Graph the Ops for loss calculation.
    loss = cal_loss(y_, y)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = training(loss, FLAGS.learning_rate)

    # Create a session for running Ops on the Graph.
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for _ in range(FLAGS.max_steps):
        sess.run(train_op, feed_dict={x: train_images, y_: train_labels})

    # Test trained model
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print(sess.run(accuracy, feed_dict={x: train_images[0][0],
    #                                     y_: train_labels[0]}))


def multilayer_network(images, hidden1_units, hidden2_units):
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
            name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
    out_layer = tf.matmul(hidden2, weights) + biases

    return out_layer


def cal_loss(labels, loss):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                       logits=loss))


def training(loss, learning_rate):
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

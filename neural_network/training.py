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

IMAGE_RESULT = 16
IMAGE_SIZE = 100
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 3

hidden_layers = [
    {
        'neurons': 128,
        'activation_function': 'sigmoid',
    },
    {
        'neurons': 32,
        'activation_function': 'sigmoid',
    },
    {
        'neurons': IMAGE_RESULT,
        'activation_function': 'softmax',
    },
]

x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS])
W = tf.Variable(tf.zeros([IMAGE_PIXELS, IMAGE_RESULT]))
b = tf.Variable(tf.zeros(IMAGE_RESULT))
y = tf.add(tf.matmul(x, W), b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, IMAGE_RESULT])


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def multilayer_network(x, hidden_layers):
    previous_size = IMAGE_PIXELS
    previous_layer = x
    for layer in hidden_layers:
        layer_size = layer['neurons']
        layer['weights'] = tf.Variable(tf.random_normal([previous_size, layer_size]))
        layer['biases'] = tf.Variable(tf.random_normal([layer_size]))

        layer['predict'] = tf.add(tf.matmul(previous_layer, layer['weights']), layer['biases'])

        if 'activation_function' in layer:
            if layer['activation_function'] == 'sigmoid':
                layer['predict'] = tf.sigmoid(layer['predict'])
            elif layer['activation_function'] == 'tanh':
                layer['predict'] = tf.tanh(layer['predict'])
            elif layer['activation_function'] == 'softmax':
                layer['predict'] = tf.nn.softmax(layer['predict'])

        if 'dropout' in layer:
            dropout = tf.constant(layer['dropout'])
            layer['predict'] = tf.nn.dropout(layer['predict'], dropout)

        variable_summaries(layer['weights'])
        variable_summaries(layer['biases'])
        tf.summary.histogram('pre_activations', layer['predict'])

        previous_size = layer_size
        previous_layer = layer['predict']

    return hidden_layers[-1]['predict']


# Construct model
predict = multilayer_network(x, hidden_layers)

# Add to the Graph the Ops for loss calculation.
loss = tf.contrib.losses.softmax_cross_entropy(predict, y)
tf.summary.scalar('cross_entropy', loss)

# Add to the Graph the Ops that calculate and apply gradients.
train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

# Create a session for running Ops on the Graph.
sess = tf.InteractiveSession()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./train', sess.graph)

tf.global_variables_initializer().run()
saver = tf.train.Saver()


def get_test_images(path):
    images = []
    for filename in glob.iglob(path, recursive=True):
        image = Image.open(filename)
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

    # Train
    for _ in range(FLAGS.max_steps):
        summary = sess.run(train_op, feed_dict={x: train_images, y_: train_labels})
        train_writer.add_summary(summary, _)

    saver.save(sess, './saved_graphs/awale')
    print("Saving session graph")


def display_count_pebble():
    test_images = get_test_images('board_images/*.png')

    success = restore_session()
    if not success:
        return

    # Run trained model
    correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    predictions = sess.run(predict, feed_dict={x: test_images})
    number_pebble = get_pebble_count(predictions)
    print(number_pebble)


def display_accuracy():
    train_images, train_labels = get_training_images_and_labels('images/**/*.png')

    success = restore_session()
    if not success:
        return

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: train_images, y_: train_labels}))

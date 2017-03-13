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

NUM_CLASSES = 2
IMAGE_SIZE = 100
CHANNELS = 3
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNELS


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
        train_labels.append(label)
        for filename in filenames:
            image = Image.open(filename)
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            train_images.append(np.array(image))

    train_images = np.array(train_images)
    return train_images, np.array(train_labels)


def run_training():
    train_images, train_labels = get_training_image_label()

    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(len(train_images))

    # Construct model
    logits = multilayer_network(images_placeholder,
                                FLAGS.hidden1, FLAGS.hidden2)

    # Backward propagation
    # Add to the Graph the Ops for loss calculation.
    loss = cal_loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = evaluation(logits, labels_placeholder)

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # for step in range(FLAGS.max_steps):
    #     feed_dict = fill_feed_dict(train_images, train_labels,
    #                                images_placeholder,
    #                                labels_placeholder)
    #
    #     _, loss_value = sess.run(train_op, feed_dict=feed_dict)
    #     if step % 100 == 0:
    #         # Print status to stdout.
    #         print('Step %d: loss = %.2f' % (step, loss_value))
    #         if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
    #             saver.save(sess, FLAGS.train_dir, global_step=step)
    #             print('Training Data Eval:')
    #             do_eval(sess,
    #                     eval_correct,
    #                     images_placeholder,
    #                     labels_placeholder,
    #                     train_images,
    #                     train_labels)


def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32,
                                        shape=(batch_size, IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


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


def cal_loss(loss, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=loss,
                                                       labels=labels))


def training(loss, learning_rate):
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def fill_feed_dict(images_feed, labels_feed, images_pl, labels_pl):
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict

def do_eval(sess, eval_correct, images_placeholder, labels_placeholder,
            train_images, train_labels):
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = 4 // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(train_images, train_labels,
                                   images_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
        precision = true_count / num_examples
        print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
              (num_examples, true_count, precision))

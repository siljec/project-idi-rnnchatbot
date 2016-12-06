# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

sys.path.insert(0, '../Preprocessing') # To access methods from another file from another folder
from create_vocabulary import read_vocabulary_from_file
from preprocess import generate_all_files
from tokenize import sentence_to_token_ids

from preprocess import file_len

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import gridLSTM_model


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 4, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 64, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 100000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 100000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "./Ola_data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./Ola_data", "Training directory.")
tf.app.flags.DEFINE_string("log_dir", "./Ola_data/log_dir", "Logging directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 10, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

# Paths
vocab_path = '../Preprocessing/vocabulary.txt'
x_train_path = '../Preprocessing/x_train.txt'
y_train_path = '../Preprocessing/y_train.txt'
x_dev_path = '../Preprocessing/x_val.txt'
y_dev_path = '../Preprocessing/y_val.txt'

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_EOT = b"_EOT"
_UNK = b"_UNK"

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
EOT_ID = 3
UNK_ID = 4


def input_pipeline(root='../Preprocessing/', start_name='train_merged.txt'):

    # Finds all filenames that match the root and start_name
    filenames = [root + filename for filename in os.listdir(root) if filename.startswith(start_name)]

    # Adds the filenames to the queue
    # Can also add args such as num_epocs and shuffle. shuffle=True will shuffle the files from 'filenames'
    filename_queue = tf.train.string_input_producer(filenames)
    print("Files added to queue: ", filenames)

    return filename_queue


def get_batch(source, train_set, batch_size=FLAGS.batch_size):

    # Feed buckets until one of them reach the batch_size
    while len(max(train_set)) < batch_size:

        # Convert tensor to array
        holder = source.eval()
        holder = holder.split(',')

        # x_data is on the left side of the comma, while y_data is on the right. Also casting to integers.
        x = [int(i) for i in holder[0].split()]
        y = [int(i) for i in holder[1].split()]

        # Feed the correct bucket to input the read line. Lines longer than the largest bucket is excluded.
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
            if len(x) < source_size and len(y) < target_size:
                train_set[bucket_id].append([x, y])
                break

    # Find the largest bucket (that made the while loop terminate)
    largest_bucket = max(train_set)
    largest_bucket_index = train_set.index(largest_bucket)

    # Extracting data that should be returned as training data
    # This should be of length batch_size. Can later add a check just to be sure
    train_data = train_set[largest_bucket_index]

    # # Clean the bucket with the extracted data
    # train_set[largest_bucket_index] = []

    return train_set, train_data, largest_bucket_index


def check_for_needed_files_and_create():
    if not os.path.isdir("./../../ubuntu-ranking-dataset-creator"):
        print("Ubuntu Dialogue Corpus not found or is not on the right path. ")
        print('1')
        print('cd out from project-idi-rnnchatbot')
        print('2')
        print('\t git clone https://github.com/rkadlec/ubuntu-ranking-dataset-creator.git')
        print('3')
        print('\t cd ubuntu-ranking-dataset-creator/src')
        print('4')
        print('\t ./generate.sh')
    if not os.path.isfile("./../Preprocessing/x_train.txt"):
        generate_all_files(FLAGS.en_vocab_size)
    if not os.path.isfile("./../Preprocessing/y_train.txt"):
        generate_all_files(FLAGS.en_vocab_size)
    if not os.path.isfile("./../Preprocessing/x_val.txt"):
        generate_all_files(FLAGS.en_vocab_size)
    if not os.path.isfile("./../Preprocessing/y_val.txt"):
        generate_all_files(FLAGS.en_vocab_size)
    if not os.path.isfile("./../Preprocessing/x_test.txt"):
        generate_all_files(FLAGS.en_vocab_size)
    if not os.path.isfile("./../Preprocessing/y_test.txt"):
        generate_all_files(FLAGS.en_vocab_size)
    if not os.path.isfile("./../Preprocessing/vocabulary.txt"):
        generate_all_files(FLAGS.en_vocab_size)


def read_data(source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

    Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """

    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                #target_ids.append(EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = gridLSTM_model.GridLSTM_model(
        FLAGS.en_vocab_size,
        FLAGS.fr_vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        forward_only=forward_only,
        dtype=dtype)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def get_session_configs():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


def train():
    """Train a en->fr translation model using WMT data."""

    print("Checking for needed files")
    check_for_needed_files_and_create()

    # Prepare Ubuntu Dialogue Corpus data.
    print("Preparing Ubuntu Dialogue Corpus data in %s" % FLAGS.data_dir)

    x_train = x_train_path
    y_train = y_train_path
    x_dev = x_dev_path
    y_dev = y_dev_path

    filename_queue = input_pipeline()
    filename_queue_dev = input_pipeline(start_name='val_merged.txt')

    # Avoid allocating all of the GPU memory
    config = get_session_configs()

    with tf.Session(config=config) as sess:
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False)

        # Stream data
        print("Setting up coordinator")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Read data into buckets and compute their sizes.
        print("Reading development and training data (limit: %d)." % FLAGS.max_train_data_size)
        # dev_set = read_data(x_dev, y_dev)
        # train_set = read_data(x_train, y_train, FLAGS.max_train_data_size)
        # train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        # train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        print("Creating scaled bucket probability")
        # train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
        #                        for i in xrange(len(train_bucket_sizes))]

        # This is for the training loop.
        train_set = [[] for _ in _buckets]
        dev_set = [[] for _ in _buckets]
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        # Create log writer object
        print("Create log writer object")
        summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, graph=tf.get_default_graph())

        reader_train_data = tf.TextLineReader()  # skip_header_lines=int, number of lines to skip
        _, txt_row_train_data = reader_train_data.read(filename_queue)

        reader_dev_data = tf.TextLineReader()
        _, txt_row_dev_data = reader_dev_data.read(filename_queue_dev)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        print("Starts training loop")

        try:
            while True: #not coord.should_stop():
                print("New training epoch")
                train_set, batch, bucket_id = get_batch(txt_row_train_data, train_set)

                print(train_set)

                # Get a batch and make a step.
                start_time = time.time()
                print("Get batch")
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)

                # Clean out trained bucket
                train_set[bucket_id] = []

                print("Make step")
                _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)

                print("Calculate variables")
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint
                current_step += 1

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % FLAGS.steps_per_checkpoint == 0:
                    # Print statistics for the previous epoch.

                    dev_set, batch, bucket_id = get_batch(txt_row_dev_data, dev_set)

                    perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                    print("global step %d learning rate %.4f step-time %.2f perplexity "
                          "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))

                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                        sess.run(model.learning_rate_decay_op)
                    previous_losses.append(loss)

                    # Save checkpoint and zero timer and loss.
                    print("Save checkpoint")
                    checkpoint_path = os.path.join(FLAGS.train_dir, "Ola.ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                    step_time, loss = 0.0, 0.0
                    perplexity_summary = tf.Summary()

                    # Run evals on development set and print their perplexity.
                    print("Run evaluation on development set")
                    for bucket_id in xrange(len(_buckets)):
                        if len(dev_set[bucket_id]) == 0:
                            print("  eval: empty bucket %d" % bucket_id)
                            continue
                        encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)

                        # Clean out used bucket
                        dev_set[bucket_id] = []

                        _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
                        eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                        print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                        bucket_value = perplexity_summary.value.add()
                        bucket_value.tag = "perplexity_bucket %d" % bucket_id
                        bucket_value.simple_value = eval_ppx
                    summary_writer.add_summary(perplexity_summary, model.global_step.eval())
                    sys.stdout.flush()
        except tf.errors.OutOfRangeError:
            print('Done training, epoch reached')
        finally:
            #coord.request_stop()
            coord.join(threads)


def decode():
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, True)
        model.batch_size = 1  # We decode one sentence at a time.
        # Load vocabularies.
        vocab, rev_vocab = read_vocabulary_from_file(vocab_path)
        # Decode from standard input.
        sys.stdout.write("Human: ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        while sentence:

            # Get token-ids for the input sentence.
            token_ids = sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab)

            # Which bucket does it belong to?
            bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)

            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

            # If there is an EOS symbol in outputs, cut them at that point.
            if EOT_ID in outputs:
                outputs = outputs[:outputs.index(EOT_ID)]

            print("Ola: " + " ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))
            print("Human: ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


def self_test():
    """Test the translation model."""
    with tf.Session() as sess:
        print("Self-test for neural translation model.")
        # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
        model = gridLSTM_model.GridLSTM_model(10, 10, [(3, 3), (6, 6)], 32, 2, 5.0, 32, 0.3, 0.99, num_samples=8)
        sess.run(tf.initialize_all_variables())
        # Fake data set for both the (3, 3) and (6, 6) bucket.
        data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                    [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
        for _ in xrange(5):  # Train the fake model for 5 steps.
            bucket_id = random.choice([0, 1])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(data_set, bucket_id)
            model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)


def main(_):
    if FLAGS.self_test:
        self_test()
    elif FLAGS.decode:
        decode()
    else:
        train()

if __name__ == "__main__":
  tf.app.run()

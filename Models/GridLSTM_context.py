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

import os
import sys
import re
from math import exp
from random import choice
import time

sys.path.insert(0, '../') # To access methods from another file from another folder
from variables import paths_from_model as paths, tokens, _buckets, vocabulary_size, steps_per_checkpoint, print_frequency, max_training_steps, size, batch_size, num_layers, use_gpu
sys.path.insert(0, '../Preprocessing') # To access methods from another file from another folder
from create_vocabulary import read_vocabulary_from_file
from tokenize import sentence_to_token_ids
from helpers import check_for_needed_files_and_create, preprocess_input

import numpy as np
import tensorflow as tf

from six.moves import xrange  # pylint: disable=redefined-builtin

import gridLSTM_model
import fasttext

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", batch_size, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", size, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", num_layers, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", vocabulary_size, "English vocabulary size.")
tf.app.flags.DEFINE_integer("print_frequency", print_frequency, "How many training steps to do per print.")
tf.app.flags.DEFINE_integer("max_train_steps", max_training_steps, "How many training steps to do.")
tf.app.flags.DEFINE_string("data_dir", "./GridContext_data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./GridContext_data", "Training directory.")
tf.app.flags.DEFINE_string("log_dir", "./GridContext_data/log_dir", "Logging directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", steps_per_checkpoint, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

_PAD, PAD_ID = tokens['padding']
_GO, GO_ID = tokens['go']
_EOS, EOS_ID = tokens['eos']
_EOT, EOT_ID = tokens['eot']
_UNK, UNK_ID = tokens['unk']


# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.


def input_pipeline(root=paths['preprocess_root_files_context'], start_name=paths['train_file_context']):
    # Finds all filenames that match the root and start_name
    filenames = [root + filename for filename in os.listdir(root) if filename.startswith(start_name)]

    # Adds the filenames to the queue
    # Can also add args such as num_epocs and shuffle. shuffle=True will shuffle the files from 'filenames'
    filename_queue = tf.train.string_input_producer(filenames)
    print("Files added to queue: ", filenames)

    return filename_queue


def get_batch(source, train_set, batch_size=FLAGS.batch_size, ac_function=max):
    # Feed buckets until one of them reach the batch_size
    while ac_function([len(x) for x in train_set]) < batch_size:

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
    _, largest_bucket_index = max([(len(x), i) for i, x in enumerate(train_set)])

    return train_set, largest_bucket_index


def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = gridLSTM_model.GridLSTM_model(
        FLAGS.vocab_size,
        FLAGS.vocab_size,
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
        # tf.initialize_all_variables() will soon be deprecated
        session.run(tf.initialize_all_variables())
        # session.run(tf.global_variables_initializer())
    return model


def get_session_configs():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    return config


def train():
    """Train a en->fr translation model using WMT data."""
    boot_time = time.time()
    print("Checking for needed files")
    check_for_needed_files_and_create(FLAGS.vocab_size)

    print("Creating file queue")
    filename_queue = input_pipeline(start_name=paths['train_file_context'])
    filename_queue_dev = input_pipeline(start_name=paths['dev_file_context'])

    # Avoid allocating all of the GPU memory
    config = get_session_configs()

    with tf.device(use_gpu):
        with tf.Session(config=config) as sess:
            # Create model.
            print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
            model = create_model(sess, False)

            # Stream data
            print("Setting up coordinator")
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

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

            train_time = time.time()
            boot_time = time.time()-boot_time

            minutes = int(boot_time / 60)
            seconds = boot_time % 60
            print("Time ", minutes, " minutes ", seconds, " seconds to boot")

            print("Starting training loop")
            try:
                while current_step < FLAGS.max_train_steps:  # not coord.should_stop():
                    if current_step % FLAGS.print_frequency == 0:
                        print("Step number: " + str(current_step))

                    # Get a batch
                    train_set, bucket_id = get_batch(txt_row_train_data, train_set)
                    start_time = time.time()
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)


                    # Clean out trained bucket
                    train_set[bucket_id] = []

                    # Make a step
                    _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)

                    # Calculating variables
                    step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                    loss += step_loss / FLAGS.steps_per_checkpoint
                    current_step += 1

                    # Once in a while, we save checkpoint, print statistics, and run evals.
                    if current_step % FLAGS.steps_per_checkpoint == 0:
                        duration = time.time() - train_time
                        minutes = int(duration / 60)
                        seconds = duration % 60
                        check_time = time.time()
                        print("Time ", minutes, " minutes ", seconds, " seconds to train")
                        # Print statistics for the previous epoch.
                        dev_set, bucket_id = get_batch(txt_row_dev_data, dev_set, ac_function=min)

                        perplexity = exp(float(loss)) if loss < 300 else float("inf")
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

                        # Adding perplexity to tensorboard
                        perplexity_summary = tf.Summary()
                        overall_value = perplexity_summary.value.add()
                        overall_value.tag = "perplexity_overall"
                        overall_value.simple_value = perplexity

                        # Run evals on development set and print their perplexity.
                        print("Run evaluation on development set")
                        for bucket_id in xrange(len(_buckets)):
                            if len(dev_set[bucket_id]) == 0:
                                print("  eval: empty bucket %d" % bucket_id)
                                continue
                            encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)

                            # Clean out used bucket
                            del dev_set[bucket_id][:FLAGS.batch_size]

                            _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
                            eval_ppx = exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                            print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

                            # Adding bucket perplexity to tensorboard
                            bucket_value = perplexity_summary.value.add()
                            bucket_value.tag = "perplexity_bucket %d" % bucket_id
                            bucket_value.simple_value = eval_ppx
                        summary_writer.add_summary(perplexity_summary, model.global_step.eval())
                        sys.stdout.flush()
                        duration = time.time() - check_time
                        minutes = int(duration / 60)
                        seconds = duration % 60
                        print("Time ", minutes, " minutes ", seconds, " seconds to do checkpoint")
                        train_time = time.time()
            except tf.errors.OutOfRangeError:
                print('Done training, epoch reached')
            finally:
                coord.request_stop()
            coord.join(threads)



def swap_eos(sentence):
    sentence_holder = []
    for word in sentence:
        if word == '_EOS':
            sentence_holder.append(' \n')
        else:
            sentence_holder.append(word)
    return sentence_holder


def decode():
    with tf.Session(config=get_session_configs()) as sess:
        # Create model and load parameters.
        model = create_model(sess, True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Load trained FastText model
        fast_text_model = model = fasttext.load_model(paths['fast_text_model_path_context'], encoding='utf-8')

        # Load vocabularies.
        vocab, rev_vocab = read_vocabulary_from_file(paths['vocab_path_context'])

        # Get vocab_word vectors TODO: Should be a file to load
        vocab_vectors = {}
        for word, item in vocab.iteritems():
            vector = fast_text_model[word]
            vocab_vectors[word] = vector, np.linalg.norm(vector)

        # Decode from standard input.
        sys.stdout.write("Human: ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        sentence = preprocess_input(sentence, fast_text_model, vocab_vectors)
        context = ""
        while sentence:
            # Get token-ids for the input sentence.
            token_ids = sentence_to_token_ids(tf.compat.as_bytes(context + sentence), vocab)

            # Which bucket does it belong to?
            bucket_id = min([b for b in xrange(len(_buckets))
                             if _buckets[b][0] > len(token_ids)])

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                {bucket_id: [(token_ids, [])]}, bucket_id)

            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)

            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

            # If there is an EOS symbol in outputs, cut them at that point.
            if EOT_ID in outputs:
              outputs = outputs[:outputs.index(EOT_ID)]

            # Print out sentence corresponding to outputs.
            output = [tf.compat.as_str(rev_vocab[output]) for output in outputs]
            output = swap_eos(output)
            print("Ola: " + " ".join(output))
            print("Human: ", end="")

            context = sentence # or context = output
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            sentence = preprocess_input(sentence)


def self_test():
    """Test the model."""
    with tf.Session(config=get_session_configs()) as sess:
        print("Self-test for neural translation model.")
        # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
        model = gridLSTM_model.GridLSTM_model(10, 10, [(3, 3), (6, 6)], 32, 2, 5.0, 32, 0.3, 0.99, num_samples=8)
        # sess.run(tf.global_variables_initializer())
        sess.run(tf.initialize_all_variables())
        # Fake data set for both the (3, 3) and (6, 6) bucket.
        data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                    [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
        for _ in xrange(5):  # Train the fake model for 5 steps.
            bucket_id = choice([0, 1])
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

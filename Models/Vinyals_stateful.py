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
from math import exp
from random import choice
import time
import fasttext

sys.path.insert(0, '../Preprocessing') # To access methods from another file from another folder
from create_vocabulary import read_vocabulary_from_file
from preprocess_helpers import load_pickle_file, get_time

from helpers import check_for_needed_files_and_create, preprocess_input, get_stateful_batch, input_pipeline, get_session_configs, self_test, decode_sentence, check_and_shuffle_file
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import seq2seq_stateful_model as seq2seq_model
sys.path.insert(0, '../')
from variables import paths_from_model as paths, tokens, _buckets, vocabulary_size, max_training_steps, \
    steps_per_checkpoint, print_frequency, size, batch_size, num_layers, use_gpu


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", batch_size, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", size, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", num_layers, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", vocabulary_size, "English vocabulary size.")
tf.app.flags.DEFINE_integer("print_frequency", print_frequency, "How many training steps to do per print.")
tf.app.flags.DEFINE_integer("max_train_steps", max_training_steps, "How many training steps to do.")
tf.app.flags.DEFINE_string("data_dir", "./Stateful_data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./Stateful_data", "Training directory.")
tf.app.flags.DEFINE_string("log_dir", "./Stateful_data/log_dir", "Logging directory.")
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

_buckets = [_buckets[-1]]


def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.vocab_size,
        FLAGS.vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        forward_only=forward_only)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def train():
    """Train a en->fr translation model using WMT data."""

    print("Checking for needed files")
    check_for_needed_files_and_create()

    print("Creating file queues")

    filename_queue = input_pipeline(root=paths['stateful_datafiles'], start_name="train", shuffle=True)

    filename_queue_dev = input_pipeline(root=paths['stateful_datafiles'], start_name="dev", shuffle=True)

    perplexity_log_path = os.path.join(FLAGS.train_dir, paths['perplexity_log'])

    if not os.path.exists(perplexity_log_path):
        with open(perplexity_log_path, 'w') as fileObject:
            fileObject.write("Step \tPerplexity \tBucket perplexity")

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
            step_time, loss = 0.0, 0.0
            current_step = 0
            train_set = [[] for _ in range(batch_size)]
            previous_losses = []

            # Create log writer object
            print("Create log writer object")
            summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, graph=tf.get_default_graph())

            _, txt_row_train_data = tf.TextLineReader().read(filename_queue)

            _, txt_row_dev_data = tf.TextLineReader().read(filename_queue_dev)

            lowest_perplexity = 20.0

            train_time = time.time()

            # Need a initial state for the encoder rnn
            initial_state = np.zeros((num_layers, batch_size, size))
            state = initial_state

            print("Starts training loop")

            try:
                while FLAGS.max_train_steps >= current_step:  # not coord.should_stop():
                    if current_step % FLAGS.print_frequency == 0:
                        print("Step number" + str(current_step))

                    # Get a batch
                    train_set, batch_train_set, state = get_stateful_batch(txt_row_train_data, train_set, state, _buckets[0])
                    start_time = time.time()
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(batch_train_set)

                    # Make a step
                    _, step_loss, _, state = model.step(sess, encoder_inputs, decoder_inputs, target_weights, state, False)

                    # Calculating variables
                    step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                    loss += step_loss / FLAGS.steps_per_checkpoint
                    current_step += 1

                    # Once in a while, we save checkpoint, print statistics, and run evals.
                    if current_step % FLAGS.steps_per_checkpoint == 0:

                        check_time = time.time()
                        print(get_time(train_time), "to train")

                        # Print statistics for the previous epoch.
                        dev_set, batch_dev_set, _ = get_stateful_batch(txt_row_dev_data, dev_set, initial_state, _buckets[0])

                        perplexity = exp(float(loss)) if loss < 300 else float("inf")
                        print("global step %d learning rate %.4f step-time %.2f perplexity "
                              "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))

                        # Decrease learning rate if no improvement was seen over last 3 times.
                        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                            sess.run(model.learning_rate_decay_op)
                        previous_losses.append(loss)

                        # Save checkpoint and zero timer and loss.
                        print("Save checkpoint")
                        checkpoint_path = os.path.join(FLAGS.train_dir, "Vinyals.ckpt")
                        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                        step_time, loss = 0.0, 0.0

                        # Adding perplexity to tensorboard
                        perplexity_summary = tf.Summary()
                        overall_value = perplexity_summary.value.add()
                        overall_value.tag = "perplexity_overall"
                        overall_value.simple_value = perplexity

                        # Run evals on development set and print their perplexity.
                        print("Run evaluation on development set")
                        bucket_perplexity = ""
                        for bucket_id in xrange(len(_buckets)):
                            if len(dev_set[bucket_id]) == 0:
                                print("  eval: empty bucket %d" % bucket_id)
                                continue
                            encoder_inputs, decoder_inputs, target_weights, state = model.get_batch(batch_dev_set)

                            _, eval_loss, _, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, initial_state, True)
                            eval_ppx = exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                            print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

                            bucket_perplexity += "\t" + str(eval_ppx)

                            # Adding bucket perplexity to tensorboard
                            bucket_value = perplexity_summary.value.add()
                            bucket_value.tag = "perplexity_bucket %d" % bucket_id
                            bucket_value.simple_value = eval_ppx
                        summary_writer.add_summary(perplexity_summary, model.global_step.eval())

                        with open(os.path.join(FLAGS.train_dir, paths['perplexity_log']), 'a') as fileObject:
                            fileObject.write(str(model.global_step) + " \t" + str(perplexity) + bucket_perplexity + "\n")

                        # Save model if checkpoint was the best one
                        if perplexity < lowest_perplexity:
                            lowest_perplexity = perplexity
                            checkpoint_path = os.path.join(FLAGS.train_dir, "Vinyals_stateful_best_.ckpt")
                            model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                        sys.stdout.flush()
                        get_time(check_time, "to do checkpoint")
                        train_time = time.time()
            except tf.errors.OutOfRangeError:
                print('Done training, epoch reached')
            finally:
                coord.request_stop()
            coord.join(threads)


def decode():
    # Avoid allocating all of the GPU memory
    config = get_session_configs()

    with tf.Session(config=config) as sess:
        # Create model and load parameters.
        model = create_model(sess, True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        vocab, rev_vocab = read_vocabulary_from_file(paths['vocab_path'])

        # Load vocabulary vectors
        vocab_vectors = load_pickle_file(paths['vocab_vectors'])

        # Load FastText model used for preprocessing
        print("Load existing FastText model...")
        fast_text_model = fasttext.load_model(paths['fast_text_model'], encoding='utf-8')

        # Decode from standard input.
        sys.stdout.write("Human: ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        sentence = preprocess_input(sentence, fast_text_model, vocab_vectors)
        while sentence:

            output = decode_sentence(sentence, vocab, rev_vocab, model, sess)

            print("Vinyals: " + " ".join(output))
            print("Human: ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            sentence = preprocess_input(sentence, fast_text_model, vocab_vectors)




def main(_):
    if FLAGS.self_test:
        self_test(seq2seq_model.Seq2SeqModel)
    elif FLAGS.decode:
        decode()
    else:
        train()

if __name__ == "__main__":
    tf.app.run()
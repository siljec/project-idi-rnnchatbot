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
from preprocess_helpers import load_pickle_file, get_time, shuffle_file

from helpers import check_for_needed_files_and_create, preprocess_input, get_batch, input_pipeline, get_session_configs, self_test, decode_sentence, check_and_shuffle_file
sys.path.insert(0, '../')
from variables import paths_from_model as paths, tokens, _buckets, vocabulary_size, max_training_steps, print_frequency, steps_per_checkpoint, size, num_layers, batch_size, use_gpu
from variables import contextFullTurns, context

import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin

import gridLSTM_model

if context:
    from variables import paths_from_preprocessing_context as paths
if contextFullTurns:
    from variables import paths_from_preprocessing_contextFullTurns as paths

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", batch_size, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", size, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", num_layers, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", vocabulary_size, "English vocabulary size.")
tf.app.flags.DEFINE_integer("print_frequency", print_frequency, "How many training steps to do per print.")
tf.app.flags.DEFINE_integer("max_train_steps", max_training_steps, "How many training steps to do.")
tf.app.flags.DEFINE_string("data_dir", "./Ola_data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./Ola_data", "Training directory.")
tf.app.flags.DEFINE_string("log_dir", "./Ola_data/log_dir", "Logging directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", steps_per_checkpoint, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.

_PAD, PAD_ID = tokens['padding']
_GO, GO_ID = tokens['go']
_EOS, EOS_ID = tokens['eos']
_EOT, EOT_ID = tokens['eot']
_UNK, UNK_ID = tokens['unk']

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
        session.run(tf.initialize_all_variables())
    return model


def train():
    """Train a en->fr translation model using WMT data."""

    print("Checking for needed files")
    check_for_needed_files_and_create()
    train_path = paths['train_path']
    shuffle_file(train_path, train_path)

    print("Creating file queue")
    filename_queue = input_pipeline(start_name=paths['train_file'])
    filename_queue_dev = input_pipeline(start_name=paths['dev_file'])

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
            train_set = [[] for _ in _buckets]
            dev_set = [[] for _ in _buckets]
            step_time, loss = 0.0, 0.0
            current_step = 0
            previous_losses = []
            read_line = 0
            reading_file_path = ""

            # Create log writer object
            print("Create log writer object")
            summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, graph=tf.get_default_graph())

            reader_train_data = tf.TextLineReader()  # skip_header_lines=int, number of lines to skip
            key, txt_row_train_data = reader_train_data.read(filename_queue)

            reader_dev_data = tf.TextLineReader()
            _, txt_row_dev_data = reader_dev_data.read(filename_queue_dev)

            lowest_perplexity = 20.0

            train_time = time.time()

            print("Starting training loop")
            try:
                while current_step < FLAGS.max_train_steps:  # not coord.should_stop():
                    if current_step % FLAGS.print_frequency == 0:
                        print("Step number: " + str(current_step))

                    read_line, reading_file_path = check_and_shuffle_file(key, sess, read_line, paths['train_path'])

                    # Get a batch
                    train_set, bucket_id = get_batch(txt_row_train_data, train_set, FLAGS.batch_size)
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
                        check_time = time.time()
                        print(get_time(train_time, "to train"))
                        # Print statistics for the previous epoch.
                        dev_set, bucket_id = get_batch(txt_row_dev_data, dev_set, FLAGS.batch_size, ac_function=min)

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
                        bucket_perplexity = ""
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

                            bucket_perplexity += "\t" + str(eval_ppx)

                            # Adding bucket perplexity to tensorboard
                            bucket_value = perplexity_summary.value.add()
                            bucket_value.tag = "perplexity_bucket %d" % bucket_id
                            bucket_value.simple_value = eval_ppx
                        summary_writer.add_summary(perplexity_summary, model.global_step.eval())

                        with open(os.path.join(FLAGS.train_dir, paths['perplexity_log']), 'a') as fileObject:
                            fileObject.write(str(model.global_step) + " \t" + str(perplexity) + bucket_perplexity + "\n")
                        # Save model if checkpoint was the best one
                        if perplexity < lowest_perplexity:  # and current_step > 400000:
                            lowest_perplexity = perplexity
                            checkpoint_path = os.path.join(FLAGS.train_dir, "Ola_best_.ckpt")
                            model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                        sys.stdout.flush()
                        print(get_time(check_time), "to do checkpoint")
                        train_time = time.time()
            except tf.errors.OutOfRangeError:
                print('Done training, epoch reached')
            finally:
                coord.request_stop()
            coord.join(threads)


def decode():
    with tf.Session(config=get_session_configs()) as sess:
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
            print("Ola: " + " ".join(output))
            print("Human: ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            sentence = preprocess_input(sentence, fast_text_model, vocab_vectors)


def main(_):
    if FLAGS.self_test:
        self_test(gridLSTM_model.GridLSTM_model)
    elif FLAGS.decode:
        decode()
    else:
        train()

if __name__ == "__main__":
  tf.app.run()

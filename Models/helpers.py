import os
import sys
import re
sys.path.insert(0, '../Preprocessing') # To access methods from another file from another folder
from preprocess import start_preprocessing
from variables import paths_from_model, tokens
from preprocessing3 import distance
from preprocess_helpers import shuffle_file
from variables import tokens, paths_from_model as paths, _buckets
import tensorflow as tf
import numpy as np
from random import choice

_, UNK_ID = tokens['unk']
_, EOT_ID = tokens['eot']


def read_words_from_misspelling_file(path):
    dictionary = {}
    with open(path) as fileobject:
        for line in fileobject:
            splitted_line = line.split(' ', 1)
            wrong = splitted_line[0]
            correct = splitted_line[1].strip()
            dictionary[wrong] = correct

    return dictionary


def replace_misspelled_word_helper(candidate, dictionary):
    if (candidate in dictionary):
        # print "replacing ", candidate, " with ", dictionary[candidate]
        return dictionary[candidate]
    return candidate


def replace_misspelled_words_in_sentence(sentence, misspelllings_path):
    dictionary = read_words_from_misspelling_file(misspelllings_path) #get the misspelled words as a dictionary
    tokenized_sentence = sentence.split(' ')
    final_sentence = ""
    for word in tokenized_sentence:
        new_word = replace_misspelled_word_helper(word, dictionary)
        final_sentence += " " + new_word
    return final_sentence


def check_for_needed_files_and_create():
    if not os.path.isdir(paths_from_model['ubuntu']):
        print("Ubuntu Dialogue Corpus not found or is not on the right path. ")
        print('1')
        print('cd out from project-idi-rnnchatbot')
        print('2')
        print('\t git clone https://github.com/rkadlec/ubuntu-ranking-dataset-creator.git')
        print('3')
        print('\t cd ubuntu-ranking-dataset-creator/src')
        print('4')
        print('\t ./generate.sh')

    if not os.path.isfile(paths_from_model['train_path']):
        print("You should start preprocessing")
    if not os.path.isfile(paths_from_model['dev_path']):
        print("You should start preprocessing")
    if not os.path.isfile(paths_from_model['test_path']):
        print("You should start preprocessing")
    if not os.path.isfile(paths_from_model['vocab_path']):
        print("You should start preprocessing")


def preprocess_input(sentence, fast_text_model, vocab):
    emoji_token = " " + tokens['emoji'][0] + " "
    dir_token = tokens['directory'][0]
    url_token = " " + tokens['url'][0] + " "

    sentence = sentence.strip().lower()
    sentence = re.sub(' +', ' ', sentence)  # Will remove multiple spaces
    sentence = re.sub('(?<=[a-z])([!?,.])', r' \1', sentence)  # Add space before special characters [!?,.]
    sentence = re.sub(r'(https?://[^\s]+)', url_token, sentence)  # Exchange urls with URL token
    sentence = re.sub(r'((?:^|\s)(?::|;|=)(?:-)?(?:\)|\(|D|P|\|)(?=$|\s))', emoji_token,
                  sentence)  # Exchange smiles with EMJ token NB: Will neither take :) from /:) nor from :)D
    sentence = re.sub('(?<=[a-z])([!?,.])', r' \1', sentence)  # Add space before special characters [!?,.]
    sentence = re.sub('"', '', sentence)  # Remove "
    sentence = re.sub('((\/\w+)|(\.\/\w+)|(\w+(?=(\/))))()((\/)|(\w+)|(\.\w+)|(\w+|\-|\~))+', dir_token,
                  sentence)  # Replace directory-paths
    sentence = re.sub("(?!(')([a-z]{1})(\s))(')(?=\w|\s)", "", sentence)  # Remove ', unless it is like "banana's"
    sentence = replace_misspelled_words_in_sentence(sentence, paths_from_model['misspellings'])

    # Must replace OOV with most similar vocab-words:
    unk_words = {}
    for word in sentence.split():
        if word not in vocab:
            unk_words[word] = fast_text_model[word]

    # Find most similar words
    similar_words = {}
    for unk_word, unk_vector in unk_words.iteritems():
        min_dist = 10
        word = ""
        for key, value in vocab.iteritems():
            cur_dist = distance(unk_vector, value[0], value[1])
            # Save the word that is most similar
            if cur_dist < min_dist:
                min_dist = cur_dist
                word = key
        similar_words[unk_word] = word

    # Replace words
    for word, similar_word in similar_words.iteritems():
        sentence.replace(word, similar_word)

    return sentence

_WORD_SPLIT = re.compile(b"([.,!?\":;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens"""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def sentence_to_token_ids(sentence, vocabulary):
    """Convert a string to list of integers representing token-ids.
    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
    Returns:
    a list of integers, the token-ids for the sentence.
    """
    words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]


def check_and_shuffle_file(key, sess, read_line, file_path):
    # Check if we should shuffle training file
    holder = int(sess.run(key).split(":")[1])
    if holder < read_line:
        shuffle_file(file_path, file_path)
        print("Training file shuffled")

    return holder


def get_stateful_batch(source, train_set, state, bucket_size):

    # Find empty lists in
    empty_conversations = [index for index, conversation in enumerate(train_set) if conversation == []]

    # Reset state where there are new conversations
    for entry in empty_conversations:
        state[entry][0] = [0] * bucket_size[0]
        state[entry][1] = [0] * bucket_size[0]


    # Feed batch
    while empty_conversations != []:

        current_index = empty_conversations.pop()

        # Convert tensor to array
        holder = source.eval()
        holder = holder.split(',')

        # x_data is on the left side of the comma, while y_data is on the right. Also casting to integers.
        x = [int(i) for i in holder[0].split()]
        y = [int(i) for i in holder[1].split()]

        # Fill an entire conversation to the list
        while x != EOT_ID:
            # Feed the correct bucket to input the read line. Lines longer than the largest bucket is excluded.
            train_set[current_index].append([x, y])

            # Convert tensor to array
            holder = source.eval()
            holder = holder.split(',')

            # x_data is on the left side of the comma, while y_data is on the right. Also casting to integers.
            x = [int(i) for i in holder[0].split()]
            y = [int(i) for i in holder[1].split()]

    # Return the first pairs in all of the lists
    batch_training_set = [pairs[0] for pairs in train_set]

    # Remove the batch_training_set in the train_set
    train_set = [pairs[1:] for pairs in train_set]

    return train_set, batch_training_set, state


def get_batch(source, train_set, batch_size, ac_function=max):
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


def input_pipeline(root=paths['preprocess_root_files'], start_name=paths['train_file'], shuffle=False):
    # Finds all filenames that match the root and start_name
    filenames = [root + filename for filename in os.listdir(root) if filename.startswith(start_name)]

    # Adds the filenames to the queue
    # Can also add args such as num_epocs and shuffle. shuffle=True will shuffle the files from 'filenames'
    filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)
    # print("Files added to queue: ", filenames)

    return filename_queue


def get_session_configs():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    return config


def self_test(test_model):
    """Test the model."""

    # Avoid allocating all of the GPU memory
    config = get_session_configs()

    with tf.Session(config=config) as sess:
          print("Self-test for neural translation model.")
          # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
          model = test_model(10, 10, [(3, 3), (6, 6)], 32, 2,
                                             5.0, 32, 0.3, 0.99, num_samples=8)
          sess.run(tf.initialize_all_variables())

          # Fake data set for both the (3, 3) and (6, 6) bucket.
          data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                      [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
          for _ in xrange(5):  # Train the fake model for 5 steps.
            bucket_id = choice([0, 1])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                data_set, bucket_id)
            model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                       bucket_id, False)


def decode_sentence(sentence, vocab, rev_vocab, model, sess):

        # Get token-ids for the input sentence.
        token_ids = sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab)

        # Which bucket does it belong to?
        if len(token_ids) >= _buckets[-1][0]:
            print("Sentence too long. Slicing it to fit a bucket")
            token_ids = token_ids[:(_buckets[-1][0] - 1)]
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
        return output

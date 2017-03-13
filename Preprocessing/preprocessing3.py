from itertools import izip
import numpy as np
import time
import os
from preprocess_helpers import find_dictionary, save_to_pickle, get_time, load_pickle_file, save_vocabulary, save_dict_to_file


def preprocessing3(buckets, source_file_x, source_file_y, bucket_data_x, bucket_data_y, vocab_size, vocab_txt_path, vocab_pickle_path, fasttext_model, vocab_vectors_path, unk_vectors_path, unk_to_vocab_pickle_path, unk_to_vocab_txt_path, save_frequency_unk_words, final_x_path, final_y_path, init_tokens):

    # Step 1: Filter out data to buckets
    get_data_for_buckets(buckets, source_file_x, bucket_data_x, source_file_y, bucket_data_y)

    # Step 2: Create new vocabulary
    sorted_dict = find_dictionary(x_train=bucket_data_x, y_train=bucket_data_y)

    # Vocab is an array
    vocab = sorted_dict[:vocab_size]

    save_vocabulary(vocab_txt_path, vocab, init_tokens)
    save_to_pickle(vocab_pickle_path, vocab)

    unknown_words_dict = get_unk_dict(sorted_dict=sorted_dict, vocab_size=vocab_size)

    # Find word embeddings
    unknown_words_embeddings, vocab_words_embeddings = get_vectors_to_words(fasttext_model, vocab_txt_path, vocab_vectors_path, unknown_words_dict, init_tokens)
    save_to_pickle(unk_vectors_path, unknown_words_embeddings)

    # Get most similar words to unk
    unk_to_vocab_mapping = get_most_similar_words_for_unk(unknown_words_embeddings, vocab_words_embeddings, unk_to_vocab_pickle_path, unk_to_vocab_txt_path, save_frequency_unk_words)

    # Replace unk words in dataset
    replace_unk_with_most_similar(bucket_data_x, final_x_path, unk_to_vocab_mapping)
    replace_unk_with_most_similar(bucket_data_y, final_y_path, unk_to_vocab_mapping)


def get_data_for_buckets(buckets, source_file_x, bucket_data_x, source_file_y, bucket_data_y):
    biggest_bucket = buckets[-1]  # Only need to check if data fits our biggest bucket

    new_x_data = []
    new_y_data = []

    with open(source_file_x) as sourceObjectX, open(source_file_y) as sourceObjectY:
        for x, y in izip(sourceObjectX, sourceObjectY):
            x_words = x.split()
            y_words = y.split()
            if len(x_words) <= biggest_bucket[0] and len(y_words) <= biggest_bucket[1]:
                new_x_data.append(x)
                new_y_data.append(y)

    with open(bucket_data_x, 'w') as bucketObjectX, open(bucket_data_y, 'w') as bucketObjectY:
        for x, y in izip(new_x_data, new_y_data):
            bucketObjectX.write(x)
            bucketObjectY.write(y)


# Step 3: Find Vectors to word
def get_unk_dict(sorted_dict, vocab_size, init_tokens=['_PAD', '_GO', '_EOS', '_EOT', '_UNK']):
    unknown_dict = {}
    counter = 0
    for key in sorted_dict:
        if counter < vocab_size:
            if key[0] not in init_tokens:
                counter += 1
        else:
            unknown_dict[key[0]] = 0

    return unknown_dict


def get_vectors_to_words(model, vocabulary_path, vocab_vectors_path, unknown_words, init_tokens=['_PAD', '_GO', '_EOS', '_EOT', '_UNK']):
    print('Get vectors and length for vocabulary words')
    with open(vocabulary_path) as vocabObject:
        vocab_words = {}
        for word in vocabObject:
            word = word.strip()
            if word in init_tokens:
                continue
            vector = np.array(model[word])
            vocab_words[word] = vector, np.linalg.norm(vector)

    save_to_pickle(vocab_vectors_path, vocab_words)

    print('Get vectors for out-of-vocabulary words')
    for key in unknown_words:
        unknown_words[key] = np.array(model[key])

    print("# UNK words: ", len(unknown_words))
    return unknown_words, vocab_words


def distance(unk, known, known_length):
    unk_len = np.linalg.norm(unk)
    numerator = np.dot(unk, known)
    denominator = unk_len * known_length
    return 1 - (numerator / denominator)

def get_most_similar_words_for_unk(unknown_words, vocab_words, unknown_dict_pickle_path, unk_to_vocab_txt_path, save_freq):

    # The resulting dictionary consisting of 'unk_word' : 'most similar vocab word'
    unknown_words_results = {}
    # If a previously dictionary is saved, this one will be fed with words that has NOT computed a similar word
    new_unk_words_dict = {}

    # If pickle file exists, load into unknown_words_results
    if os.path.exists(unknown_dict_pickle_path):
        unknown_words_results = load_pickle_file(unknown_dict_pickle_path)
        for key, value in unknown_words.iteritems():
            if key not in unknown_words_results:
                # If the word is not computed, add to new_unk_words_dict so it can be computed later
                new_unk_words_dict[key] = value

        # Set unknown_words to the words that is not computed
        unknown_words = new_unk_words_dict

    # Create lists for faster computation
    known_words_list = [(key, value[0], value[1]) for key, value in vocab_words.iteritems()]
    unknown_words_list = [(key, value) for key, value in unknown_words.iteritems()]

    counter = 1
    start_time_unk = time.time()
    # Loop all unknown_words
    for unk_key, unk_values in unknown_words_list:
        min_dist = 1
        word = ""
        if (counter % 5000) == 0:
            print("     Calculated " + str(counter) + " unknown words")
        # Loop all vocab words for calculating the distance
        for key, value, dis in known_words_list:
            cur_dist = distance(unk_values, value, dis)
            # Save the word that is most similar
            if cur_dist < min_dist:
                min_dist = cur_dist
                word = key
        # Save most similar vocab_word to the unk_word
        unknown_words_results[unk_key] = word
        counter += 1

        # Once in a while, save checkpoints
        if counter % save_freq == 0:
            save_to_pickle(unknown_dict_pickle_path, unknown_words_results)
            print("   Saved temporarily unknown_words_dictionary")
    print("Time to get similar words for all UNK:", get_time(start_time_unk))
    save_to_pickle(unknown_dict_pickle_path, unknown_words_results)
    save_dict_to_file(unk_to_vocab_txt_path, unknown_words_results)

    return unknown_words_results


def replace_word_helper(candidate, dictionary):
    if candidate in dictionary:
        return dictionary[candidate]
    return candidate


def read_words_from_misspelling_file(path):
    dictionary = {}
    with open(path) as fileobject:
        for line in fileobject:
            splitted_line = line.split(' ', 1)
            wrong = splitted_line[0]
            correct = splitted_line[1].strip()
            dictionary[wrong] = correct

    return dictionary


def replace_unk_with_most_similar(source_file_path, new_file_path, unk_to_vocab_mapping):
    new_file = open(new_file_path, 'w')
    with open(source_file_path) as fileobject:
        for line in fileobject:
            words = line.split(' ')
            sentence = ""
            last_word = words.pop().strip()
            for word in words:
                new_word = replace_word_helper(word, unk_to_vocab_mapping)
                sentence += new_word + ' '
            new_word = replace_word_helper(last_word, unk_to_vocab_mapping)
            new_file.write(sentence + new_word + '\n')
    new_file.close()


sorted_dict = find_dictionary(x_train="./datafiles/bucket_data_x.txt", y_train="./datafiles/bucket_data_y.txt")
# Vocab is an array
vocab = sorted_dict[:100000]
save_vocabulary("./datafiles/vocab100000.txt", vocab, ['_PAD', '_GO', '_EOS', '_EOT', '_UNK'])
import operator
import re
import time
import random
import numpy as np
from preprocess_helpers import distance
from create_vocabulary import find_dictionary

def get_stats(path, num_longest=20, more_than_words=50, less_than_words=5):

    print("\n############## Stats for " + path + " ##############")

    sentence_lengths = dict()

    with open(path) as file_object:
        longest = [0 for _ in range(num_longest)]
        max_length = 0
        turns_with_more_than_x_words = 0
        turns_with_less_than_x_words = 0
        all_lines = file_object.readlines()
        num_turns = len(all_lines)
        num_words = 0
        for line in all_lines:
            length = len(line.split(' '))
            num_words += length
            if length > more_than_words:
                turns_with_more_than_x_words += 1
            if length < less_than_words:
                turns_with_less_than_x_words += 1
            if longest[0] < length:
                longest.pop(0)
                longest.append(length)
                longest.sort()
            if max_length < length:
                max_length = length
            if length in sentence_lengths:
                sentence_lengths[length] += 1
            else:
                sentence_lengths[length] = 1

    print("File: " + path + ". Turns in total: " + str(num_turns))
    print("Longest turn: " + str(max_length))

    print("Longest " + str(num_longest) + " turns: " + str(longest))

    print("Turns with more than " + str(more_than_words) + " words: " + str(turns_with_more_than_x_words))

    print("Turns with less than " + str(less_than_words) + " words: " + str(turns_with_less_than_x_words))

    type_length, type_num = max(sentence_lengths.iteritems(), key=operator.itemgetter(1))

    print("There are most turns with length: " + str(type_length) + ". Num turns: " + str(type_num))

    print("Average length of turn: " + str(int(num_words/num_turns)))


def get_bucket_stats(path, buckets=[(40, 40), (60, 60), (85, 85), (110, 110), (150, 150)]):

    print("\n############## Stats for " + path + " ##############")
    bucket_content = [0 for _ in buckets]
    total_lines = 0
    print("Buckets", buckets)

    with open(path) as file_object:
        for line in file_object:
            total_lines += 1
            # Find correct bucket
            x, y = line.split(',')
            for bucket_id, (source_size, target_size) in enumerate(buckets):
                if len(x) < source_size and len(y) < target_size:
                    bucket_content[bucket_id] += 1
                    break

    print("Occurrences in each bucket: " + str(bucket_content))

    all_turns = sum(bucket_content)
    bucket_content = ["{0:.2f}".format((100.0*num) / total_lines) + "%" for num in bucket_content]

    print("Occurrences in each bucket by percent: " + str(bucket_content))

    no_match = total_lines - all_turns

    print("Number of turns that did not fit: " + str(no_match) + "/" + str(total_lines) + "\t = " +
          "{0:.2f}".format((100.0*no_match)/total_lines) + "%")


def get_dictionary_stats(x_train, y_train, occurrence=10000):
    print("\n############## Stats for Vocabulary ##############")
    dictionary = {}

    print("Finding words in " + str(x_train))
    with open(x_train) as fileobject:
        for line in fileobject:
            sentence = line.strip().split(' ')
            for word in sentence:
                if word in dictionary:
                    dictionary[word] += 1
                else:
                    dictionary[word] = 1

    print("Finding words in " + str(y_train))
    with open(y_train) as fileobject:
        for line in fileobject:
            sentence = line.strip().split(' ')
            for word in sentence:
                if word in dictionary:
                    dictionary[word] += 1
                else:
                    dictionary[word] = 1

    sorted_dict = sorted(dictionary.items(), key=operator.itemgetter(1), reverse = True)
    counter = 0

    words_represented = 0
    words_not_represented = 0

    for key, value in sorted_dict:
        if counter < 100000:
            words_represented += value
        else:
            words_not_represented += value
        if counter % occurrence == 0:
            print(counter, key, value)
        counter += 1

    # Removing the _EOS_ token. The other tokens are added later
    words_represented -= sorted_dict[0][1]

    print("Words in dictionary occurs " + str(words_represented) + " times")
    print("Words not in dictionary occurs " + str(words_not_represented) + " times")
    print("UNK-token is represented as " + str("{0:.2f}".format(words_not_represented/(words_represented + words_not_represented))) +
          "% of all words")


def get_number_of_urls(path):
    urls = []
    with open(path) as file_object:
        for line in file_object:
            new_urls = re.findall(r'(https?://[^\s]+)', line)
            urls.extend(new_urls)

    print("Number of URLs in " + path + ": " + str(len(urls)))


def estimate_vector_similarity_time(num_known_words=100000, dimension=100, real_number_of_unk_words=1260513, fraction=10):
    # Create random known_words:
    known_words = {"known_"+str(i): (np.array([random.random()])*dimension, random.random()*random.randint(1, 10)) for i in range(num_known_words)}

    # Create random part of unk_words
    part_of_dict = {"unk_"+str(i): np.array([random.random()])*dimension for i in range(fraction)}

    print("   Placeholders created and fed")

    multiplier = real_number_of_unk_words/fraction

    start_time = time.time()
    counter = 1
    for unk_key, unk_values in part_of_dict.iteritems():
        min_dist = 1
        word = ""
        for key, value in known_words.iteritems():
            cur_dist = distance(unk_values, value[0], value[1])
            if cur_dist < min_dist:
                min_dist = cur_dist
                word = key
        part_of_dict[unk_key] = word
        print("   Estimation " + str(counter) + "/" + str(fraction) + " done")
        counter += 1

    est_time = (time.time() - start_time) * multiplier

    m, s = divmod(est_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)

    print("Estimated time for calculating " + str(real_number_of_unk_words) + " unknown words, with " +
          str(len(known_words)) + "-dictionary and " + str(dimension) +
          "-word embedding: %d days %d hours %02d minutes %02d seconds" % (d, h, m, s))


def get_emojis(path):
    smiley_pattern = r'(?::|;|=)(?:-)?(?:\)|\(|D|P|\|)'  # NB: will take :) from /:) and :)D
    smiley_pattern = r'(?::|;|=)(?:-)?(?:\)|\(|D|P|\|)(?=$|\s)' # NB: Will take :) from /:) but not from :)D
    smiley_pattern = r'((?:^|\s)(?::|;|=)(?:-)?(?:\)|\(|D|P|\|)(?=$|\s))' # NB: Will neither take :) from /:) nor from :)D
    smiles = []
    with open(path) as file_object:
        for line in file_object:
            smiley = re.findall(smiley_pattern, line)
            smiles.extend(smiley)

    print("Number of smiles in " + path + ": " + str(len(smiles)))

    # Counting every different smiley
    smiley_dict = {}
    for s in smiles:
        if s in smiley_dict:
            smiley_dict[s] += 1
        else:
            smiley_dict[s] = 1

    for key, value in smiley_dict.items():
        print(key, value)


def get_unknown_words_stats(vocab_size=100000):
    sorted_dict = find_dictionary(x_train='./datafiles/spell_checked_data_x.txt', y_train='./datafiles/spell_checked_data_y.txt')
    print("Known and unknown words found")
    known_words = sorted_dict[:vocab_size]
    unknown_words = sorted_dict[vocab_size:]

    print("Number of words in vocabulary %d" % (len(known_words)))
    print("Number of unknown words %d" % (len(unknown_words)))

    print(known_words[:10])
    print(unknown_words[-10:])


# get_stats('x_train.txt', more_than_words=40, less_than_words=6)
# get_stats('y_train.txt', more_than_words=50, less_than_words=11)
# get_bucket_stats('train_merged.txt', buckets=[(5, 10), (10, 15), (20, 25), (40, 50)])
# get_dictionary_stats('./datafiles/no_unk_words_x.txt', './datafiles/no_unk_words_y.txt')
# get_number_of_urls('./datafiles/x_train_spell_check.txt')
# get_number_of_urls('./datafiles/y_train_spell_check.txt')
# get_emojis('./datafiles/x_train_spell_check.txt')
# estimate_vector_similarity_time()
get_unknown_words_stats(vocab_size=40000)

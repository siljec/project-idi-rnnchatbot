import operator
import re
import time
import random
import numpy as np
from preprocessing3 import distance
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


def get_number_of_urls(path):
    urls = []
    with open(path) as file_object:
        for line in file_object:
            new_urls = re.findall(r'(https?://[^\s]+)', line)
            urls.extend(new_urls)

    print("Number of URLs in " + path + ": " + str(len(urls)))


def estimate_vector_similarity_time_dict(num_known_words=100000, dimension=100, real_number_of_unk_words=1260513, fraction=10):
    # Create random known_words:
    known_words = {"known_"+str(i): (np.array([random.random()]*dimension), random.random()*random.randint(1, 10)) for i in range(num_known_words)}

    # Create random part of unk_words
    part_of_dict = {"unk_"+str(i): np.array([random.random()]*dimension) for i in range(fraction)}

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


def estimate_vector_similarity_time_list(num_known_words=100000, dimension=100, real_number_of_unk_words=1260513, fraction=10):
    # Create random known_words:

    known_words = [("known_"+str(i), np.array([random.random()]*dimension), random.random()*random.randint(1, 10)) for i in range(num_known_words)]

    # print(known_words[0])

    # Create random part of unk_words
    part_of_dict = [("unk_"+str(i), np.array([random.random()]*dimension)) for i in range(fraction)]
    # part_of_dict = np.array(part_of_dict)
    # Placeholder for results
    result_dict = {}

    print("   Placeholders created and fed")

    multiplier = real_number_of_unk_words/fraction
    start_time = time.time()
    counter = 1
    for unk_key, unk_values in part_of_dict:
        min_dist = 1
        word = ""
        for key, value, dis in known_words:
            cur_dist = distance(unk_values, value, dis)
            if cur_dist < min_dist:
                min_dist = cur_dist
                word = key
        result_dict[unk_key] = word
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


def get_unknown_words_stats(vocab_size=100000, occurrence=10000):
    sorted_dict = find_dictionary(x_train='./datafiles/spell_checked_data_x.txt', y_train='./datafiles/spell_checked_data_y.txt')
    print("Known and unknown words found")
    known_words = sorted_dict[:vocab_size]
    unknown_words = sorted_dict[vocab_size:]


    num_words_5 = 0
    num_words_4 = 0
    num_words_3 = 0
    num_words_2 = 0
    num_words_1 = 0

    words_represented = 0
    words_not_represented = 0
    counter = 0

    for key, value in sorted_dict:
        if counter < vocab_size:
            words_represented += value
        else:
            words_not_represented += value
            if value == 5:
                num_words_5 += value
            elif value == 4:
                num_words_4 += value
            elif value == 3:
                num_words_3 += value
            elif value == 2:
                num_words_2 += value
            elif value == 1:
                num_words_1 += value
        if counter % occurrence == 0:
            print(counter, key, value)
        counter += 1

    all_words = words_represented + words_not_represented

    print("\nNumber of unique words in vocabulary %i (Total occurrences: %i)" % (len(known_words), words_represented))
    print("Number of unique unknown words %i (Total occurrences: %i)" % (len(unknown_words), words_not_represented))

    # Removing the _EOS_ token. The other tokens are added later
    # all_words -= sorted_dict[0][1]

    print("\nUNK-token is represented as %.2f%% of all words" % (100.0 * words_not_represented / all_words))
    print("UNK-token with frequency 5 is represented as %.2f%% of all words (%i unique words)" % ((100.0 * num_words_5 / all_words), num_words_5/5))
    print("UNK-token with frequency 4 is represented as %.2f%% of all words (%i unique words)" % ((100.0 * num_words_4 / all_words), num_words_4/4))
    print("UNK-token with frequency 3 is represented as %.2f%% of all words (%i unique words)" % ((100.0 * num_words_3 / all_words), num_words_3/3))
    print("UNK-token with frequency 2 is represented as %.2f%% of all words (%i unique words)" % ((100.0 * num_words_2 / all_words), num_words_2/2))
    print("UNK-token with frequency 1 is represented as %.2f%% of all words (%i unique words)" % ((100.0 * num_words_1 / all_words), num_words_1/1))

    print("Words in dictionary occurs " + str(words_represented) + " times")
    print("Words not in dictionary occurs " + str(words_not_represented) + " times")
    print("UNK-token is represented as " + str("{0:.2f}".format(words_not_represented/(words_represented + words_not_represented))) +
          "% of all words")

def get_unique_words(x_path, y_path):
    words_set = set()

    with open(x_path) as file_object:
        for line in file_object:
            words = line.split()
            for word in words:
                words_set.add(word)
    with open(y_path) as file_object:
        for line in file_object:
            words = line.split()
            for word in words:
                words_set.add(word)
    print("Unique words in " + x_path + " and " + y_path + " is %i" %(len(words_set)))

def get_number_of_turns(x_path, y_path):
    turns = 0
    with open(x_path) as file_object:
        for _ in file_object:
            turns += 1
    with open(y_path) as file_object:
        for _ in file_object:
            turns += 1
    print(str(turns) + " number of turns in " + x_path + " and " + y_path)

def get_all_words(x_path, y_path):
    all_words = 0
    with open(x_path) as file_object:
        for line in file_object:
            words = line.split()
            for word in words:
                all_words += 1
    with open(y_path) as file_object:
        for line in file_object:
            words = line.split()
            for word in words:
                all_words += 1
    print(str(all_words) + " words in " + x_path + " and " + y_path)

def get_number_of_urls(path):
    urls = []
    with open(path) as file_object:
        for line in file_object:
            new_urls = re.findall(r'(https?://[^\s]+)', line)
            urls.extend(new_urls)

    print("Number of URLs in " + path + ": " + str(len(urls)))


def get_emojis(path):
    smiley_pattern = r'(?::|;|=)(?:-)?(?:\)|\(|D|P|\|)'  # NB: will take :) from /:) and :)D
    smiley_pattern = r'(?::|;|=)(?:-)?(?:\)|\(|D|P|\|)(?=$|\s)' # NB: Will take :) from /:) but not from :)D
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


def find_percentage_of_vocab_size(x_path, y_path, percentage):
    vocab = find_dictionary(x_path, y_path)

    all_words = 0.0

    for word, occurrences in vocab:
        all_words += occurrences

    vocab_size = 0
    vocab_occurrences = 0.0
    for word, occurrences in vocab:
        vocab_occurrences += occurrences
        vocab_size += 1
        if (vocab_occurrences / all_words) >= percentage:
            break

    print("Need vocabulary size %i to cover %f of the dataset (%i / %i)" %(vocab_size, percentage, vocab_occurrences, all_words))

# get_stats('x_train.txt', more_than_words=40, less_than_words=6)
# get_stats('y_train.txt', more_than_words=50, less_than_words=11)
# get_bucket_stats('train_merged.txt', buckets=[(5, 10), (10, 15), (20, 25), (40, 50)])
# get_dictionary_stats('./datafiles/no_unk_words_x.txt', './datafiles/no_unk_words_y.txt')
# get_number_of_urls('./datafiles/spell_checked_data_x.txt')
# get_number_of_urls('./datafiles/spell_checked_data_y.txt')
# get_emojis('./datafiles/spell_checked_data_x.txt')
# estimate_vector_similarity_time_dict()
# estimate_vector_similarity_time_list()
# get_unknown_words_stats(vocab_size=20000)
# get_dictionary_stats('./spell_checked_data_x.txt', './spell_checked_data_y.txt')
# get_number_of_urls('./spell_checked_data_x.txt')
# get_number_of_urls('./spell_checked_data_y.txt')
# get_emojis('./spell_checked_data_x.txt')
#find_percentage_of_vocab_size("./datafiles/spell_checked_data_x.txt", "./datafiles/spell_checked_data_y.txt", 0.95)
#get_unique_words('./datafiles/bucket_data_x.txt','./datafiles/bucket_data_y.txt')
get_number_of_turns('./datafiles/raw_data_x.txt', './datafiles/raw_data_y.txt')
get_all_words('./datafiles/raw_data_x.txt', './datafiles/raw_data_y.txt')
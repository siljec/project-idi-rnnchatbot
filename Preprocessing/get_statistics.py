import operator
import os, sys
import time
import random
from preprocessing3 import distance
from create_vocabulary import find_dictionary
from itertools import izip
from preprocess_helpers import split_line, get_time
sys.path.insert(0, '../')
from variables import folders

# Create histogram
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import plotly.plotly as py  # tools to communicate with Plotly's server
import plotly
plotly.tools.set_credentials_file(username='siljech', api_key='eUYFnzxp2LySOfqBJLmy')
from plotly.graph_objs import *

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
   # for key, item in sentence_lengths.iteritems():
    #    print(item)
     #   if key == 50:
      #      break
    #print(sentence_lengths)

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
            x = len(x.split())
            y = len(y.split())

            for bucket_id, (source_size, target_size) in enumerate(buckets):
                if x < source_size and y < target_size:
                    bucket_content[bucket_id] += 1
                    break

    print("Occurrences in each bucket: " + str(bucket_content))

    all_turns = sum(bucket_content)
    bucket_content = ["{0:.2f}".format((100.0*num) / total_lines) + "%" for num in bucket_content]

    print("Occurrences in each bucket by percent: " + str(bucket_content))

    no_match = total_lines - all_turns

    print("Number of turns that did not fit: " + str(no_match) + "/" + str(total_lines) + "\t = " +
          "{0:.2f}".format((100.0*no_match)/total_lines) + "%")


def get_size_of_bucket_sizes(max_bucket, x_path, y_path):
    occurrences_of_pairs_in_buckets = [0]*(max_bucket+1)

    with open(x_path) as sourceObjectX, open(y_path) as sourceObjectY:
        for x, y in izip(sourceObjectX, sourceObjectY):
            x_words = x.split()
            y_words = y.split()
            x_len = len(x_words)
            y_len = len(y_words)
            if (x_len <= max_bucket and y_len <= max_bucket):
                max_size = max(len(x_words), len(y_words))
                occurrences_of_pairs_in_buckets[max_size]+=1
    print('Occurences of pairs in buckets')
    for i in range(len(occurrences_of_pairs_in_buckets)):
        print(occurrences_of_pairs_in_buckets[i])


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
    sorted_dict = find_dictionary(x_train='./opensubtitles/spell_checked_data_x.txt', y_train='./opensubtitles/spell_checked_data_y.txt')
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


def get_word_histogram(x_path, y_path):
    vocab = find_dictionary(x_path, y_path)
    vocab = vocab[5000:50000]
    occurences = []
    for _, occurence in vocab:
        occurences.append(occurence)
    trace1 = {

        "y": occurences,
        "marker": {"color": "rgb(127,205,187)"},
        "orientation": "v",
        "type": "bar"
    }
    data = Data([trace1])
    fig = Figure(data=data)
    plot_url = py.plot(fig)

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



def get_conversation_turn_stats(folders, bucket_size = 30, max_turns=3000):
    start_time = time.time()
    number_of_files_read = 0
    # Array with occurrences of conversations with i turns. Array[i] get #conversations with i turns
    turns = [0] * max_turns
    exceeded_max_turns = 0
    total_turns = 0.0
    longest_conversation = 0

    more_than_500 = 0
    more_than_1000 = 0
    more_than_2000 = 0

    for folder in folders:
        folder_path = "../../ubuntu-ranking-dataset-creator/src/dialogs/" + folder
        for filename in os.listdir(folder_path):
            number_of_files_read += 1
            file_path = folder_path + "/" + filename
            num_turns = get_num_turns_in_file(file_path, bucket_size, max_turns)
            if num_turns > 2000:
                more_than_2000 += 1
            elif (num_turns > 1000):
                more_than_1000 += 1
            elif num_turns > 500:
                more_than_500 += 1

            if num_turns < max_turns:
                turns[num_turns] +=1
                total_turns += num_turns
            else:
                exceeded_max_turns += 1
        print("Done with folder: " + str(folder) + ", read " + str(number_of_files_read) + " conversations")
    print "Number of files read: " + str(number_of_files_read)

    # Find stats
    max_occ = 0
    max_index = 0
    for i in range(2,   max_turns):
        if turns[i] > max_occ:
            max_occ = turns[i]
            max_index = i
    print("Mode is " + str(max_index) + " with " + str(max_occ) + " occurrences")

    print("Average turns per conversation is " + str(total_turns/number_of_files_read))
    #print("Average turns per conversation without two turns is " + str((total_turns-2*turns[2])/(number_of_files_read-turns[2])))

    for occurrence in turns:
        print(occurrence)
    print("Exceeded max turns " + str(exceeded_max_turns))
    print("More than 2000 " + str(more_than_2000))
    print("More than 1000 " + str(more_than_1000))
    print("More than 500 " + str(more_than_500))
    print get_time(start_time)



def get_num_turns_in_file(path, bucket_size, max_turns):
    user1_first_line = True
    num_turns = 0
    exceeds_bucket_size = False

    x_len = 0
    y_len = 0
    with open(path) as fileobject:
        for line in fileobject:
            text, current_user = split_line(line)

            # Which folders has only two turns and fits in our bucket
            num_words = len(text.split())
            if num_words > bucket_size:
                exceeds_bucket_size = True

            if text == "":
                continue
            if user1_first_line:
                init_user, previous_user = current_user, current_user
                user1_first_line = False
                num_turns = 0
            # The user is still talking
            if current_user == previous_user:
                if current_user == init_user:
                    x_len += num_words
                else:
                    y_len += num_words
            # A new user responds
            else:

                if y_len != 0 and x_len <=bucket_size and y_len <=bucket_size:
                    num_turns += 2
                # reset lengths if we are done with user 2, i.e. start of a new training pair
                if y_len != 0:
                    x_len = 0
                    y_len = 0
                if current_user == init_user:
                    x_len = num_words
                else:
                    y_len = num_words
            previous_user = current_user
    # if num_turns < 3:
    #     if not exceeds_bucket_size:
    #         print(path)
    if num_turns > max_turns:
        print("Path " + path + "exceeds " + str(max_turns) + " turns")
    return num_turns



def non_turns_exceed_max_turns_in_conv(file_path, fit_1, fit_2, fit_3, fit_4, fit_5, fit_6):
    go_token = ""
    eos_token = " . "
    eot_token = ""

    ending_symbols = tuple(["!", "?", ".", "_EMJ", "_EMJ "])
    user1_first_line = True


    x_train = []
    y_train = []
    num_turns = 0
    fit_1_bool = True
    fit_2_bool = True
    fit_3_bool = True
    fit_4_bool = True
    fit_5_bool = True
    fit_6_bool = True

    sentence_holder = ""
    with open(file_path) as fileobject:
        for line in fileobject:
            text, user = split_line(line)
            if text == "":
                continue
            current_user = user
            if user1_first_line:
                init_user, previous_user = current_user, current_user
                user1_first_line = False
                sentence_holder = ""

            if current_user == previous_user:  # The user is still talking
                if text.endswith(ending_symbols):
                    sentence_holder += text + " "
                else:
                    sentence_holder += text + eos_token
            else:  # A new user responds
                sentence_holder += eot_token + "\n"

                if current_user == init_user:  # Init user talks (should add previous sentence to y_train)
                    y_train.append(sentence_holder)
                    words = sentence_holder.split()
                    num_words = len(words)
                    if num_words > fit_1:
                        fit_1_bool = False
                    if num_words > fit_2:
                        fit_2_bool = False
                    if num_words > fit_3:
                        fit_3_bool = False
                    if num_words > fit_4:
                        fit_4_bool = False
                    if num_words > fit_5:
                        fit_5_bool = False
                    if num_words > fit_6:
                        fit_6_bool = False
                else:
                    x_train.append(sentence_holder)
                    words = sentence_holder.split()
                    num_words = len(words)
                    if num_words > fit_1:
                        fit_1_bool = False
                    if num_words > fit_2:
                        fit_2_bool = False
                    if num_words > fit_3:
                        fit_3_bool = False
                    if num_words > fit_4:
                        fit_4_bool = False
                    if num_words > fit_5:
                        fit_5_bool = False
                    if num_words > fit_6:
                        fit_6_bool = False
                if text.endswith(ending_symbols):
                    sentence_holder = go_token + text + " "
                else:
                    sentence_holder = go_token + text + eos_token

            previous_user = current_user

    if current_user != init_user:
        y_train.append(sentence_holder + eot_token + "\n")
        words = sentence_holder.split()
        num_words = len(words)
        if num_words > fit_1:
            fit_1_bool = False
        if num_words > fit_2:
            fit_2_bool = False
        if num_words > fit_3:
            fit_3_bool = False
        if num_words > fit_4:
            fit_4_bool = False
        if num_words > fit_5:
            fit_5_bool = False
        if num_words > fit_6:
            fit_6_bool = False

    y_len = len(y_train)
    num_turns = len(x_train) + y_len
    if (y_len == len(x_train)):
        return num_turns, fit_1_bool, fit_2_bool, fit_3_bool, fit_4_bool, fit_5_bool, fit_6_bool
    else:
        print("different y and x len")
        return 0, False, False, False, False, False, False

def get_conversation_stats_for_context(folders, fit_1, fit_2, fit_3, fit_4, fit_5, fit_6):
    start_time = time.time()
    number_of_files_checked = 0
    fits_1_conv = 0
    fits_1_turns= 0
    fits_2_conv = 0
    fits_2_turns = 0
    fits_3_conv = 0
    fits_3_turns = 0
    fits_4_conv = 0
    fits_4_turns = 0
    fits_5_conv = 0
    fits_5_turns = 0
    fits_6_conv = 0
    fits_6_turns = 0

    counter = 0
    nice_files= []
    for folder in folders:
        folder_path = "../../ubuntu-ranking-dataset-creator/src/dialogs/" + folder
        for filename in os.listdir(folder_path):
            number_of_files_checked += 1
            file_path = folder_path + "/" + filename
            num_turns, fit_1_bool, fit_2_bool, fit_3_bool, fit_4_bool, fit_5_bool, fit_6_bool, = non_turns_exceed_max_turns_in_conv(file_path, fit_1, fit_2, fit_3, fit_4, fit_5, fit_6)
            if fit_1_bool:
                fits_1_conv += 1
                fits_1_turns += num_turns
                if counter < 40:
                    counter +=1
                    nice_files.append(file_path)
            if fit_2_bool:
                fits_2_conv += 1
                fits_2_turns += num_turns
            if fit_3_bool:
                fits_3_conv += 1
                fits_3_turns += num_turns
            if fit_4_bool:
                fits_4_conv += 1
                fits_4_turns += num_turns
            if fit_5_bool:
                fits_5_conv += 1
                fits_5_turns += num_turns
            if fit_6_bool:
                fits_6_conv += 1
                fits_6_turns += num_turns
        print("Done with folder: " + str(folder) + ", read " + str(number_of_files_checked) + " files")

    print "Number of files read: " + str(number_of_files_checked)
    print(str(fits_1_conv) + " conversations fits with max len " + str(fit_1) + ". Has " + str(fits_1_turns) + " turns")
    print(str(fits_2_conv) + " conversations fits with max len " + str(fit_2) + ". Has " + str(fits_2_turns) + " turns")
    print(str(fits_3_conv) + " conversations fits with max len " + str(fit_3) + ". Has " + str(fits_3_turns) + " turns")
    print(str(fits_4_conv) + " conversations fits with max len " + str(fit_4) + ". Has " + str(fits_4_turns) + " turns")
    print(str(fits_5_conv) + " conversations fits with max len " + str(fit_5) + ". Has " + str(fits_5_turns) + " turns")
    print(str(fits_6_conv) + " conversations fits with max len " + str(fit_6) + ". Has " + str(fits_6_turns) + " turns")
    for files in nice_files:
        print(files)
    print get_time(start_time)

#get_stats('./datafiles/spell_checked_data_x.txt', more_than_words=30, less_than_words=10)
#get_stats('./datafiles/spell_checked_data_y.txt', more_than_words=30, less_than_words=10)
#get_bucket_stats('./datafiles/training_data.txt', buckets=[(10, 10), (20, 20), (35, 35), (50, 50)])
# get_dictionary_stats('./datafiles/no_unk_words_x.txt', './datafiles/no_unk_words_y.txt')
# get_number_of_urls('./datafiles/spell_checked_data_x.txt')
# get_number_of_urls('./datafiles/spell_checked_data_y.txt')
# get_emojis('./datafiles/spell_checked_data_x.txt')
# estimate_vector_similarity_time_dict()
# estimate_vector_similarity_time_list()
#get_unknown_words_stats(vocab_size=30000)
# get_dictionary_stats('./spell_checked_data_x.txt', './spell_checked_data_y.txt')
# get_number_of_urls('./spell_checked_data_x.txt')
# get_number_of_urls('./spell_checked_data_y.txt')
# get_emojis('./spell_checked_data_x.txt')
# find_percentage_of_vocab_size("./opensubtitles/spell_checked_data_x.txt", "./opensubtitles/spell_checked_data_y.txt", 0.99)
# find_percentage_of_vocab_size("./opensubtitles/spell_checked_data_x.txt", "./opensubtitles/spell_checked_data_y.txt", 0.98)
# find_percentage_of_vocab_size("./opensubtitles/spell_checked_data_x.txt", "./opensubtitles/spell_checked_data_y.txt", 0.97)
# find_percentage_of_vocab_size("./opensubtitles/spell_checked_data_x.txt", "./opensubtitles/spell_checked_data_y.txt", 0.96)
# find_percentage_of_vocab_size("./opensubtitles/spell_checked_data_x.txt", "./opensubtitles/spell_checked_data_y.txt", 0.95)
#get_unknown_words_stats(20000, occurrence=3000)

#get_unique_words('./datafiles/bucket_data_x.txt','./datafiles/bucket_data_y.txt')
#get_number_of_turns('./datafiles/raw_data_x.txt', './datafiles/raw_data_y.txt')
#get_all_words('./datafiles/raw_data_x.txt', './datafiles/raw_data_y.txt')
#get_word_histogram('./datafiles/raw_data_x.txt', './datafiles/raw_data_y.txt')
#get_size_of_bucket_sizes(100, './context/bucket_data_x.txt', './context/bucket_data_y.txt')

#get_conversation_turn_stats(folders, 1000, 1000)
get_conversation_stats_for_context(folders, 30, 40, 50, 60, 70, 80)

import os, re
from create_vocabulary import read_vocabulary_from_file, encode_sentence
from random import shuffle
from itertools import izip
import operator
import pickle
import time

# Code beautify helpers

def path_exists(path):
    return os.path.exists(path)


def get_time(start_time, sufix=""):
    duration = time.time() - start_time
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    return "%d hours %d minutes %d seconds %s" % (h, m, s, sufix)

def file_len(file_name):
    with open(file_name) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def load_pickle_file(path):
    with open(path) as fileObject:
        obj = pickle.load(fileObject)
    return obj

def save_to_file(file_name, array):
    with open(file_name, 'w') as fileObject:
        for line in array:
            fileObject.write(line)

# Pre-process 1
def split_line(line):
    data = line.split("\t")
    current_user = data[1]
    text = data[3].strip().lower()  # user user timestamp text
    return text, current_user

def do_regex_on_file(path):
    url_token = "_URL"
    emoji_token = " _EMJ"
    dir_token = "_DIR"

    data = []

    with open(path) as fileObject:
        for line in fileObject:
            data.append(do_regex_on_line(line, url_token, emoji_token, dir_token))
    return data

def do_regex_on_line(line, url_token, emoji_token, dir_token):
    text = re.sub(' +', ' ', line)  # Will remove multiple spaces
    text = re.sub(r'((www\.|https?:\/\/)[^\s]+)', url_token, text)  # Exchange urls with URL token
    text = re.sub(r'((?:^|\s)(?::|;|=)(?:-)?(?:\)|\(|D|P|\|)(?=$|\s))', emoji_token, text)  # Exchange smiles with EMJ token NB: Will neither take :) from /:) nor from :)D
    text = re.sub('"', '', text)  # Remove "
    text = re.sub("(?!(')([a-z]{1})(\s))(')(?=\w|\s)", "", text)  # Remove ', unless it is like "banana's"
    text = re.sub("[^\s]+(.(org|com|edu|net|uk)(?=$|\s))", url_token, text)  # Will replace ubuntu.com with URL token
    text = re.sub('((~\/)|(\/\w+)|(\.\/\w+)|(\w+(?=(\/))))((\/)|(\w+)|(\.\w+)|(\w+|\-|\~))+', dir_token, text)  # Replace directory-paths
    text = re.sub('(?<=[a-z])([!?,.])', r' \1', text)  # Add space before special characters [!?,.]
    text = re.sub('(_EMJ \.)', '_EMJ', text)  # Remove period after EMJ token
    text = re.sub(' +', ' ', text)  # Will remove multiple spaces
    return text


def do_regex_on_line_opensubtitles(text):
    text = re.sub(' +', ' ', text)  # Will remove multiple spaces
    text = re.sub('\'\s[v][e]', ' have', text)  # i' ve - i have
    text = re.sub('\'\s[r][e]', ' are', text)   # you' re - you are
    text = re.sub('\'\s[n][t]', ' not', text)   # are' nt - are not
    text = re.sub('([j][u][s][t])(?=[^\s])', 'just ', text)   # Space after just<a-z>
    text = re.sub('([y][o][u][r])(?=[^s|\s])', 'your ', text)   # Space after your<a-z>
    text = re.sub('([y][o][u][r][s])(?=[\s])', 'yours ', text)   # Space after yours<a-z>
    text = re.sub('([w][h][y])(?=[^s|^\s])', 'why ', text)   # Space after why<a-z>
    text = re.sub('([w][h][y][s])(?=[^\s])', 'whys ', text)   # Space after whys<a-z>
    text = re.sub('(^|(?<=\s))([i][v][e])', 'i have ', text)   # ive - i have
    text = re.sub('\<([i]|\/\s([i]|[i]\s)|[m][u][s][l][c]|\s[i])\>', '', text)   # remove <i> </ i>
    text = re.sub('(\<)|(\>)', ' ', text)       # remove single <>
    text = re.sub('\#|\@', '', text)       # remove hashtags and @
    text = re.sub('\[\s(.+)\s\]', '', text)       # remove [ action ]
    text = re.sub('\'\s[m]', ' am', text)       # i' m - i am
    text = re.sub('\'\s[i][l]|\'\s[l][l]|\'\s[i][i]|\'\s[l][i]', ' will', text)  # you' ll/il - you will
    text = re.sub('[i][i](\s|$)', 'll ', text)  # caii - call
    text = re.sub('\'\s[s]', "'s", text) # remove space between <word>' s
    text = re.sub('\'\s[t]', "'t", text) # remove space between <word>' t
    text = re.sub('\'\s[d]', "'d", text) # remove space between <word>' d e.g. where'd you go?
    text = re.sub('^\'|\'$|\'( +)$|(\s+(\'+\s+)+)', ' ', text) # remove ' in the beginning/end of a sentence
    text = re.sub('\sy\'\s|^y\'', ' you ', text) # y' - you
    text = re.sub('\"', ' ', text)  # Will remove single "
    text = re.sub(' +', ' ', text)  # Will remove multiple spaces
    return text





# Pre-process 1
def replace_word_helper(candidate, dictionary):
    if candidate in dictionary:
        # print "replacing ", candidate, " with ", dictionary[candidate]
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

def replace_mispelled_words_in_file(source_file_path, new_file_path, misspelled_vocabulary):
    dictionary = read_words_from_misspelling_file(misspelled_vocabulary)
    new_file = open(new_file_path, 'w')
    with open(source_file_path) as fileobject:
        for line in fileobject:
            sentence = line.split(' ')
            last_word = sentence.pop().strip()
            for word in sentence:
                new_word = replace_word_helper(word, dictionary)
                new_file.write(new_word + ' ')
            new_word = replace_word_helper(last_word, dictionary)
            new_file.write(new_word + '\n')

    new_file.close()

# Pre-process 2

def save_vocabulary(path, obj, init_tokens):

    with open(path, 'w') as fileObject:
        for token in init_tokens:
            fileObject.write(token + '\n')

        num_words = len(obj) - 1

        for i in range(num_words-1):
            fileObject.write(obj[i][0] + '\n')

        fileObject.write(obj[num_words][0])


def save_dict_to_file(file_name, obj):
    with open(file_name, 'w') as fileObject:
        for key, value in obj.iteritems():
            fileObject.write(key + " : " + value + "\n")
    print("Dictionary saved to file " + file_name)


def save_to_pickle(pickle_name, obj):
    with open(pickle_name, 'w') as pickleObject:
        pickle.dump(obj, pickleObject)
    print("Object saved to pickle " + pickle_name)


def find_dictionary(x_train, y_train):
    dictionary = {}

    with open(x_train) as fileobject:
        for line in fileobject:
            sentence = line.strip().split(' ')
            for word in sentence:
                if word.strip() == "":
                    continue
                if word in dictionary:
                    dictionary[word] += 1
                else:
                    dictionary[word] = 1

    with open(y_train) as fileobject:
        for line in fileobject:
            sentence = line.strip().split(' ')
            for word in sentence:
                if word.strip() == "":
                    continue
                if word in dictionary:
                    dictionary[word] += 1
                else:
                    dictionary[word] = 1

    sorted_dict = sorted(dictionary.items(), key=operator.itemgetter(1), reverse = True)
    return sorted_dict


def merge_files(x_path, y_path, final_file):
    with open(x_path) as x_file, open(y_path) as y_file, open(final_file, 'w') as final_file:
        for x, y in izip(x_file, y_file):
            final_file.write(x)
            final_file.write(y)


def create_final_merged_files(x_path, y_path, vocabulary_path, train_path, val_path, test_path, val_size_fraction,
                              test_size_fraction):
    vocabulary, _ = read_vocabulary_from_file(vocabulary_path)
    train_final = open(train_path, 'w')
    val_final = open(val_path, 'w')
    test_final = open(test_path, 'w')
    num_lines = file_len(x_path)

    train_size = num_lines * (1 - val_size_fraction - test_size_fraction)
    val_size = train_size + num_lines * test_size_fraction
    line_counter = 0
    with open(x_path) as x_file, open(y_path) as y_file:
        for x, y in izip(x_file, y_file):
            if line_counter < train_size:
                train_final.write( encode_sentence(x.strip().split(" "), vocabulary) + ", " + encode_sentence(y.strip().split(" "), vocabulary) + '\n')
            elif line_counter < val_size:
                val_final.write(encode_sentence(x.strip().split(" "), vocabulary) + ", " + encode_sentence(y.strip().split(" "), vocabulary) + '\n')
            else:
                test_final.write(encode_sentence(x.strip().split(" "), vocabulary) + ", " + encode_sentence(y.strip().split(" "), vocabulary) + '\n')
            line_counter += 1
    train_final.close()
    val_final.close()
    test_final.close()


def replace_UNK_words_in_file(source_file_path, new_file_path, dictionary):
    new_file = open(new_file_path, 'w')
    with open(source_file_path) as fileobject:
        for line in fileobject:
            words = line.split(' ')
            sentence = ""
            last_word = words.pop().strip()
            for word in words:
                new_word = replace_word_helper(word, dictionary)
                sentence += new_word + ' '
            new_word = replace_word_helper(last_word, dictionary)
            new_file.write(sentence + new_word + '\n')
    new_file.close()


def shuffle_file(path, target_file):
    lines = open(path).readlines()
    shuffle(lines)
    open(target_file, 'w').writelines(lines)


# Currently not in use
# Used to shuffle the list
def get_random_folders():
    folders = os.listdir("../../ubuntu-ranking-dataset-creator/src/dialogs")
    shuffle(folders)
    new_folders = []
    for i in range(len(folders)):
        if folders[i] != ".DS_Store":
            new_folders.append(folders[i])
    return new_folders


def create_final_files(source_path, vocabulary_path, train_path, val_path, test_path, val_size_fraction,
                       test_size_fraction):
    vocabulary, _ = read_vocabulary_from_file(vocabulary_path)
    train_final = open(train_path, 'w')
    val_final = open(val_path, 'w')
    test_final = open(test_path, 'w')
    num_lines = file_len(source_path)
    train_size = num_lines * (1 - val_size_fraction - test_size_fraction)
    val_size = train_size + num_lines * test_size_fraction
    line_counter = 0
    with open(source_path) as fileobject:
        for line in fileobject:
            if line_counter < train_size:
                train_final.write(encode_sentence(line.strip().split(" "), vocabulary) + '\n')
            elif line_counter < val_size:
                val_final.write(encode_sentence(line.strip().split(" "), vocabulary) + '\n')
            else:
                test_final.write(encode_sentence(line.strip().split(" "), vocabulary) + '\n')
            line_counter += 1.0
    train_final.close()
    val_final.close()
    test_final.close()


def create_vocabulary_and_return_unknown_words(sorted_dict, vocab_path, vocab_size, init_tokens=['_PAD', '_GO', '_EOS', '_EOT', '_UNK']):
    unknown_dict = {}

    vocabulary = open(vocab_path, 'w')

    for token in init_tokens:
        vocabulary.write(token + '\n')

    counter = 0
    for key in sorted_dict:
        if counter < vocab_size:
            if key[0] not in init_tokens:
                vocabulary.write(str(key[0])+ '\n')
                counter += 1
        else:
            unknown_dict[key[0]] = 0

    vocabulary.close()

    return unknown_dict

def from_index_to_words(vocab_path, source_file_path, new_file_path):
    vocab, rev_vocab = read_vocabulary_from_file(vocab_path)
    with open(source_file_path) as source_object, open(new_file_path, 'w') as new_file:
        for line in source_object:
            x, y = line.split(', ')
            sentence = ""
            words = [rev_vocab[int(word.strip())] for word in x.split(' ')]
            for word in words:
                sentence += word + " "
            words = [rev_vocab[int(word.strip())] for word in y.split(' ')]
            sentence += " ##### "
            for word in words:
                sentence += word + " "
            sentence += "\n"
            new_file.write(sentence)
    print("File converted from integers to words in vocabulary.")

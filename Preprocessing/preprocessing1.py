import time
import re
import os
from preprocess_helpers import get_time, path_exists, save_to_file


def split_line(line):
    data = line.split("\t")
    current_user = data[1]
    text = data[3].strip().lower()  # user user timestamp text
    return text, current_user


# Reads all folders and squash into one file
def preprocess_training_file(path, x_train_path, y_train_path):
    go_token = ""
    eos_token = " . "
    eot_token = ""

    ending_symbols = tuple(["!", "?", ".", "_EMJ", "_EMJ "])

    user1_first_line = True

    x_train = []
    y_train = []

    sentence_holder = ""
    with open(path) as fileobject:
        for line in fileobject:
            text, current_user = split_line(line)
            if text == "":
                continue
            if user1_first_line:
                init_user, previous_user = current_user, current_user
                user1_first_line = False
                sentence_holder = go_token

            if current_user == previous_user:  # The user is still talking
                if text.endswith(ending_symbols):
                    sentence_holder += text + " "
                else:
                    sentence_holder += text + eos_token
            else:  # A new user responds
                sentence_holder += eot_token + "\n"

                if current_user == init_user:  # Init user talks (should add previous sentence to y_train)
                    y_train.append(sentence_holder)
                else:
                    x_train.append(sentence_holder)
                if text.endswith(ending_symbols):
                    sentence_holder = go_token + text + " "
                else:
                    sentence_holder = go_token + text + eos_token

            previous_user = current_user

    if current_user != init_user:
        y_train.append(sentence_holder + eot_token + "\n")

    with open(x_train_path, 'a') as xTrainObject, open(y_train_path, 'a') as yTrainObject:
        for i in range(len(y_train)):
            xTrainObject.write(x_train[i].strip() + "\n")
            yTrainObject.write(y_train[i].strip() + "\n")


def read_every_data_file_and_create_initial_files(folders, initial_x_file_path, initial_y_file_path):
    start_time = time.time()
    number_of_files_read = 0
    for folder in folders:
        folder_path = "../../ubuntu-ranking-dataset-creator/src/dialogs/" + folder
        for filename in os.listdir(folder_path):
            number_of_files_read += 1
            file_path = folder_path + "/" + filename
            preprocess_training_file(file_path, initial_x_file_path, initial_y_file_path)
        print("Done with folder: " + str(folder) + ", read " + str(number_of_files_read) + " files")

    print "Number of files read: " + str(number_of_files_read)
    print get_time(start_time)


# -------------------------- Step 2: Do regex --------------------------


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


# -------------------------- Step 3: Do misspellings --------------------------

def read_words_from_misspelling_file(path):
    dictionary = {}
    with open(path) as fileobject:
        for line in fileobject:
            splitted_line = line.split(' ', 1)
            wrong = splitted_line[0]
            correct = splitted_line[1].strip()
            dictionary[wrong] = correct

    return dictionary


def replace_word_helper(candidate, dictionary):
    if candidate in dictionary:
        # print "replacing ", candidate, " with ", dictionary[candidate]
        return dictionary[candidate]
    return candidate


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


def preprocess1(folders, force_create_new_files, raw_data_x_path, raw_data_y_path, regex_x_path, regex_y_path, spell_checked_data_x_path, spell_checked_data_y_path, misspellings_path):

    # Step 1: Extract dialogs from UDC all source files to x and y
    if force_create_new_files and path_exists(raw_data_x_path) and path_exists(raw_data_y_path):
        os.remove(raw_data_x_path)
        os.remove(raw_data_y_path)
    if path_exists(raw_data_x_path) and path_exists(raw_data_y_path):
        print('Source files already created')
    else:
        print('Reading all the files and create initial files...')
        read_every_data_file_and_create_initial_files(folders=folders,
                                                      initial_x_file_path=raw_data_x_path,
                                                      initial_y_file_path=raw_data_y_path)

    # Step 2: Do regex
    regex_x = do_regex_on_file(raw_data_x_path)
    save_to_file(regex_x_path, regex_x)
    regex_y = do_regex_on_file(raw_data_y_path)
    save_to_file(regex_y_path, regex_y)

    # Step 3: Misspellings
    if path_exists(spell_checked_data_x_path) and path_exists(spell_checked_data_y_path) and not force_create_new_files:
        print('Spellcheck already done')
    else:
        print('Spellchecker for the initial files, create new spell checked files...')
        replace_mispelled_words_in_file(source_file_path=regex_x_path,
                                        new_file_path=spell_checked_data_x_path,
                                        misspelled_vocabulary=misspellings_path)

        replace_mispelled_words_in_file(source_file_path=regex_y_path,
                                        new_file_path=spell_checked_data_y_path,
                                        misspelled_vocabulary=misspellings_path)


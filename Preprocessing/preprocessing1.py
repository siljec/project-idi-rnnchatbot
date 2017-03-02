import time
import re
import os
from preprocess_helpers import get_time, path_exists, save_to_file, split_line, do_regex_on_file, read_words_from_misspelling_file, replace_mispelled_words_in_file

# Extract dialogs, concatenate sentences in the same turn
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


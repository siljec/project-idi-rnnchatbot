import os
import time
from preprocess_helpers import split_line_and_do_regex, get_time


# Reads all folders and squash into one file with context
def preprocess_training_file(path, x_train_path, y_train_path):
    go_token = ""
    eos_token = " . "
    url_token = " _URL "
    emoji_token = " _EMJ "
    eot_token = ""
    dir_token = "_DIR"

    ending_symbols = tuple(["!", "?", ".", "_EMJ "])

    user1_first_line = True

    x_train = []
    y_train = []

    context = ""
    new_context = ""

    sentence_holder = ""

    with open(path) as fileobject:
        for line in fileobject:
            text, current_user = split_line_and_do_regex(line, url_token=url_token, emoji_token=emoji_token, dir_token=dir_token)
            if text == "":
                continue
            if user1_first_line:
                init_user, previous_user = current_user, current_user
                user1_first_line = False
                sentence_holder = go_token

            # CONTEXT OPTION 1: context is just the final sentence in the turn
            if current_user != init_user:
                if text.endswith(ending_symbols):
                    context = new_context
                    new_context = text + " "
                else:
                    context = new_context
                    new_context = text + eos_token

            # The user is still talking
            if current_user == previous_user:
                if text.endswith(ending_symbols):
                    sentence_holder += text + " "
                else:
                    sentence_holder += text + eos_token

            # A new user responds
            else:

                sentence_holder += eot_token

                # Init user talks (should add previous sentence to y_train)
                if current_user == init_user:
                    y_train.append(sentence_holder + "\n")
                    # CONTEXT OPTION 2: the whole sentence
                    # context = sentence_holder

                # Responding user talks
                else:
                    x_train.append(context + " " + sentence_holder + "\n")

                # Reset sentence holder with new text from new user
                if text.endswith(ending_symbols):
                    sentence_holder = text + " "
                else:
                    sentence_holder = text + eos_token

            previous_user = current_user

    if current_user != init_user:
        y_train.append(sentence_holder + eot_token + "\n")
    with open(x_train_path, 'a') as xTrainObject, open(y_train_path, 'a') as yTrainObject:
        for i in range(len(y_train)):
            xTrainObject.write(x_train[i].strip() + "\n")
            yTrainObject.write(y_train[i].strip() + "\n")


# ------------- Methods called from preprocess --------------------------------------

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

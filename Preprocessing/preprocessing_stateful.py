from itertools import izip
import os, sys, glob
import fasttext
from preprocess_helpers import split_line, do_regex_on_line, do_misspellings_on_line, read_words_from_misspelling_file, \
    save_to_file, load_pickle_file, replace_word_helper, path_exists, read_vocabulary_from_file, encode_sentence
sys.path.insert(0, '../')
from variables import folders, _buckets, tokens, paths_from_preprocessing_stateful as paths

# This script will only work for the stateful preprocessing, as we will use several files for training

# Step 1: Filter out conversations that fits the bucket size (init: (30,30)), then save the file to the stateful folder

def read_every_source_file_and_save_to_dest(dest_path):
    fasttext_dictionary = load_pickle_file(paths['unk_to_vocab_pickle_path'])
    number_of_files_read = 0
    file_name = 1
    misspellings_dictionary = read_words_from_misspelling_file(paths['misspellings_path'])
    for folder in folders:
        folder_path = paths['source_folder_root'] + folder
        for filename in glob.glob(os.path.join(folder_path, '*')):
            number_of_files_read += 1
            file_path = filename  # folder_path + "/" + filename
            increment = preprocess_on_stateful(file_path, _buckets[-1][0], file_name, misspellings_dictionary, dest_path, fasttext_dictionary)
            if increment:
                file_name += 1

        print("Done with folder: " + str(folder) + ", read " + str(number_of_files_read) + " conversations")
    print "Number of files created: " + str(file_name)
    return file_name


def preprocess_on_stateful(path, bucket_size, file_name_number, misspellings_dictionary, dest_path, fasttext_dictionary):
    go_token = ""
    eos_token = " . "
    eot_token = ""

    ending_symbols = tuple(["!", "?", ".", "_EMJ", "_EMJ "])

    exceeds_bucket_size = False
    user1_first_line = True

    x_train = []
    y_train = []

    sentence_holder = ""

    with open(path) as fileobject:
        for line in fileobject:
            text, user = split_line(line)
            text = do_regex_on_line(text, tokens['url'][0], " " + tokens['emoji'][0], tokens['directory'][0])
            text = do_misspellings_on_line(text, misspellings_dictionary)

            line_words = text.split()
            num_words = len(line_words) + len(sentence_holder.split())

            # If a sentence is more than 30 (biggest bucket) words, the entire conversation should be skipped
            if num_words > bucket_size:
                exceeds_bucket_size = True
                break

            # Empty sentences should be skipped
            if text == "":
                continue
            else:
                current_user = user
            # Replace unknown words
            for word in line_words:
                replace_word_helper(word, fasttext_dictionary)

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

    if not exceeds_bucket_size and y_train != [] and x_train != []:
        if current_user != init_user:
            y_train.append(sentence_holder + eot_token + "\n")
        save_to_file(dest_path + str(file_name_number)+"_x.txt", x_train)
        save_to_file(dest_path + str(file_name_number)+"_y.txt", y_train)
        return True
    return False


def load_fasttext_model(path):
    if path_exists(path):
        print("Load existing FastText model...")
        model = fasttext.load_model(path, encoding='utf-8')
    else:
        raise ImportError("Fast text model does not exists")
    return model


def convert_word_files_to_to_int_words(source_folder, dest_path, num_conversations):
    num_dev_files = num_conversations * 0.1
    num_test_files = num_conversations * 0.1
    num_train_files = num_conversations - (num_dev_files + num_test_files)

    conversations_read = 0
    vocabulary, _ = read_vocabulary_from_file(paths['vocabulary_txt_path'])
    filenames = glob.glob(os.path.join(source_folder, '*'))
    for i in range(0, len(filenames), 2):
        # x_file
        x_path = filenames[i]
        # y_file
        y_path = filenames[i].replace('x.', 'y.')

        if num_train_files > conversations_read:
            target_file_path = dest_path + "train" + os.path.basename(filenames[i]).replace('_x', '')
        elif num_train_files + num_dev_files > conversations_read:
            target_file_path = dest_path + "dev" + os.path.basename(filenames[i]).replace('_x', '')
        else:
            target_file_path = dest_path + "test" + os.path.basename(filenames[i]).replace('_x', '')

        create_encoded_file(x_path, y_path, vocabulary, target_file_path)
        conversations_read += 1


def create_encoded_file(x_path, y_path, vocabulary, train_path):
    train_file = open(train_path, 'w')

    with open(x_path) as x_file, open(y_path) as y_file:
        for x, y in izip(x_file, y_file):
            train_file.write(encode_sentence(x.strip().split(" "), vocabulary) + ", " + encode_sentence(y.strip().split(" "), vocabulary) + '\n')

    # We use EOT token to represent end of conversation, as the token is not in use elsewhere
    train_file.write(str(tokens['eot'][1]) + ", " + str(tokens['eot'][1]))
    train_file.close()


num_files_created = read_every_source_file_and_save_to_dest(paths['stateful_raw_files'])
convert_word_files_to_to_int_words(paths['stateful_raw_files'], paths['stateful_datafiles'], num_files_created)

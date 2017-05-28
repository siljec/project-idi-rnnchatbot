import time
import sys
from preprocessing1 import preprocess1
from preprocessing2 import preprocessing2
from preprocessing3 import preprocessing3
from preprocessing1_context import preprocess1_context
from preprocessing1_contextFullTurns import preprocess1_contextFullTurns
from preprocessing1_opensubtitles import preprocess1_opensubtitles
import tensorflow as tf


from preprocess_helpers import path_exists, shuffle_file, create_final_merged_files, from_index_to_words

sys.path.insert(0, '../')
from variables import tokens_init_list, _buckets, paths_from_preprocessing as paths, vocabulary_size
from variables import contextFullTurns, context, opensubtitles, folders

""" FILL IN CORRECT INFO """
force_create_new_files = False
force_train_fast_model_all_over = False
""" ------------------- """

tf.app.flags.DEFINE_boolean("context", context, "Set to True for context.")
tf.app.flags.DEFINE_boolean("context_full_turns", contextFullTurns, "Set to True for contextFullTurns.")
tf.app.flags.DEFINE_boolean("open_subtitles", opensubtitles, "Set to True for openSubtitles.")

# Find correct path
folder = "datafiles"

if context:
    from variables import paths_from_preprocessing_context as paths
if contextFullTurns:
    from variables import paths_from_preprocessing_contextFullTurns as paths
buckets = [_buckets[-1]]  # Only need the biggest bucket
if opensubtitles:
    from variables import paths_from_preprocessing_opensubtitles as paths
    folder = "opensubtitles"




vocab_size = vocabulary_size - len(tokens_init_list)  # Minus number of init tokens
save_frequency_unk_words = 50000
val_size_fraction = 0.1
test_size_fraction = 0.1





def start_preprocessing():
    print("-------------------- INFORMATION --------------------")
    print("Force create new files: " + str(force_create_new_files))
    print("Force train fast model: " + str(force_train_fast_model_all_over))
    print("Context: " + str(context))
    print("ContextFullTurns: " + str(contextFullTurns))
    print("Opensubtitles: " + str(opensubtitles))
    print("Vocabulary size: " + str(vocabulary_size))
    print("Folder: " + str(folder))
    print("Will start preprocessing in 4 seconds")
    time.sleep(4)

    print("-------------------- PARAMETERS ---------------------")
    print("Vocabulary size: %i" % (vocab_size + len(tokens_init_list)))
    if not opensubtitles:
        print("Read number of folders: %i" % len(folders))
    print("-----------------------------------------------------\n")

    # Step 1
    if context:
        preprocess1_context(folders, force_create_new_files, paths['raw_data_x_path'], paths['raw_data_y_path'],
                            paths['regex_x_path'], paths['regex_y_path'], paths['spell_checked_data_x_path'],
                            paths['spell_checked_data_y_path'], paths['misspellings_path'])
    elif contextFullTurns:
        preprocess1_contextFullTurns(folders, force_create_new_files, paths['raw_data_x_path'], paths['raw_data_y_path'],
                            paths['regex_x_path'], paths['regex_y_path'], paths['spell_checked_data_x_path'],
                            paths['spell_checked_data_y_path'], paths['misspellings_path'])
    elif opensubtitles:
        preprocess1_opensubtitles(paths['spell_checked_data_x_path'],
                            paths['spell_checked_data_y_path'], paths['misspellings_path'])
    else:
        preprocess1(folders, force_create_new_files, paths['raw_data_x_path'], paths['raw_data_y_path'],
                    paths['regex_x_path'], paths['regex_y_path'], paths['spell_checked_data_x_path'],
                    paths['spell_checked_data_y_path'], paths['misspellings_path'])

    # Step 2
    fast_text_model = preprocessing2(paths['spell_checked_data_x_path'], paths['spell_checked_data_y_path'],
                                     paths['fast_text_train_path'], folder, force_train_fast_model_all_over)

    # Step 3
    preprocessing3(buckets, paths['spell_checked_data_x_path'], paths['spell_checked_data_y_path'],
                   paths['bucket_data_x_path'], paths['bucket_data_y_path'], vocab_size, paths['vocabulary_txt_path'],
                   paths['vocabulary_pickle_path'], fast_text_model, paths['vocab_vectors_path'],
                   paths['unk_vectors_path'], paths['unk_to_vocab_pickle_path'], paths['unk_to_vocab_txt_path'],
                   save_frequency_unk_words, paths['final_data_x_path'], paths['final_data_y_path'], tokens_init_list)


    # Step 4
    # Creating final files. X and Y are separated with comma (,))
    if force_create_new_files or not path_exists(paths['unshuffled_training_data']):
        print('Creating final merged files')
        create_final_merged_files(paths['final_data_x_path'], paths['final_data_y_path'],
                                  paths['vocabulary_txt_path'], paths['unshuffled_training_data'],
                                  paths['unshuffled_validation_data'], paths['unshuffled_test_data'],
                                  val_size_fraction, test_size_fraction)

    if force_create_new_files or not path_exists(paths['training_data']):
        print('Shuffle files')
        shuffle_file(paths['unshuffled_training_data'], paths['training_data'])
        shuffle_file(paths['unshuffled_validation_data'], paths['validation_data'])
        shuffle_file(paths['unshuffled_test_data'], paths['test_data'])

        from_index_to_words(paths['vocabulary_txt_path'], paths['test_data'], paths['test_file_words_path'])

#start_preprocessing()
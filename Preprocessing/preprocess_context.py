from preprocess_helpers import read_every_data_file_and_create_initial_files, merge_files, create_fast_text_model, \
    get_most_similar_words, replace_UNK_words_in_file, create_final_merged_files, shuffle_file, path_exists, \
    get_most_similar_words_for_UNK, save_dict_to_file
from spell_error_fix import replace_mispelled_words_in_file
from create_vocabulary import find_dictionary, create_vocabulary_and_return_unknown_words
import fasttext
import os
import pickle
import time

# --------- All paths -----------------------------------------------------------------------------------
force_create_new_files = False
force_train_fast_model_all_over = False

print("-------------------- INFORMATION --------------------")
print("Force create new files: " + str(force_create_new_files))
print("Force train fast model: " + str(force_train_fast_model_all_over))
print("Will start preprocessing in 4 seconds")
print("-----------------------------------------------------\n")
time.sleep(4)

tokens = ['_PAD', '_GO', '_EOS', '_EOT', '_UNK', '_URL', '_EMJ', '_DIR']

vocab_size = 100000 - len(tokens)  # Minus number of tokens
val_size_fraction = 0.1
test_size_fraction = 0.1

source_folder_root = "../../ubuntu-ranking-dataset-creator/src/dialogs/"
squashed_source_data_x = "./datafiles/squashed_source_data_x.txt"
squashed_source_data_y = "./datafiles/squashed_source_data_y.txt"

spell_checked_data_x = "./datafiles/spell_checked_data_x.txt"
spell_checked_data_y = "./datafiles/spell_checked_data_y.txt"
spell_checked_vocabulary = "./datafiles/misspellings.txt"

fast_text_training_data = "./datafiles/fast_text_training_data.txt"

no_unk_words_x = "./datafiles/no_unk_words_x.txt"
no_unk_words_y = "./datafiles/no_unk_words_y.txt"

unshuffled_training_data = "./datafiles/unshuffled_training_data.txt"
unshuffled_validation_data = "./datafiles/unshuffled_validation_data.txt"
unshuffled_test_data = "./datafiles/unshuffled_test_data.txt"

training_data = "./datafiles/training_data.txt"
validation_data = "./datafiles/validation_data.txt"
test_data = "./datafiles/test_data.txt"

vocabulary = "./datafiles/vocabulary.txt"


# --------- Folders to loop -----------------------------------------------------------------------------

folders = ['30', '356', '195', '142', '555', '43', '50', '36', '46', '85', '41', '118', '166', '104', '471', '37',
           '115', '47', '290', '308', '191', '457', '32', '231', '45', '133', '222', '213', '89', '92', '374', '98',
           '219', '25', '21', '182', '140', '129', '264', '132', '258', '243', '42', '456', '301', '9', '269', '88',
           '211', '123', '112', '23', '149', '105', '145', '39', '287', '249', '66', '51', '305', '241', '136',
           '57', '174', '245', '407', '17', '281', '205', '235', '383', '38', '183', '2', '521', '408', '18', '347',
           '74', '392', '334', '56', '156', '278', '230', '14', '265', '194', '187', '77', '163', '479', '82',
           '320', '147', '178', '373', '172', '113', '75', '564', '224', '214', '71', '151', '226', '237', '167',
           '52', '12', '128', '84', '342', '64', '102', '165', '91', '107', '97', '242', '44', '532', '336', '76',
           '180', '130', '155', '393', '229', '94', '33', '13', '146', '73', '8', '958', '62', '125', '359', '6',
           '198', '255', '49', '302', '154', '260', '313', '103', '263', '294', '196', '335', '170', '11', '152',
           '19', '126', '596', '95', '29', '86', '210', '16', '204', '181', '349', '527', '386', '5', '223', '68',
           '65', '201', '288', '28', '251', '364', '285', '343', '171', '274', '325', '247', '150', '449', '169',
           '199', '283', '157', '368', '252', '282', '26', '176', '234', '232', '338', '22', '108', '168', '240',
           '134', '418', '273', '441', '277', '248', '179', '186', '80', '188', '184', '238', '53', '93', '207',
           '109', '233', '425', '79', '122', '27', '444', '24', '54', '208', '162', '111', '153', '90', '236',
           '159', '138', '135', '266', '250', '256', '110', '148', '318', '67', '341', '346', '293', '225', '189',
           '59', '217', '433', '760', '321', '330', '117', '315', '738', '594', '48', '322', '297', '100', '63',
           '34', '304', '58', '228', '55', '120', '516', '3', '124', '192', '202', '119', '286', '221', '141',
           '137', '398', '139', '354', '216', '96', '327', '259', '177', '299', '20', '31', '7', '197', '121',
           '206', '69', '257', '15', '185', '291', '72', '144', '212', '366', '4', '116', '78', '175', '326', '365',
           '577', '367', '160', '35', '87', '81', '61', '271', '314', '161', '200', '101', '127', '190', '173',
           '303', '99', '209', '106', '164', '40', '215', '483', '254', '114', '143', '193', '203', '261', '70',
           '60', '465', '218', '83', '131', '239', '227', '10', '220', '272', '158', '384']


print("-------------------- PARAMETERS ---------------------")
print("Vocabulary size: %i" % (vocab_size + len(tokens)))
print("Read number of folders: %i" % len(folders))
print("-----------------------------------------------------\n")


# --------- Read all Ubuntu source files, do regex operations and squash them into two giant files ------

if force_create_new_files and path_exists(squashed_source_data_x) and path_exists(squashed_source_data_y):
    os.remove(squashed_source_data_x)
    os.remove(squashed_source_data_y)
if path_exists(squashed_source_data_x) and path_exists(squashed_source_data_y):
    print('Source files already created')
else:
    print('Reading all the files and create initial files...')
    read_every_data_file_and_create_initial_files(folders=folders,
                                              initial_x_file_path=squashed_source_data_x,
                                              initial_y_file_path=squashed_source_data_y)


# --------- Do spell check ------------------------------------------------------------------------------

if path_exists(spell_checked_data_x) and path_exists(spell_checked_data_y) and not force_create_new_files:
    print('Spellcheck already done')
else:
    print('Spellchecker for the initial files, create new spell checked files...')
    replace_mispelled_words_in_file(source_file_path=squashed_source_data_x,
                                new_file_path=spell_checked_data_x,
                                misspelled_vocabulary=spell_checked_vocabulary)

    replace_mispelled_words_in_file(source_file_path=squashed_source_data_y,
                                new_file_path=spell_checked_data_y,
                                misspelled_vocabulary=spell_checked_vocabulary)


# --------- Merge training files to one for feeding into fasttext model ---------------------------------

if not path_exists(fast_text_training_data) or not force_create_new_files:
    merge_files(x_path=spell_checked_data_x, y_path=spell_checked_data_y, final_file=fast_text_training_data)


# --------- Create FastText model and replace vectors for FastText model --------------------------------
print('Creating vocabulary for FastText model...')
sorted_dict = find_dictionary(x_train=spell_checked_data_x, y_train=spell_checked_data_y)
unknown_words = create_vocabulary_and_return_unknown_words(sorted_dict=sorted_dict, vocab_path=vocabulary, vocab_size=vocab_size, init_tokens=tokens)

# If model exists, just read parameters in stead of training all over
if path_exists("./datafiles/model.bin") and not force_train_fast_model_all_over:
    print("Load existing FastText model...")
    model = fasttext.load_model('./datafiles/model.bin', encoding='utf-8')
else:
    print("Create FastText model...")
    model = create_fast_text_model(merged_spellcheck_path=fast_text_training_data)

print("Find most similar words to out-of-vocabulary words...")
unknown_words, vocab_words = get_most_similar_words(model=model, vocabulary_path=vocabulary, unknown_words=unknown_words)
unknown_words = get_most_similar_words_for_UNK(unknown_words=unknown_words, vocab_words=vocab_words,
                                               unknown_dict_pickle_path="./datafiles/unknown_words.pickle",
                                               unknown_dict_file_path="./datafiles/unknown_words.txt",
                                               save_freq=5)

    # --------- Replace unknown words in dataset ---------------------------------

if force_create_new_files or not path_exists(no_unk_words_x) or not path_exists(no_unk_words_y):
    replace_UNK_words_in_file(source_file_path=spell_checked_data_x, new_file_path=no_unk_words_x, dictionary=unknown_words)
    replace_UNK_words_in_file(source_file_path=spell_checked_data_y, new_file_path=no_unk_words_y, dictionary=unknown_words)

if force_create_new_files or not path_exists(unshuffled_training_data):
    print('Creating final merged files')
    create_final_merged_files(no_unk_words_x, no_unk_words_y, vocabulary, unshuffled_training_data,
                              unshuffled_validation_data, unshuffled_test_data, val_size_fraction, test_size_fraction)

if force_create_new_files or not path_exists(training_data):
    print('Shuffle files')
    shuffle_file(unshuffled_training_data, training_data)
    shuffle_file(unshuffled_validation_data, validation_data)
    shuffle_file(unshuffled_test_data, test_data)

import time
import sys
from preprocessing1 import preprocess1
from preprocessing2 import preprocessing2
from preprocessing3 import preprocessing3
from preprocessing1_context import preprocess1_context
from preprocessing1_contextFullTurns import preprocess1_contextFullTurns
from preprocessing1_opensubtitles import preprocess1_opensubtitles


from preprocess_helpers import path_exists, shuffle_file, create_final_merged_files, from_index_to_words

sys.path.insert(0, '../')
from variables import tokens_init_list, _buckets, paths_from_preprocessing as paths, vocabulary_size
from variables import contextFullTurns, context, opensubtitles

""" FILL IN CORRECT INFO """
force_create_new_files = False
force_train_fast_model_all_over = False
dataset = "opensubtitles"
#dataset = "UDC"
""" ------------------- """

# Find correct path
if context:
    from variables import paths_from_preprocessing_context as paths
if contextFullTurns:
    from variables import paths_from_preprocessing_contextFullTurns as paths
buckets = [_buckets[-1]]  # Only need the biggest bucket
if dataset == "opensubtitles":
    from variables import paths_from_preprocessing_opensubtitles as paths




vocab_size = vocabulary_size - len(tokens_init_list)  # Minus number of init tokens
save_frequency_unk_words = 50000
val_size_fraction = 0.1
test_size_fraction = 0.1



# Folders to loop
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
#folders = ['test']



def start_preprocessing():
    print("-------------------- INFORMATION --------------------")
    print("Force create new files: " + str(force_create_new_files))
    print("Force train fast model: " + str(force_train_fast_model_all_over))
    print("Context: " + str(context))
    print("ContextFullTurns: " + str(contextFullTurns))
    print("Opensubtitles: " + str(opensubtitles))
    print("Vocabulary size: " + vocabulary_size)
    print("Will start preprocessing in 4 seconds")
    time.sleep(4)

    print("-------------------- PARAMETERS ---------------------")
    print("Vocabulary size: %i" % (vocab_size + len(tokens_init_list)))
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
                                     paths['fast_text_train_path'], force_train_fast_model_all_over)

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
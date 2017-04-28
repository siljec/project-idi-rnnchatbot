# File to hold global variables

paths_from_model = {
    'preprocess_root_files': '../Preprocessing/datafiles/',
    'vocab_path': '../Preprocessing/datafiles/vocabulary.txt',
    'train_path': '../Preprocessing/datafiles/training_data.txt',
    'train_file': 'training_data.txt',
    'dev_path': '../Preprocessing/datafiles/validation_data.txt',
    'dev_file': 'validation_data.txt',
    'test_path': '../Preprocessing/datafiles/test_data.txt',
    'test_file': 'test_data.txt',
    'misspellings': '../misspellings.txt',
    'fast_text_model': '../Preprocessing/datafiles/model.bin',
    'ubuntu': './../../ubuntu-ranking-dataset-creator',
    'preprocess_root_files_context': '../Preprocessing/context/',
    'vocab_path_context': '../Preprocessing/context/vocabulary.txt',
    'train_path_context': '../Preprocessing/context/training_data.txt',
    'train_file_context': 'training_data.txt',
    'dev_path_context': '../Preprocessing/context/validation_data.txt',
    'dev_file_context': 'validation_data.txt',
    'test_path_context': '../Preprocessing/context/test_data.txt',
    'test_file_context': 'test_data.txt',
    'misspellings_context': '../Preprocessing/context/misspellings.txt',
    'fast_text_model_context': '../Preprocessing/context/model.bin',
    'vocab_vectors': '../Preprocessing/datafiles/vocab_vectors_path.pickle',
    'vocab_vectors_context': '../Preprocessing/context/vocab_vectors_path.pickle',
    'stateful_datafiles': '../Preprocessing/stateful/datafiles/',
    'merged_train_stateful_path_file1': "../Preprocessing/stateful/datafiles/merged_training_file1.txt",
    'merged_train_stateful_path_file2': "../Preprocessing/stateful/datafiles/merged_training_file2.txt",
    'merged_dev_stateful_path': "../Preprocessing/stateful/datafiles/merged_dev_file.txt",
    'perplexity_log': 'perplexity_log.txt'
}

paths_from_model_context_full_turns = {
    'ubuntu': './../../ubuntu-ranking-dataset-creator',
    'preprocess_root_files_context': '../Preprocessing/contextFullTurns/',
    'vocab_path': '../Preprocessing/contextFullTurns/vocabulary.txt',
    'train_path': '../Preprocessing/contextFullTurns/training_data.txt',
    'train_file': 'training_data.txt',
    'dev_path': '../Preprocessing/contextFullTurns/validation_data.txt',
    'dev_file': 'validation_data.txt',
    'test_path': '../Preprocessing/contextFullTurns/test_data.txt',
    'test_file': 'test_data.txt',
    'fast_text_model': '../Preprocessing/contextFullTurns/model.bin',
    'vocab_vectors': '../Preprocessing/contextFullTurns/vocab_vectors_path.pickle',
    'misspellings': '../misspellings.txt',
    'perplexity_log': 'perplexity_log.txt'
}


paths_from_model_context = {
    'ubuntu': './../../ubuntu-ranking-dataset-creator',
    'preprocess_root_files_context': '../Preprocessing/context/',
    'vocab_path': '../Preprocessing/context/vocabulary.txt',
    'train_path': '../Preprocessing/context/training_data.txt',
    'train_file': 'training_data.txt',
    'dev_path': '../Preprocessing/context/validation_data.txt',
    'dev_file': 'validation_data.txt',
    'test_path': '../Preprocessing/context/test_data.txt',
    'test_file': 'test_data.txt',
    'fast_text_model': '../Preprocessing/context/model.bin',
    'vocab_vectors': '../Preprocessing/context/vocab_vectors_path.pickle',
    'misspellings': '../misspellings.txt',
    'perplexity_log': 'perplexity_log.txt'
}


paths_from_preprocessing = {
    'source_folder_root': "../../ubuntu-ranking-dataset-creator/src/dialogs/",
    'raw_data_x_path': "./datafiles/raw_data_x.txt",
    'raw_data_y_path': "./datafiles/raw_data_y.txt",

    'regex_x_path': "./datafiles/regex_x.txt",
    'regex_y_path': "./datafiles/regex_y.txt",

    'spell_checked_data_x_path': "./datafiles/spell_checked_data_x.txt",
    'spell_checked_data_y_path': "./datafiles/spell_checked_data_y.txt",
    'misspellings_path': "../misspellings.txt",

    'fast_text_train_path': "./datafiles/fast_text_train.txt",
    'fasttext_model_path': "./datafiles/model.bin",

    'bucket_data_x_path': "./datafiles/bucket_data_x.txt",
    'bucket_data_y_path': "./datafiles/bucket_data_y.txt",

    'final_data_x_path': "./datafiles/final_data_x.txt",
    'final_data_y_path': "./datafiles/final_data_y.txt",

    'unshuffled_training_data': "./datafiles/unshuffled_training_data.txt",
    'unshuffled_validation_data': "./datafiles/unshuffled_validation_data.txt",
    'unshuffled_test_data': "./datafiles/unshuffled_test_data.txt",

    'training_data': "./datafiles/training_data.txt",
    'validation_data': "./datafiles/validation_data.txt",
    'test_data': "./datafiles/test_data.txt",

    'vocabulary_txt_path': "./datafiles/vocabulary.txt",
    'vocabulary_pickle_path': "./datafiles/vocabulary.pickle",

    'vocab_vectors_path': "./datafiles/vocab_vectors_path.pickle",
    'unk_vectors_path': "./datafiles/unk_vectors_path.pickle",
    'unk_to_vocab_pickle_path': "./datafiles/unk_to_vocab.pickle",
    'unk_to_vocab_txt_path': "./datafiles/unk_to_vocab.txt",
    'test_file_words_path': "./datafiles/test_file_words_path.txt"
}

paths_from_preprocessing_context = {
    'source_folder_root': "../../ubuntu-ranking-dataset-creator/src/dialogs/",
    'raw_data_x_path': "./context/raw_data_x.txt",
    'raw_data_y_path': "./context/raw_data_y.txt",

    'regex_x_path': "./context/regex_x.txt",
    'regex_y_path': "./context/regex_y.txt",

    'spell_checked_data_x_path': "./context/spell_checked_data_x.txt",
    'spell_checked_data_y_path': "./context/spell_checked_data_y.txt",
    'misspellings_path': "./context/misspellings.txt",

    'fast_text_train_path': "./context/fast_text_train.txt",
    'fasttext_model_path': "./context/model.bin",

    'bucket_data_x_path': "./context/bucket_data_x.txt",
    'bucket_data_y_path': "./context/bucket_data_y.txt",

    'final_data_x_path': "./context/final_data_x.txt",
    'final_data_y_path': "./context/final_data_y.txt",

    'unshuffled_training_data': "./context/unshuffled_training_data.txt",
    'unshuffled_validation_data': "./context/unshuffled_validation_data.txt",
    'unshuffled_test_data': "./context/unshuffled_test_data.txt",

    'training_data': "./context/training_data.txt",
    'validation_data': "./context/validation_data.txt",
    'test_data': "./context/test_data.txt",

    'vocabulary_txt_path': "./context/vocabulary.txt",
    'vocabulary_pickle_path': "./context/vocabulary.pickle",

    'vocab_vectors_path': "./context/vocab_vectors_path.pickle",
    'unk_vectors_path': "./context/unk_vectors_path.pickle",
    'unk_to_vocab_pickle_path': "./context/unk_to_vocab.pickle",
    'unk_to_vocab_txt_path': "./context/unk_to_vocab.txt",

    'test_file_words_path': "./context/test_file_words_path.txt"

}


paths_from_preprocessing_contextFullTurns = {
    'source_folder_root': "../../ubuntu-ranking-dataset-creator/src/dialogs/",
    'raw_data_x_path': "./contextFullTurns/raw_data_x.txt",
    'raw_data_y_path': "./contextFullTurns/raw_data_y.txt",

    'regex_x_path': "./contextFullTurns/regex_x.txt",
    'regex_y_path': "./contextFullTurns/regex_y.txt",

    'spell_checked_data_x_path': "./contextFullTurns/spell_checked_data_x.txt",
    'spell_checked_data_y_path': "./contextFullTurns/spell_checked_data_y.txt",
    'misspellings_path': "../misspellings.txt",

    'fast_text_train_path': "./contextFullTurns/fast_text_train.txt",
    'fasttext_model_path': "./contextFullTurns/model.bin",

    'bucket_data_x_path': "./contextFullTurns/bucket_data_x.txt",
    'bucket_data_y_path': "./contextFullTurns/bucket_data_y.txt",

    'final_data_x_path': "./contextFullTurns/final_data_x.txt",
    'final_data_y_path': "./contextFullTurns/final_data_y.txt",

    'unshuffled_training_data': "./contextFullTurns/unshuffled_training_data.txt",
    'unshuffled_validation_data': "./contextFullTurns/unshuffled_validation_data.txt",
    'unshuffled_test_data': "./contextFullTurns/unshuffled_test_data.txt",

    'training_data': "./contextFullTurns/training_data.txt",
    'validation_data': "./contextFullTurns/validation_data.txt",
    'test_data': "./contextFullTurns/test_data.txt",

    'vocabulary_txt_path': "./contextFullTurns/vocabulary.txt",
    'vocabulary_pickle_path': "./contextFullTurns/vocabulary.pickle",

    'vocab_vectors_path': "./contextFullTurns/vocab_vectors_path.pickle",
    'unk_vectors_path': "./contextFullTurns/unk_vectors_path.pickle",
    'unk_to_vocab_pickle_path': "./contextFullTurns/unk_to_vocab.pickle",
    'unk_to_vocab_txt_path': "./contextFullTurns/unk_to_vocab.txt",

    'test_file_words_path': "./contextFullTurns/test_file_words_path.txt"

}

paths_from_preprocessing_opensubtitles = {
    'source_folder_root': "../../opensubtitles-parser/data/",
    'raw_data_x_path': "./opensubtitles/raw_data_x.txt",
    'raw_data_y_path': "./opensubtitles/raw_data_y.txt",

    'regex_x_path': "./opensubtitles/regex_x.txt",
    'regex_y_path': "./opensubtitles/regex_y.txt",

    'spell_checked_data_x_path': "./opensubtitles/spell_checked_data_x.txt",
    'spell_checked_data_y_path': "./opensubtitles/spell_checked_data_y.txt",
    'misspellings_path': "../misspellings.txt",

    'fast_text_train_path': "./opensubtitles/fast_text_train.txt",
    'fasttext_model_path': "./opensubtitles/model.bin",

    'bucket_data_x_path': "./opensubtitles/bucket_data_x.txt",
    'bucket_data_y_path': "./opensubtitles/bucket_data_y.txt",

    'final_data_x_path': "./opensubtitles/final_data_x.txt",
    'final_data_y_path': "./opensubtitles/final_data_y.txt",

    'unshuffled_training_data': "./opensubtitles/unshuffled_training_data.txt",
    'unshuffled_validation_data': "./opensubtitles/unshuffled_validation_data.txt",
    'unshuffled_test_data': "./opensubtitles/unshuffled_test_data.txt",

    'training_data': "./opensubtitles/training_data.txt",
    'validation_data': "./opensubtitles/validation_data.txt",
    'test_data': "./opensubtitles/test_data.txt",

    'vocabulary_txt_path': "./opensubtitles/vocabulary.txt",
    'vocabulary_pickle_path': "./opensubtitles/vocabulary.pickle",

    'vocab_vectors_path': "./opensubtitles/vocab_vectors_path.pickle",
    'unk_vectors_path': "./opensubtitles/unk_vectors_path.pickle",
    'unk_to_vocab_pickle_path': "./opensubtitles/unk_to_vocab.pickle",
    'unk_to_vocab_txt_path': "./opensubtitles/unk_to_vocab.txt",

    'test_file_words_path': "./opensubtitles/test_file_words_path.txt"
}

paths_from_preprocessing_stateful = {
    'source_folder_root': "../../ubuntu-ranking-dataset-creator/src/dialogs/",
    'misspellings_path': "../misspellings.txt",
    'stateful_datafiles': "./stateful/datafiles/",
    'stateful_raw_files': "./stateful/raw_files/",

    'fast_text_train_path': "./datafiles/fast_text_train.txt",
    'fasttext_model_path': "./datafiles/model.bin",

    'vocabulary_txt_path': "./datafiles/vocabulary.txt",
    'vocabulary_pickle_path': "./datafiles/vocabulary.pickle",

    'vocab_vectors_path': "./datafiles/vocab_vectors_path.pickle",
    'unk_vectors_path': "./datafiles/unk_vectors_path.pickle",
    'unk_to_vocab_pickle_path': "./datafiles/unk_to_vocab.pickle",
    'unk_to_vocab_txt_path': "./datafiles/unk_to_vocab.txt",
    'test_file_words_path': "./datafiles/test_file_words_path.txt",
    'merged_train_path_file1': "./stateful/datafiles/merged_training_file1.txt",
    'merged_train_path_file2': "./stateful/datafiles/merged_training_file2.txt",
    'merged_test_path': "./stateful/datafiles/merged_test_file.txt",
    'merged_dev_path': "./stateful/datafiles/merged_dev_file.txt",
    'perplexity_log': 'perplexity_log.txt'
}


tokens = {
    'padding': ('_PAD', 0),
    'go': ('_GO', 1),
    'eos': ('_EOS', 2),
    'eot': ('_EOT', 3),
    'unk': ('_UNK', 4),
    'url': ('_URL', None),
    'emoji': ('_EMJ', None),
    'directory': ('_DIR', None)
}

tokens_list = ['_PAD', '_GO', '_EOS', '_EOT', '_UNK', '_URL', '_EMJ', '_DIR']
tokens_init_list = ['_PAD', '_GO', '_EOS', '_EOT', '_UNK']

_buckets = [(10, 10), (16, 16), (22, 22), (30, 30)]

vocabulary_size = 30000

print_frequency = 1000
steps_per_checkpoint = 10000
max_training_steps = 300005
size = 1024
num_layers = 2
batch_size = 24

learning_rate = 0.1
optimizer = "Adagrad" # LR = 0.1
#optimizer = "GradientDescent" # LR = 0.5
#optimizer = "Adam" # LR = 0.0001
word_embedding_size = 1000

use_gpu = '/gpu:0'

context = False
contextFullTurns = False
opensubtitles = False
if opensubtitles:
    vocabulary_size = 20000
if contextFullTurns:
    _buckets = [(10, 10), (16, 16), (22, 22), (60, 60)]

# For UDC dataset
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
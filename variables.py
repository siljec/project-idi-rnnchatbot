

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
    'misspellings': '../Preprocessing/datafiles/misspellings.txt',
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
}

paths_from_preprocessing = {
    'source_folder_root': "../../ubuntu-ranking-dataset-creator/src/dialogs/",
    'raw_data_x_path': "./datafiles/raw_data_x.txt",
    'raw_data_y_path': "./datafiles/raw_data_y.txt",

    'regex_x_path': "./datafiles/regex_x.txt",
    'regex_y_path': "./datafiles/regex_y.txt",

    'spell_checked_data_x_path': "./datafiles/spell_checked_data_x.txt",
    'spell_checked_data_y_path': "./datafiles/spell_checked_data_y.txt",
    'misspellings_path': "./datafiles/misspellings.txt",

    'fast_text_train_path': "./datafiles/fast_text_train.txt",

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

_buckets = [(10, 10), (20, 20), (35, 35), (50, 50)]

vocabulary_size = 30000
print_frequency = 1000
steps_per_checkpoint = 5000
max_training_steps = 250005
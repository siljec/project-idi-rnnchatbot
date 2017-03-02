import time
from preprocessing1 import preprocess1
from preprocessing2 import preprocessing2
from preprocessing3 import preprocessing3
from preprocessing1_context import preprocess1_context

from preprocess_helpers import path_exists, shuffle_file, create_final_merged_files

# --------- All paths -----------------------------------------------------------------------------------
force_create_new_files = False
force_train_fast_model_all_over = False
context = False
dir = "datafiles"
if context:
    dir = "context"

print("-------------------- INFORMATION --------------------")
print("Force create new files: " + str(force_create_new_files))
print("Force train fast model: " + str(force_train_fast_model_all_over))
print("Context: " + str(context))
print("Files will be saved to: " + str(dir))
print("Will start preprocessing in 4 seconds")
print("-----------------------------------------------------\n")
time.sleep(4)

tokens = ['_PAD', '_GO', '_EOS', '_EOT', '_UNK', '_URL', '_EMJ', '_DIR']
init_tokens = ['_PAD', '_GO', '_EOS', '_EOT', '_UNK']
buckets = [(50, 50)]

vocab_size = 2000 - len(init_tokens)  # Minus number of init tokens
save_frequency_unk_words = 50000
val_size_fraction = 0.1
test_size_fraction = 0.1

source_folder_root = "../../ubuntu-ranking-dataset-creator/src/dialogs/"
raw_data_x_path = "./" + dir + "/raw_data_x.txt"
raw_data_y_path = "./" + dir + "/raw_data_y.txt"

regex_x_path = "./" + dir + "/regex_x.txt"
regex_y_path = "./" + dir + "/regex_y.txt"

spell_checked_data_x_path = "./" + dir + "/spell_checked_data_x.txt"
spell_checked_data_y_path = "./" + dir + "/spell_checked_data_y.txt"
misspellings_path = "./datafiles/misspellings.txt"

fast_text_train_path = "./" + dir + "/fast_text_train.txt"

bucket_data_x_path = "./" + dir + "/bucket_data_x.txt"
bucket_data_y_path = "./" + dir + "/bucket_data_y.txt"

final_data_x_path = "./" + dir + "/final_data_x.txt"
final_data_y_path = "./" + dir + "/final_data_y.txt"

unshuffled_training_data = "./" + dir + "/unshuffled_training_data.txt"
unshuffled_validation_data = "./" + dir + "/unshuffled_validation_data.txt"
unshuffled_test_data = "./" + dir + "/unshuffled_test_data.txt"

training_data = "./" + dir + "/training_data.txt"
validation_data = "./" + dir + "/validation_data.txt"
test_data = "./" + dir + "/test_data.txt"

vocabulary_txt_path = "./" + dir + "/vocabulary.txt"
vocabulary_pickle_path = "./" + dir + "/vocabulary.pickle"

vocab_vectors_path = "./" + dir + "/vocab_vectors_path.pickle"
unk_vectors_path = "./" + dir + "/unk_vectors_path.pickle"
unk_to_vocab_pickle_path = "./" + dir + "/unk_to_vocab.pickle"
unk_to_vocab_txt_path = "./" + dir + "/unk_to_vocab.txt"




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

folders = ['test']

print("-------------------- PARAMETERS ---------------------")
print("Vocabulary size: %i" % (vocab_size + len(init_tokens)))
print("Read number of folders: %i" % len(folders))
print("-----------------------------------------------------\n")

# Step 1
if context:
    preprocess1_context(folders, force_create_new_files, raw_data_x_path, raw_data_y_path, regex_x_path, regex_y_path, spell_checked_data_x_path, spell_checked_data_y_path, misspellings_path)
else:
    preprocess1(folders, force_create_new_files, raw_data_x_path, raw_data_y_path, regex_x_path, regex_y_path, spell_checked_data_x_path, spell_checked_data_y_path, misspellings_path)
# Step 2
fast_text_model = preprocessing2(spell_checked_data_x_path, spell_checked_data_y_path, fast_text_train_path, force_train_fast_model_all_over)
# Step 3
preprocessing3(buckets, spell_checked_data_x_path, spell_checked_data_y_path, bucket_data_x_path, bucket_data_y_path,
               vocab_size, vocabulary_txt_path, vocabulary_pickle_path, fast_text_model, vocab_vectors_path,
               unk_vectors_path, unk_to_vocab_pickle_path, unk_to_vocab_txt_path, save_frequency_unk_words, final_data_x_path,
               final_data_y_path, init_tokens)

if force_create_new_files or not path_exists(unshuffled_training_data):
    print('Creating final merged files')
    create_final_merged_files(final_data_x_path, final_data_y_path, vocabulary_txt_path, unshuffled_training_data,
                              unshuffled_validation_data, unshuffled_test_data, val_size_fraction, test_size_fraction)

if force_create_new_files or not path_exists(training_data):
    print('Shuffle files')
    shuffle_file(unshuffled_training_data, training_data)
    shuffle_file(unshuffled_validation_data, validation_data)
    shuffle_file(unshuffled_test_data, test_data)

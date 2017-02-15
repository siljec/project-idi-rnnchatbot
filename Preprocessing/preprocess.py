import os, re, time
from create_vocabulary import read_vocabulary_from_file, encode_sentence, create_vocabulary_and_return_unknown_words, find_dictionary
from spell_error_fix import replace_mispelled_words_in_file, replace_misspelled_word_helper
from random import shuffle
from itertools import izip
import fasttext
import numpy as np
import time
import pickle



def get_time(start_time):
    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration / 60)
    seconds = duration % 60
    return "Time " + str(minutes) + " minutes " + str(seconds) +" seconds"


def preprocess_training_file(path, x_train_path, y_train_path):
    go_token = ""
    eos_token = " _EOS "
    url_token = " _URL "
    emoji_token = " _EMJ "
    eot_token = ""

    smiley_pattern = r'(?::|;|=)(?:-)?(?:\)|\(|D|P|\|)'  # NB: will take :) from /:) and :)D
    smiley_pattern = r'(?::|;|=)(?:-)?(?:\)|\(|D|P|\|)(?=$|\s)'  # NB: Will take :) from /:) but not from :)D
    smiley_pattern = r'((?:^|\s)(?::|;|=)(?:-)?(?:\)|\(|D|P|\|)(?=$|\s))'  # NB: Will neither take :) from /:) nor from :)D

    user1_first_line = True

    x_train = []
    y_train = []

    sentence_holder = ""

    with open(path) as fileobject:
        for line in fileobject:
            data = line.split("\t")
            current_user = data[1]
            text = data[3].strip().lower()
            text = re.sub(' +', ' ', text)  # Will remove multiple spaces
            text = re.sub(r'(https?://[^\s]+)', url_token, text)  # Exchange urls with URL token
            text = re.sub(smiley_pattern, emoji_token, text)  # Exchange smiles with EMJ token
            text = re.sub('(?<=[a-z])([!?,.])', r' \1', text)  # Add space before special characters [!?,.]

            if user1_first_line:
                init_user = current_user
                previous_user = current_user
                user1_first_line = False
                sentence_holder = go_token

            if current_user == previous_user:  # The user is still talking
                sentence_holder += text + eos_token
            else:  # A new user responds
                if ('_EOS' in sentence_holder):
                    sentence_holder += eot_token + "\n"
                else:
                    sentence_holder += eot_token + "\n"
                if current_user == init_user:  # Init user talks (should add previous sentence to y_train)
                    y_train.append(sentence_holder)
                else:
                    x_train.append(sentence_holder)
                sentence_holder = go_token + text + eos_token

            previous_user = current_user

    if current_user != init_user:
        y_train.append(sentence_holder + eot_token + "\n")

    x_train_file = open(x_train_path, 'a')
    y_train_file = open(y_train_path, 'a')

    for i in range(len(y_train)):
        x_train_file.write(x_train[i].strip() + "\n")
        y_train_file.write(y_train[i].strip() + "\n")

    x_train_file.close()
    y_train_file.close()


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def merge_files(x_path, y_path, final_file):
    with open(x_path) as x_file, open(y_path) as y_file, open(final_file, 'w') as final_file:
        for x, y in izip(x_file, y_file):
            final_file.write(x)
            final_file.write(y)


def create_final_merged_files(x_path, y_path, vocabulary_path, train_path, val_path, test_path, val_size_fraction,
                              test_size_fraction):
    vocabulary, _ = read_vocabulary_from_file(vocabulary_path)
    train_final = open(train_path, 'w')
    val_final = open(val_path, 'w')
    test_final = open(test_path, 'w')
    num_lines = file_len(x_path)

    train_size = num_lines * (1 - val_size_fraction - test_size_fraction)
    val_size = train_size + num_lines * test_size_fraction
    line_counter = 0
    with open(x_path) as x_file, open(y_path) as y_file:
        for x, y in izip(x_file, y_file):
            if line_counter < train_size:
                train_final.write(
                    encode_sentence(x.strip().split(" "), vocabulary) + ", " + encode_sentence(y.strip().split(" "),
                                                                                               vocabulary) + '\n')
            elif line_counter < val_size:
                val_final.write(
                    encode_sentence(x.strip().split(" "), vocabulary) + ", " + encode_sentence(y.strip().split(" "),
                                                                                               vocabulary) + '\n')
            else:
                test_final.write(
                    encode_sentence(x.strip().split(" "), vocabulary) + ", " + encode_sentence(y.strip().split(" "),
                                                                                               vocabulary) + '\n')
            line_counter += 1
    train_final.close()
    val_final.close()
    test_final.close()


def create_final_files(source_path, vocabulary_path, train_path, val_path, test_path, val_size_fraction,
                       test_size_fraction):
    vocabulary, _ = read_vocabulary_from_file(vocabulary_path)
    train_final = open(train_path, 'w')
    val_final = open(val_path, 'w')
    test_final = open(test_path, 'w')
    num_lines = file_len(source_path)
    train_size = num_lines * (1 - val_size_fraction - test_size_fraction)
    val_size = train_size + num_lines * test_size_fraction
    line_counter = 0
    with open(source_path) as fileobject:
        for line in fileobject:
            if line_counter < train_size:
                train_final.write(encode_sentence(line.strip().split(" "), vocabulary) + '\n')
            elif line_counter < val_size:
                val_final.write(encode_sentence(line.strip().split(" "), vocabulary) + '\n')
            else:
                test_final.write(encode_sentence(line.strip().split(" "), vocabulary) + '\n')
            line_counter += 1.0
    train_final.close()
    val_final.close()
    test_final.close()


def read_every_data_file_and_create_initial_files(folders, initial_x_file_path, initial_y_file_path):
    start_time = time.time()
    number_of_files_read = 0  # Can remove, but nice for the report best regards siljus christus
    for folder in folders:
        folder_path = "../../ubuntu-ranking-dataset-creator/src/dialogs/" + folder
        for filename in os.listdir(folder_path):
            number_of_files_read += 1
            file_path = folder_path + "/" + filename
            preprocess_training_file(file_path, initial_x_file_path, initial_y_file_path)
        print("Done with folder: " + str(folder) + ", read " + str(number_of_files_read) + " files")

    print("Number of files read: " + str(number_of_files_read))
    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration / 60)
    seconds = duration % 60
    print("Time ", minutes, " minutes ", seconds, " seconds")

def get_most_similar_words_for_UNK(unknown_words, vocab_words):
    start_time_unk = time.time()
    for unk_key, unk_values in unknown_words.iteritems():
        min_dist = 1
        word = ""
        for key, value in vocab_words.iteritems():
            cur_dist = distance(unk_values, value[0], value[1])
            if cur_dist < min_dist:
                min_dist = cur_dist
                word = key
        unknown_words[unk_key] = word
    print("Time to get similar words for all UNK:", get_time(start_time_unk))
    return unknown_words


def replace_UNK_words_in_file(source_file_path, new_file_path, dictionary):
	new_file = open(new_file_path, 'a')
	with open(source_file_path) as fileobject:
		for line in fileobject:
			sentence = line.split(' ')
			last_word = sentence.pop()
			for word in sentence:
				new_word = replace_misspelled_word_helper(word, dictionary)
				new_file.write(new_word + ' ')
			new_word = replace_misspelled_word_helper(last_word, dictionary)
			new_file.write(new_word)
	new_file.close()


# Used to shuffle the list
def get_random_folders():
    folders = os.listdir("../../ubuntu-ranking-dataset-creator/src/dialogs")
    shuffle(folders)
    new_folders = []
    for i in range(len(folders)):
        if folders[i] != ".DS_Store":
            new_folders.append(folders[i])
    return new_folders


def shuffle_file(path, target_file):
    lines = open(path).readlines()
    shuffle(lines)
    open(target_file, 'w').writelines(lines)



def distance(unk, known, known_lenght):
    unk_len = np.linalg.norm(unk)
    numerator = np.dot(unk, known)
    denominator = unk_len * known_lenght
    return 1 - (numerator / denominator)







def generate_all_files(vocab_size=100000, tokens=["_PAD", "_GO ", "_EOS", "_EOT", "_UNK"], val_size_fraction=0.1,
                       test_size_fraction=0.1):
    vocab_size -= len(tokens)
    start_time = time.time()

    # Get current location and find the folder the script is running from
    path = os.getcwd()
    current_folder = path.split("/")[-1]

    # Different folders have different paths to source files
    if current_folder == "Models":
        misspelled_words_path = './../Preprocessing/misspellings.txt'  # Is not generated by the code. Don't delete!
        vocabulary_path = './../Preprocessing/vocabulary.txt'
        x_train_initial_path = './../Preprocessing/x_train_init.txt'
        y_train_initial_path = './../Preprocessing/y_train_init.txt'
        x_train_spell_check_path = './../Preprocessing/x_train_spell_check.txt'
        y_train_spell_check_path = './../Preprocessing/y_train_spell_check.txt'
        x_train_fixed_UNK = './../Preprocessing/x_train_fixed_UNK.txt'
        y_train_fixed_UNK = './../Preprocessing/y_train_fixed_UNK.txt'
        x_train_final_path = './../Preprocessing/x_train.txt'
        y_train_final_path = './../Preprocessing/y_train.txt'
        x_val_path = './../Preprocessing/x_val.txt'
        y_val_path = './../Preprocessing/y_val.txt'
        x_test_path = './../Preprocessing/x_test.txt'
        y_test_path = './../Preprocessing/y_test.txt'

        merged_spellcheck_path = "./../Preprocessing/merged_spellcheck.txt"

        train_merged_path = './../Preprocessing/train_merged.txt'
        val_merged_path = './../Preprocessing/val_merged.txt'
        test_merged_path = './../Preprocessing/test_merged.txt'

        shuffled_train_merged_path = './../Preprocessing/shuffled_train_merged.txt'
        shuffled_val_merged_path = './../Preprocessing/shuffled_val_merged.txt'
        shuffled_test_merged_path = './../Preprocessing/shuffled_test_merged.txt'

        fast_text = './../Preprocessing/fasttext_known_words.txt'
        known_words_dict = './../Preprocessing/known_words_dict.pickle'


    else:
        misspelled_words_path = './misspellings.txt'  # Is not generated by the code. Don't delete!
        vocabulary_path = './vocabulary.txt'
        x_train_initial_path = './x_train_init.txt'
        y_train_initial_path = './y_train_init.txt'
        x_train_spell_check_path = './x_train_spell_check.txt'
        y_train_spell_check_path = './y_train_spell_check.txt'
        x_train_fixed_UNK = './x_train_fixed_UNK.txt'
        y_train_fixed_UNK = './y_train_fixed_UNK.txt'

        x_train_final_path = './x_train.txt'
        y_train_final_path = './y_train.txt'
        x_val_path = './x_val.txt'
        y_val_path = './y_val.txt'
        x_test_path = './x_test.txt'
        y_test_path = './y_test.txt'

        merged_spellcheck_path = "./merged_spellcheck.txt"

        train_merged_path = './train_merged.txt'
        val_merged_path = './val_merged.txt'
        test_merged_path = './test_merged.txt'

        shuffled_train_merged_path = './shuffled_train_merged.txt'
        shuffled_val_merged_path = './shuffled_val_merged.txt'
        shuffled_test_merged_path = './shuffled_test_merged.txt'

        fast_text = './fasttext_known_words.txt'
        known_words_dict = './known_words_dict.pickle'

    # Remove all files if exists
    all_files = [x_train_initial_path, y_train_initial_path, x_train_spell_check_path, y_train_spell_check_path,
                 x_train_final_path, y_train_final_path, vocabulary_path, x_val_path, y_val_path, x_test_path,
                 y_test_path, train_merged_path, val_merged_path, test_merged_path, fast_text, merged_spellcheck_path]
    for filename in all_files:
        try:
            os.remove(filename)
        except OSError:
            print('File not found: ', filename)

    print('Shuffle data source files')
    # folders = get_random_folders()  # Cant use this because we want to train the different models with the same data.
    folders = ['30']#, '356', '195', '142', '555', '43', '50', '36', '46', '85', '41', '118', '166', '104', '471', '37']
               # '115', '47', '290', '308', '191', '457', '32', '231', '45', '133', '222', '213', '89', '92', '374', '98',
               # '219', '25', '21', '182', '140', '129', '264', '132', '258', '243', '42', '456', '301', '9', '269', '88',
               # '211', '123', '112', '23', '149', '105', '145', '39', '287', '249', '66', '51', '305', '241', '136',
               # '57', '174', '245', '407', '17', '281', '205', '235', '383', '38', '183', '2', '521', '408', '18', '347',
               # '74', '392', '334', '56', '156', '278', '230', '14', '265', '194', '187', '77', '163', '479', '82',
               # '320', '147', '178', '373', '172', '113', '75', '564', '224', '214', '71', '151', '226', '237', '167',
               # '52', '12', '128', '84', '342', '64', '102', '165', '91', '107', '97', '242', '44', '532', '336', '76',
               # '180', '130', '155', '393', '229', '94', '33', '13', '146', '73', '8', '958', '62', '125', '359', '6',
               # '198', '255', '49', '302', '154', '260', '313', '103', '263', '294', '196', '335', '170', '11', '152',
               # '19', '126', '596', '95', '29', '86', '210', '16', '204', '181', '349', '527', '386', '5', '223', '68',
               # '65', '201', '288', '28', '251', '364', '285', '343', '171', '274', '325', '247', '150', '449', '169',
               # '199', '283', '157', '368', '252', '282', '26', '176', '234', '232', '338', '22', '108', '168', '240',
               # '134', '418', '273', '441', '277', '248', '179', '186', '80', '188', '184', '238', '53', '93', '207',
               # '109', '233', '425', '79', '122', '27', '444', '24', '54', '208', '162', '111', '153', '90', '236',
               # '159', '138', '135', '266', '250', '256', '110', '148', '318', '67', '341', '346', '293', '225', '189',
               # '59', '217', '433', '760', '321', '330', '117', '315', '738', '594', '48', '322', '297', '100', '63',
               # '34', '304', '58', '228', '55', '120', '516', '3', '124', '192', '202', '119', '286', '221', '141',
               # '137', '398', '139', '354', '216', '96', '327', '259', '177', '299', '20', '31', '7', '197', '121',
               # '206', '69', '257', '15', '185', '291', '72', '144', '212', '366', '4', '116', '78', '175', '326', '365',
               # '577', '367', '160', '35', '87', '81', '61', '271', '314', '161', '200', '101', '127', '190', '173',
               # '303', '99', '209', '106', '164', '40', '215', '483', '254', '114', '143', '193', '203', '261', '70',
               # '60', '465', '218', '83', '131', '239', '227', '10', '220', '272', '158', '384']

    print('Reading all the files and create initial files...')
    read_every_data_file_and_create_initial_files(folders, x_train_initial_path, y_train_initial_path)

    print('Spellchecker for the initial files, create new spell checked files...')
    replace_mispelled_words_in_file(x_train_initial_path, x_train_spell_check_path, misspelled_words_path)
    replace_mispelled_words_in_file(y_train_initial_path, y_train_spell_check_path, misspelled_words_path)

    print('Creating vocabulary...')
    sorted_dict = find_dictionary(x_train_spell_check_path, y_train_spell_check_path)
    unknown_words = create_vocabulary_and_return_unknown_words(sorted_dict, vocabulary_path, vocab_size)

    print('Merge file for Fasttext learning')
    merge_files(x_train_spell_check_path, y_train_spell_check_path, merged_spellcheck_path)

    print('Create Fasttext model')
    start_time_fasttext = time.time()
    model = fasttext.skipgram(merged_spellcheck_path, 'model')
    print("Time to create Fasttext model: ", get_time(start_time_fasttext))

    print('Get vectors and length for vocabulary words')
    with open(vocabulary_path) as vocabObject:
        vocab_words = {}
        for word in vocabObject:
            word = word.strip()
            vector = model[word]
            vocab_words[word] = vector, np.linalg.norm(vector)
        pickle.dump(vocab_words, open(known_words_dict, "wb"))


    print('Get vectors for out-of-vocabulary words')
    for key in unknown_words:
        unknown_words[key] = model[key]


    print("# UNK words: ", len(unknown_words))
    print('Get most similar word to out-of-vocabulary words')
    unknown_words = get_most_similar_words_for_UNK(unknown_words, vocab_words)

    # Print helper to check if the words actually are similar
    counter = 0
    for key, value in unknown_words.iteritems():
        print(key, value)
        counter += 1
        if counter == 20:
            break

    print('Replace UNK words with the most similar vocab word')
    start_time_replace = time.time()
    replace_UNK_words_in_file(x_train_spell_check_path, x_train_fixed_UNK, unknown_words)
    replace_UNK_words_in_file(y_train_spell_check_path, y_train_fixed_UNK, unknown_words)
    print("Time to get replace UNK words:", get_time(start_time_replace))


    print('Creating final files...')
    create_final_files(x_train_fixed_UNK, vocabulary_path, x_train_final_path, x_val_path, x_test_path,
                       val_size_fraction, test_size_fraction)
    create_final_files(y_train_fixed_UNK, vocabulary_path, y_train_final_path, y_val_path, y_test_path,
                       val_size_fraction, test_size_fraction)

    print('Creating final merged files')
    create_final_merged_files(x_train_fixed_UNK, y_train_fixed_UNK, vocabulary_path, train_merged_path,
                              val_merged_path, test_merged_path, val_size_fraction, test_size_fraction)

    print('Shuffle files')
    shuffle_file(train_merged_path, shuffled_train_merged_path)
    shuffle_file(val_merged_path, shuffled_val_merged_path)
    shuffle_file(test_merged_path, shuffled_test_merged_path)

    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration / 60)
    seconds = duration % 60
    print("Time ", minutes, " minutes ", seconds, " seconds to generate all files")


path = os.getcwd()
current_folder = path.split("/")[-1]
if current_folder == "Preprocessing":
    generate_all_files()

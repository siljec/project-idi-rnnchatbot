import os, re, time
from create_vocabulary import read_vocabulary_from_file, encode_sentence
from spell_error_fix import replace_misspelled_word_helper
from random import shuffle
from itertools import izip
import fasttext
import pickle
import numpy as np
import time


# ------------- Code beautify helpers -----------------------------------------------

def path_exists(path):
	return os.path.exists(path)


def get_time(start_time):
	duration = time.time() - start_time
	m, s = divmod(duration, 60)
	h, m = divmod(m, 60)
	return "%d hours %d minutes %d seconds" % (h, m, s)


def file_len(file_name):
	with open(file_name) as f:
		for i, l in enumerate(f):
			pass
	return i + 1


def save_dict_to_file(file_name, pickle_name, obj):
	with open(file_name, 'w') as fileObject:
		for key, value in obj.items():
			fileObject.write(key + " : " + value + "\n")
	print("Dictionary saved to file " + file_name)

	with open(pickle_name, 'w') as pickleObject:
		pickle.dump(obj, pickleObject)
	print("Dictionary saved to pickle " + pickle_name)


# ------------- FastText helpers ----------------------------------------------------

def create_fast_text_model(merged_spellcheck_path):
	start_time_fasttext = time.time()
	model = fasttext.skipgram(merged_spellcheck_path, './datafiles/model')
	print("Time used to create Fasttext model: ", get_time(start_time_fasttext))
	return model


def get_most_similar_words(model, vocabulary_path, unknown_words):
	print('Get vectors and length for vocabulary words')
	with open(vocabulary_path) as vocabObject:
		vocab_words = {}
		for word in vocabObject:
			word = word.strip()
			vector = np.array(model[word])
			vocab_words[word] = vector, np.linalg.norm(vector)
		# pickle.dump(vocab_words, open(known_words_dict, "wb"))

	print('Get vectors for out-of-vocabulary words')
	for key in unknown_words:
		unknown_words[key] = np.array(model[key])

	print("# UNK words: ", len(unknown_words))
	return unknown_words, vocab_words


def distance(unk, known, known_length):
	unk_len = np.linalg.norm(unk)
	numerator = np.dot(unk, known)
	denominator = unk_len * known_length
	return 1 - (numerator / denominator)


# ------------- Read all data helpers -----------------------------------------------
def split_line_and_do_regex(line, url_token, emoji_token, dir_token):
	data = line.split("\t")
	current_user = data[1]
	text = data[3].strip().lower()  # user user timestamp text

	text = re.sub(' +', ' ', text)  # Will remove multiple spaces
	text = re.sub(r'(https?://[^\s]+)', url_token, text)  # Exchange urls with URL token
	text = re.sub(r'((?:^|\s)(?::|;|=)(?:-)?(?:\)|\(|D|P|\|)(?=$|\s))', emoji_token, text)  # Exchange smiles with EMJ token NB: Will neither take :) from /:) nor from :)D
	text = re.sub('(?<=[a-z])([!?,.])', r' \1', text)  # Add space before special characters [!?,.]
	text = re.sub('"', '', text)  # Remove "
	text = re.sub('((\/\w+)|(\.\/\w+)|(\w+(?=(\/))))()((\/)|(\w+)|(\.\w+)|(\w+|\-|\~))+', dir_token, text)  # Replace directory-paths
	text = re.sub("(?!(')([a-z]{1})(\s))(')(?=\w|\s)", "", text)  # Remove ', unless it is like "banana's"

	return text, current_user


# Reads all folders and squash into one file
def preprocess_training_file(path, x_train_path, y_train_path):
	go_token = ""
	eos_token = " _EOS "
	url_token = " _URL "
	emoji_token = " _EMJ "
	eot_token = ""
	dir_token = "_DIR"

	user1_first_line = True

	x_train = []
	y_train = []

	sentence_holder = ""
	with open(path) as fileobject:
		for line in fileobject:
			text, current_user = split_line_and_do_regex(line, url_token=url_token, emoji_token=emoji_token, dir_token=dir_token)

			if user1_first_line:
				init_user, previous_user = current_user, current_user
				user1_first_line = False
				sentence_holder = go_token

			if current_user == previous_user:  # The user is still talking
				sentence_holder += text + eos_token
			else:  # A new user responds
				if '_EOS' in sentence_holder:
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
				train_final.write( encode_sentence(x.strip().split(" "), vocabulary) + ", " + encode_sentence(y.strip().split(" "), vocabulary) + '\n')
			elif line_counter < val_size:
				val_final.write(encode_sentence(x.strip().split(" "), vocabulary) + ", " + encode_sentence(y.strip().split(" "), vocabulary) + '\n')
			else:
				test_final.write(encode_sentence(x.strip().split(" "), vocabulary) + ", " + encode_sentence(y.strip().split(" "), vocabulary) + '\n')
			line_counter += 1
	train_final.close()
	val_final.close()
	test_final.close()


def get_most_similar_words_for_UNK(unknown_words, vocab_words):
	counter = 1
	start_time_unk = time.time()
	for unk_key, unk_values in unknown_words.iteritems():
		min_dist = 1
		word = ""
		if (counter % 100) == 0:
			print("     Calculated " + str(counter) + " unknown words")
		for key, value in vocab_words.iteritems():
			cur_dist = distance(unk_values, value[0], value[1])
			if cur_dist < min_dist:
				min_dist = cur_dist
				word = key
		unknown_words[unk_key] = word
		counter += 1
	print("Time to get similar words for all UNK:", get_time(start_time_unk))
	return unknown_words


def replace_UNK_words_in_file(source_file_path, new_file_path, dictionary):
	new_file = open(new_file_path, 'w')
	with open(source_file_path) as fileobject:
		for line in fileobject:
			words = line.split(' ')
			sentence = ""
			last_word = words.pop().strip()
			for word in words:
				new_word = replace_misspelled_word_helper(word, dictionary)
				sentence += new_word + ' '
			new_word = replace_misspelled_word_helper(last_word, dictionary)
			new_file.write(sentence + new_word + '\n')
	new_file.close()


def shuffle_file(path, target_file):
	lines = open(path).readlines()
	shuffle(lines)
	open(target_file, 'w').writelines(lines)


# ------------- Currently not in use ------------------------------------------------
# Used to shuffle the list
def get_random_folders():
	folders = os.listdir("../../ubuntu-ranking-dataset-creator/src/dialogs")
	shuffle(folders)
	new_folders = []
	for i in range(len(folders)):
		if folders[i] != ".DS_Store":
			new_folders.append(folders[i])
	return new_folders


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


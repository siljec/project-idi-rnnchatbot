
def read_words_from_misspelling_file(path):
	dictionary = {}
	with open(path) as fileobject:
		for line in fileobject:
			splitted_line = line.split(' ', 1)
			wrong = splitted_line[0]
			correct = splitted_line[1].strip()
			dictionary[wrong] = correct

	return dictionary


def replace_misspelled_word_helper(candidate, dictionary):
	if (candidate in dictionary):
		# print "replacing ", candidate, " with ", dictionary[candidate]
		return dictionary[candidate]
	return candidate


def replace_mispelled_words_in_file(source_file_path, new_file_path, misspelled_vocabulary):
	dictionary = read_words_from_misspelling_file(misspelled_vocabulary)
	new_file = open(new_file_path, 'w')
	with open(source_file_path) as fileobject:
		for line in fileobject:
			sentence = line.split(' ')
			last_word = sentence.pop().strip()
			for word in sentence:
				new_word = replace_misspelled_word_helper(word, dictionary)
				new_file.write(new_word + ' ')
			new_word = replace_misspelled_word_helper(last_word, dictionary)
			new_file.write(new_word + '\n')

	new_file.close()




def read_words_from_misspelling_file(path):
	dictionary = {}
	with open(path) as fileobject:
		for line in fileobject:
			splitted_line = line.split(' ', 1)
			wrong = splitted_line[0]
			correct = splitted_line[1]

			correct = correct.replace('\n', '')	
			dictionary[wrong] = correct

	return dictionary

#def remove_characters(path, character):

def replace_misspelled_word(candidate, dictionary):
	if (candidate in dictionary):
		print "replacing ", candidate, " with ", dictionary[candidate]
		return dictionary[candidate]
	return candidate

def replace_mispelled_words_in_file(path, new_file_path, dictionary):
	new_file = open(new_file_path, 'a')
	with open(path) as fileobject:
		for line in fileobject:
			sentence = line.split(' ')
			last_word = sentence.pop()
			for word in sentence:
				new_word = replace_misspelled_word(word, dictionary)
				new_file.write(new_word + ' ')
			new_word = replace_misspelled_word(last_word, dictionary)
			new_file.write(new_word)

	new_file.close()


dictionary = read_words_from_misspelling_file("./misspellings.txt")

replace_mispelled_words_in_file('./x_train.txt', './x_train_spell_check.txt', dictionary)
replace_mispelled_words_in_file('./y_train.txt', './y_train_spell_check.txt', dictionary)

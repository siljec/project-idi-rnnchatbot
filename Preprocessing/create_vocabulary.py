import operator

def find_dictionary(x_train, y_train):

	dictionary = {}

	with open(x_train) as fileobject:
		for line in fileobject:
			sentence = line.split(' ')
			for word in sentence:
				if word in dictionary:
					dictionary[word] += 1
				else:
					dictionary[word] = 1

	with open(y_train) as fileobject:
		for line in fileobject:
			sentence = line.split(' ')
			for word in sentence:
				if word in dictionary:
					dictionary[word] += 1
				else:
					dictionary[word] = 1

	sorted_dict = sorted(dictionary.items(), key=operator.itemgetter(1), reverse = True)

	return sorted_dict

def create_vocabulary(dictionary, vocab_path, vocab_size):
	vocabulary = open(vocab_path, 'a')
	vocabulary.write('{ _UNK_ : 0' + " \n")
	counter = 0
	for key in sorted_dict:
		vocabulary.write(str(key[0]) + ": " + counter + " \n")
		counter += 1
		if counter > vocab_size:
			break
	vocabulary.write("}")
	vocabulary.close()

sorted_dict = find_dictionary('./x_train_spell_check.txt', './y_train_spell_check.txt')
create_vocabulary(sorted_dict, './vocabulary.txt', 1000)


	


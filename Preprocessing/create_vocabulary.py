import operator, collections
from tensorflow.python.platform import gfile


def find_dictionary(x_train, y_train):
	dictionary = {}

	with open(x_train) as fileobject:
		for line in fileobject:
			sentence = line.strip().split(' ')
			for word in sentence:
				if word.strip() == "":
					continue
				if word in dictionary:
					dictionary[word] += 1
				else:
					dictionary[word] = 1

	with open(y_train) as fileobject:
		for line in fileobject:
			sentence = line.strip().split(' ')
			for word in sentence:
				if word.strip() == "":
					continue
				if word in dictionary:
					dictionary[word] += 1
				else:
					dictionary[word] = 1

	sorted_dict = sorted(dictionary.items(), key=operator.itemgetter(1), reverse = True)
	return sorted_dict




# Finds vocabulary file, returns a dictionary with the word as a key, and occurences as value
def read_vocabulary_from_file(vocabulary_path):
	if gfile.Exists(vocabulary_path):
		rev_vocab = []
		with gfile.GFile(vocabulary_path, mode="rb") as f:
			rev_vocab.extend(f.readlines())
			rev_vocab = [line.strip() for line in rev_vocab]
			vocab = dict([(word, index) for (index, word) in enumerate(rev_vocab)])
		return vocab, rev_vocab
	else:
		raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def encode_sentence(sentence, dictionary, unk_id = 4):
	# Extract first word (and don't add any space)
	if not sentence:
		return ""
	first_word = sentence.pop(0)
	if first_word in dictionary:
		encoded_sentence = str(dictionary[first_word])
	else:
		encoded_sentence = str(unk_id)
	# Loop rest of the words (and add space in front)
	for word in sentence:
		if word in dictionary:
			encoded_word = dictionary[word]
		else:
			encoded_word = unk_id
		encoded_sentence += " " + str(encoded_word)
	return encoded_sentence

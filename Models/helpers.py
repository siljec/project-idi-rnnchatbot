import os
import sys
import re
sys.path.insert(0, '../Preprocessing') # To access methods from another file from another folder
from preprocess import start_preprocessing
from variables import paths_from_model, tokens
from preprocessing3 import distance

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

def replace_misspelled_words_in_sentence(sentence, misspelllings_path):
    dictionary = read_words_from_misspelling_file(misspelllings_path) #get the misspelled words as a dictionary
    tokenized_sentence = sentence.split(' ')
    final_sentence = ""
    for word in tokenized_sentence:
        new_word = replace_misspelled_word_helper(word, dictionary)
        final_sentence += " " + new_word
    return final_sentence


def check_for_needed_files_and_create(vocab_size):
    if not os.path.isdir(paths_from_model['ubuntu']):
        print("Ubuntu Dialogue Corpus not found or is not on the right path. ")
        print('1')
        print('cd out from project-idi-rnnchatbot')
        print('2')
        print('\t git clone https://github.com/rkadlec/ubuntu-ranking-dataset-creator.git')
        print('3')
        print('\t cd ubuntu-ranking-dataset-creator/src')
        print('4')
        print('\t ./generate.sh')

    if not os.path.isfile(paths_from_model['train_path']):
        start_preprocessing(vocab_size)
    if not os.path.isfile(paths_from_model['dev_path']):
        start_preprocessing(vocab_size)
    if not os.path.isfile(paths_from_model['test_path']):
        start_preprocessing(vocab_size)
    if not os.path.isfile(paths_from_model['vocab_path']):
        start_preprocessing(vocab_size)

def preprocess_input(sentence, fast_text_model, vocab):
    emoji_token = " " + tokens['emoji'] + " "
    dir_token = tokens['directory']
    url_token = " " + tokens['url'] + " "

    sentence = sentence.strip().lower()
    sentence = re.sub(' +', ' ', sentence)  # Will remove multiple spaces
    sentence = re.sub('(?<=[a-z])([!?,.])', r' \1', sentence)  # Add space before special characters [!?,.]
    sentence = re.sub(r'(https?://[^\s]+)', url_token, sentence)  # Exchange urls with URL token
    sentence = re.sub(r'((?:^|\s)(?::|;|=)(?:-)?(?:\)|\(|D|P|\|)(?=$|\s))', emoji_token,
                  sentence)  # Exchange smiles with EMJ token NB: Will neither take :) from /:) nor from :)D
    sentence = re.sub('(?<=[a-z])([!?,.])', r' \1', sentence)  # Add space before special characters [!?,.]
    sentence = re.sub('"', '', sentence)  # Remove "
    sentence = re.sub('((\/\w+)|(\.\/\w+)|(\w+(?=(\/))))()((\/)|(\w+)|(\.\w+)|(\w+|\-|\~))+', dir_token,
                  sentence)  # Replace directory-paths
    sentence = re.sub("(?!(')([a-z]{1})(\s))(')(?=\w|\s)", "", sentence)  # Remove ', unless it is like "banana's"
    sentence = replace_misspelled_words_in_sentence(sentence, paths_from_model['misspellings'])

    # Must replace OOV with most similar vocab-words:
    unk_words = {}
    for word in sentence.split(' '):
        if word not in vocab:
            unk_words[word] = fast_text_model[word]

    # Find most similar words
    similar_words = {}
    for unk_word, unk_vector in unk_words.iteritems():
        for key, value in vocab:
            cur_dist = distance(unk_vector, value[0], value[1])
            # Save the word that is most similar
            if cur_dist < min_dist:
                min_dist = cur_dist
                word = key
        similar_words[unk_word] = word

    # Replace words
    for word, similar_word in unk_words.iteritems():
        sentence.replace(word, similar_word)

    return sentence
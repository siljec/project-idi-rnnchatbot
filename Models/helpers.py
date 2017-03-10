import os
import sys
import re
sys.path.insert(0, '../Preprocessing') # To access methods from another file from another folder
from preprocess import start_preprocessing
from variables import paths_from_model, tokens
from preprocessing3 import distance
from variables import tokens

_, UNK_ID = tokens['unk']


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


def check_for_needed_files_and_create():
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
        print("You should start preprocessing")
    if not os.path.isfile(paths_from_model['dev_path']):
        print("You should start preprocessing")
    if not os.path.isfile(paths_from_model['test_path']):
        print("You should start preprocessing")
    if not os.path.isfile(paths_from_model['vocab_path']):
        print("You should start preprocessing")


def preprocess_input(sentence, fast_text_model, vocab):
    emoji_token = " " + tokens['emoji'][0] + " "
    dir_token = tokens['directory'][0]
    url_token = " " + tokens['url'][0] + " "

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
        min_dist = 10
        word = ""
        for key, value in vocab.iteritems():
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

_WORD_SPLIT = re.compile(b"([.,!?\":;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens"""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def sentence_to_token_ids(sentence, vocabulary):
    """Convert a string to list of integers representing token-ids.
    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
    Returns:
    a list of integers, the token-ids for the sentence.
    """
    words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]
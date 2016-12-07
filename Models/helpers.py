import os
import sys
sys.path.insert(0, '../Preprocessing') # To access methods from another file from another folder
from preprocess import generate_all_files

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
    if not os.path.isdir("./../../ubuntu-ranking-dataset-creator"):
        print("Ubuntu Dialogue Corpus not found or is not on the right path. ")
        print('1')
        print('cd out from project-idi-rnnchatbot')
        print('2')
        print('\t git clone https://github.com/rkadlec/ubuntu-ranking-dataset-creator.git')
        print('3')
        print('\t cd ubuntu-ranking-dataset-creator/src')
        print('4')
        print('\t ./generate.sh')

    if not os.path.isfile("./../Preprocessing/shuffled_test_merged.txt"):
        generate_all_files(vocab_size)
    if not os.path.isfile("./../Preprocessing/shuffled_val_merged.txt"):
        generate_all_files(vocab_size)
    if not os.path.isfile("./../Preprocessing/shuffled_train_merged.txt"):
        generate_all_files(vocab_size)
    if not os.path.isfile("./../Preprocessing/vocabulary.txt"):
        generate_all_files(vocab_size)

import time
import sys
import os
from preprocess_helpers import get_time, do_regex_on_line_opensubtitles, replace_mispelled_words_in_file, replace_word_helper, read_words_from_misspelling_file
sys.path.insert(0, '../') # To access methods from another file from another folder
from variables import paths_from_preprocessing_opensubtitles as paths

def replace_mispelled_words_in_file(sentences_array, misspelled_vocabulary):
    dictionary = read_words_from_misspelling_file(misspelled_vocabulary)
    new_sentences_array = []
    for sentence in sentences_array:
        new_sentece = ""
        sentence_array = sentence.split(' ')
        last_word = sentence_array.pop().strip()
        for word in sentence_array:
            new_word = replace_word_helper(word, dictionary)
            new_sentece += new_word + ' '
        last_word = replace_word_helper(last_word, dictionary)
        new_sentece += last_word
        new_sentences_array.append(new_sentece + '\n')

    return new_sentences_array

# Extract dialogs, concatenate sentences in the same turn
def preprocess_training_file(path, x_train_path, y_train_path, misspelled_vocabulary):

    sentences = []
    with open(path) as fileobject:
        for line in fileobject:
            text = do_regex_on_line_opensubtitles(line)
            sentences.append(text)
    sentences = replace_mispelled_words_in_file(sentences, misspelled_vocabulary)
    # Write to files
    with open(x_train_path, 'a') as input_data, open(y_train_path, 'a') as output_data:
        sentences_len = len(sentences)
        input_data.write(sentences[0].strip() + "\n")
        output_data.write(sentences[1].strip() + "\n")

        for i in range(1, sentences_len-1):
            input_data.write(sentences[i].strip() + "\n")
            output_data.write(sentences[i+1].strip() + "\n")
    return len(sentences)


def read_every_data_file_and_create_initial_files(initial_x_file_path, initial_y_file_path, misspelled_vocabulary):
    start_time = time.time()
    number_of_files_read = 0
    number_of_lines = 0
    folder_path = "../../opensubtitles-parser/data"
    for filename in os.listdir(folder_path):
        if filename[-7:]=='raw.txt':
            filename_path = paths['source_folder_root'] + filename
            number_of_files_read += 1
            num_sentences = preprocess_training_file(filename_path, initial_x_file_path, initial_y_file_path, misspelled_vocabulary)
            number_of_lines += num_sentences
            print("Done with filename: " + str(filename) + ", read " + str(number_of_files_read) + " files, processed " + str(number_of_lines) + " sentences")
        else:
            print(filename + " is not preprocessed")

    print "Number of files read: " + str(number_of_files_read)
    print get_time(start_time)

def preprocess1_opensubtitles(spell_checked_data_x_path, spell_checked_data_y_path, misspellings_path):

    # Step 1: Extract dialogs from OpenSubtitles, do regex and replace missppelings
    print('Reading all the files and create initial files...')
    read_every_data_file_and_create_initial_files(initial_x_file_path=spell_checked_data_x_path,
                                                      initial_y_file_path=spell_checked_data_y_path,
                                                      misspelled_vocabulary=misspellings_path
                                                      )

#preprocess1_opensubtitles(paths['spell_checked_data_x_path'], paths['spell_checked_data_y_path'], paths['misspellings_path'])

def create_misspellings():
    contractions_path = './contractions.txt'
    old_misspellings_path = './datafiles/misspellings.txt'
    new_path = '../misspellings.txt'
    unique_words = set()
    with open(old_misspellings_path, 'r') as fileObject, open(new_path, 'a') as newObject:
        for line in fileObject:
            words = line.split()
            key = words[0]
            if key not in unique_words:
                newObject.write(line)
                unique_words.add(words[0])
            else:
                print("double key: " + key)
    with open(contractions_path, 'r') as fileObject, open(new_path, 'a') as newObject:
        for line in fileObject:
            words = line.split()
            key = words[0]
            if key not in unique_words:
                newObject.write(line)
                unique_words.add(words[0])
            else:
                print("double key: " + key)






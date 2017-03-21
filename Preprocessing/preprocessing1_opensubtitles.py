import time
import sys
sys.path.insert(0, '../') # To access methods from another file from another folder

from preprocess_helpers import get_time, do_regex_on_line_opensubtitles, replace_mispelled_words_in_file, replace_word_helper, read_words_from_misspelling_file
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
    print(sentences)
    # Write to files
    with open(x_train_path, 'a') as input_data, open(y_train_path, 'a') as output_data:
        sentences_len = len(sentences)
        input_data.write(sentences[0].strip() + "\n")
        output_data.write(sentences[1].strip() + "\n")

        for i in range(1, sentences_len-1):
            input_data.write(sentences[i].strip() + "\n")
            output_data.write(sentences[i+1].strip() + "\n")


def read_every_data_file_and_create_initial_files(filenames, initial_x_file_path, initial_y_file_path, misspelled_vocabulary):
    start_time = time.time()
    number_of_files_read = 0
    for filename in filenames:
        filename_path = paths['source_folder_root'] + filename
        number_of_files_read += 1
        preprocess_training_file(filename_path, initial_x_file_path, initial_y_file_path, misspelled_vocabulary)
        print("Done with filename: " + str(filename) + ", read " + str(number_of_files_read) + " files")

    print "Number of files read: " + str(number_of_files_read)
    print get_time(start_time)

def preprocess1_opensubtitles(filenames, spell_checked_data_x_path, spell_checked_data_y_path, misspellings_path):

    # Step 1: Extract dialogs from OpenSubtitles, do regex and replace missppelings
    print('Reading all the files and create initial files...')
    read_every_data_file_and_create_initial_files(filenames=filenames,
                                                      initial_x_file_path=spell_checked_data_x_path,
                                                      initial_y_file_path=spell_checked_data_y_path,
                                                      misspelled_vocabulary=misspellings_path
                                                      )
files = ["1081raw.txt", "11raw.txt"]

preprocess1_opensubtitles(files, paths['spell_checked_data_x_path'], paths['spell_checked_data_y_path'], paths['misspellings_path'])

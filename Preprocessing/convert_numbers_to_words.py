from create_vocabulary import read_vocabulary_from_file
import tensorflow as tf
from itertools import izip
from random import shuffle
import glob
import os


def get_printable_sentence(input_list):
    sentence = ""
    for word in input_list:
        if word == '_EOS':
            word = "."
        sentence += word + " "

    # Remove redundant space in the end
    return sentence[:-1]


def read_file_and_convert(x_file_path, y_file_path, vocab_path, num_lines=20):
    _, rev_vocab = read_vocabulary_from_file(vocab_path)
    with open(x_file_path) as x_file, open(y_file_path) as y_file:
        for x_line, y_line in izip(x_file, y_file):
            if num_lines <= 0:
                break
            num_lines -= 1
            x_output = [tf.compat.as_str(rev_vocab[int(output)]) for output in x_line.split()]
            x_output = get_printable_sentence(x_output)

            y_output = [tf.compat.as_str(rev_vocab[int(output)]) for output in y_line.split()]
            y_output = get_printable_sentence(y_output)

            print("\n| Q:| " + x_output + "\n| A:| " + y_output)


def read_single_file_and_convert(file_path, vocab_path):
    file_paths = glob.glob(os.path.join("./stateful/datafiles", "test*"))
    print(file_paths[0])
    shuffle(file_paths)
    _, rev_vocab = read_vocabulary_from_file(vocab_path)
    for file_path in file_paths[:10]:
        with open(file_path) as f:
            for line in f:
                x, y = line.strip().split(",")
                x_output = [tf.compat.as_str(rev_vocab[int(output)]) for output in x.split()]
                x_output = get_printable_sentence(x_output)

                y_output = [tf.compat.as_str(rev_vocab[int(output)]) for output in y.split()]
                y_output = get_printable_sentence(y_output)

                print("\n| Q:| " + x_output + "\n| A:| " + y_output)
            
read_single_file_and_convert('./stateful/datafiles/test100350.txt', "./datafiles/vocabulary.txt")
#read_single_file_and_convert('./stateful/datafiles/test100404.txt', "./datafiles/vocabulary.txt")
#read_single_file_and_convert('./stateful/datafiles/test100990.txt', "./datafiles/vocabulary.txt")
#read_single_file_and_convert('./stateful/datafiles/test100017.txt', "./datafiles/vocabulary.txt")
#read_single_file_and_convert('./stateful/datafiles/test100478.txt', "./datafiles/vocabulary.txt")
#'./stateful/datafiles/test10069.txt', './stateful/datafiles/test100348.txt', './stateful/datafiles/test100577.txt', './stateful/datafiles/test100590.txt', './stateful/datafiles/test100553.txt'
# './stateful/datafiles/test100350.txt', './stateful/datafiles/test100404.txt', './stateful/datafiles/test100990.txt', './stateful/datafiles/test100017.txt', './stateful/datafiles/test100478.txt', './stateful/datafiles/test10006.txt', './stateful/datafiles/test100627.txt', './stateful/datafiles/test100630.txt',

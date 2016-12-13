from create_vocabulary import read_vocabulary_from_file
import tensorflow as tf
from itertools import izip


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


read_file_and_convert("./x_test.txt", "./y_test.txt", "./vocabulary.txt")


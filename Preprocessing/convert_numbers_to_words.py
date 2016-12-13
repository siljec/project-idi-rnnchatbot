from create_vocabulary import read_vocabulary_from_file
import tensorflow as tf


def read_file_and_convert(file_path, vocab_path, num_lines=5):
    _, rev_vocab = read_vocabulary_from_file(vocab_path)
    with open(file_path) as file_object:
        for line in file_object:
            if num_lines <= 0:
                break
            num_lines -= 1
            print(line)
            output = [tf.compat.as_str(rev_vocab[int(output)]) for output in line.split()]
            print(get_printable_sentence(output))


def get_printable_sentence(input_list):
    sentence = ""
    for word in input_list:
        if word == '_EOS':
            word = "."
        sentence += word + " "

    # Remove redundant space in the end
    return sentence[:-1]


read_file_and_convert("./x_test.txt", "./vocabulary.txt")


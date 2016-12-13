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
            output = [tf.compat.as_str(rev_vocab[output]) for output in line.split(" ")]
            print(output)


read_file_and_convert("./Example-Data/x_test.txt", "./Example-Data/vocabulary.txt")


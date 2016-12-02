import operator


def get_stats(path, num_longest=20, more_than_words=100):

    print("############## Stats for " + path + " ##############")

    sentence_lengths = dict()

    with open(path) as file_object:
        longest = [0 for _ in range(num_longest)]
        max_length = 0
        sentences_with_more_than_x_words = 0
        all_lines = file_object.readlines()
        num_lines = len(all_lines)
        num_words = 0
        for line in all_lines:
            length = len(line.split(' '))
            num_words += length
            if length > more_than_words:
                sentences_with_more_than_x_words += 1
            if longest[0] < length:
                longest.pop(0)
                longest.append(length)
                longest.sort()
            if max_length < length:
                max_length = length
            if length in sentence_lengths:
                sentence_lengths[length] += 1
            else:
                sentence_lengths[length] = 1

    print("File: " + path + ". Sentences in total: " + str(num_lines))
    print("Longest sentence: " + str(max_length))

    print("Longest sentences: " + str(longest))

    print("Sentences with more than " + str(more_than_words) + " words: " + str(sentences_with_more_than_x_words))

    type_length, type_num = max(sentence_lengths.iteritems(), key=operator.itemgetter(1))

    print("There are most sentences with length: " + str(type_length) + ". Num sentences: " + str(type_num))

    print("Average length of sentences: " + str(int(num_words/num_lines)))


get_stats('Example-Data/x_train.txt')
get_stats('Example-Data/y_train.txt')
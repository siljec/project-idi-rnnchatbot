import operator


def get_stats(path, num_longest=20, more_than_words=50, less_than_words=5):

    print("############## Stats for " + path + " ##############")

    sentence_lengths = dict()

    with open(path) as file_object:
        longest = [0 for _ in range(num_longest)]
        max_length = 0
        turns_with_more_than_x_words = 0
        turns_with_less_than_x_words = 0
        all_lines = file_object.readlines()
        num_turns = len(all_lines)
        num_words = 0
        for line in all_lines:
            length = len(line.split(' '))
            num_words += length
            if length > more_than_words:
                turns_with_more_than_x_words += 1
            if length < less_than_words:
                turns_with_less_than_x_words += 1
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

    print("File: " + path + ". Turns in total: " + str(num_turns))
    print("Longest sentence: " + str(max_length))

    print("Longest turn: " + str(longest))

    print("Sentences with more than " + str(more_than_words) + " words: " + str(turns_with_more_than_x_words))

    print("Sentences with less than " + str(less_than_words) + " words: " + str(turns_with_less_than_x_words))

    type_length, type_num = max(sentence_lengths.iteritems(), key=operator.itemgetter(1))

    print("There are most turns with length: " + str(type_length) + ". Num turns: " + str(type_num))

    print("Average length of turn: " + str(int(num_words/num_turns)))


get_stats('Example-Data/x_train.txt', more_than_words=40, less_than_words=6)
get_stats('Example-Data/y_train.txt', more_than_words=50, less_than_words=11)
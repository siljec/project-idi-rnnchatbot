import operator


def get_stats(path, num_longest=20, more_than_words=50, less_than_words=5):

    print("\n############## Stats for " + path + " ##############")

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
    print("Longest turn: " + str(max_length))

    print("Longest " + str(num_longest) + " turns: " + str(longest))

    print("Turns with more than " + str(more_than_words) + " words: " + str(turns_with_more_than_x_words))

    print("Turns with less than " + str(less_than_words) + " words: " + str(turns_with_less_than_x_words))

    type_length, type_num = max(sentence_lengths.iteritems(), key=operator.itemgetter(1))

    print("There are most turns with length: " + str(type_length) + ". Num turns: " + str(type_num))

    print("Average length of turn: " + str(int(num_words/num_turns)))


def get_bucket_stats(path, buckets=[(40, 40), (60, 60), (85, 85), (110, 110), (150, 150)]):

    print("\n############## Stats for " + path + " ##############")
    bucket_content = [0 for _ in buckets]
    total_lines = 0
    print("Buckets", buckets)

    with open(path) as file_object:
        for line in file_object:
            total_lines += 1
            # Find correct bucket
            x, y = line.split(',')
            for bucket_id, (source_size, target_size) in enumerate(buckets):
                if len(x) < source_size and len(y) < target_size:
                    bucket_content[bucket_id] += 1
                    break

    print("Occurrences in each bucket: " + str(bucket_content))

    all_turns = sum(bucket_content)
    bucket_content = ["{0:.2f}".format((100.0*num) / total_lines) + "%" for num in bucket_content]

    print("Occurrences in each bucket by percent: " +  str(bucket_content))

    no_match = total_lines - all_turns

    print("Number of turns that did not fit: " + str(no_match) + "/" + str(total_lines) + "\t = " +
          "{0:.2f}".format((100.0*no_match)/total_lines) + "%")


def get_dictionary_stats(x_train, y_train, occurrence=10000):
    print("\n############## Stats for Vocabulary ##############")
    dictionary = {}

    print("Finding words in " + str(x_train))
    with open(x_train) as fileobject:
        for line in fileobject:
            sentence = line.strip().split(' ')
            for word in sentence:
                if word in dictionary:
                    dictionary[word] += 1
                else:
                    dictionary[word] = 1

    print("Finding words in " + str(y_train))
    with open(y_train) as fileobject:
        for line in fileobject:
            sentence = line.strip().split(' ')
            for word in sentence:
                if word in dictionary:
                    dictionary[word] += 1
                else:
                    dictionary[word] = 1

    sorted_dict = sorted(dictionary.items(), key=operator.itemgetter(1), reverse = True)
    counter = 0

    words_represented = 0
    words_not_represented = 0


    for key, value in sorted_dict:
        if counter < 100000:
            words_represented += value
        else:
            words_not_represented += value
        if counter % occurrence == 0:
            print(counter, key, value)
        counter += 1

    # Removing the _EOS_ token. The other tokens are added later
    words_represented -= sorted_dict[0][1]

    print("Words in dictionary occurs " + str(words_represented) + " times")
    print("Words not in dictionary occurs " + str(words_not_represented) + " times")


# get_stats('x_train.txt', more_than_words=40, less_than_words=6)
# get_stats('y_train.txt', more_than_words=50, less_than_words=11)
# get_bucket_stats('train_merged.txt', buckets=[(5, 10), (10, 15), (20, 25), (40, 50)])
get_dictionary_stats('./x_train_spell_check.txt', './y_train_spell_check.txt')
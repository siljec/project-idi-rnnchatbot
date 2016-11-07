
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

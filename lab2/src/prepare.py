import numpy as np
import gensim
import os


# get number of words and maximum sentence length of train, validation and test
def get_statistics(filename):
    tot_words = []
    max_length = 0
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            sentence = line.strip().split()
            for word in sentence[1:]:
                if word not in tot_words:
                    tot_words.append(word)
            if len(sentence[1:]) > max_length:
                max_length = len(sentence) - 1
    return len(tot_words), max_length


# convert words in all sets to integer tokens
def get_word_to_token() -> dict:
    frequency = {}
    word2token = {'<pad>': 0, '<unk>': 1}
    for root, dirs, files in os.walk("../Dataset"):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join("../Dataset", file), 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        sentence = line.strip().split()
                        for word in sentence:
                            try:
                                frequency[word] += 1
                            except KeyError:
                                frequency[word] = 1

    for key in frequency:
        if frequency[key] > 1:
            if key not in word2token:
                word2token[key] = len(word2token)
    return word2token


# convert words in file to pre-trained word vectors
def get_word_to_vector(word2token):
    """
    Format: nparray, word2vector[token] = vector
    Length of a vector is 50
    """
    pre_trained_word = gensim.models.KeyedVectors.load_word2vec_format("../Dataset/wiki_word2vec_50.bin", binary=True)
    word2vec = np.zeros((len(word2token), pre_trained_word.vector_size))
    for key in word2token:
        try:
            word2vec[word2token[key]] = pre_trained_word[key]
        except KeyError:
            word2vec[word2token[key]] = np.random.rand(pre_trained_word.vector_size)
    return word2vec


# convert sentences in file to nparray of word tokens and labels
def get_corpus(filename, word2token, max_length):
    """
    texts = np.array([[sentence1], [sentence2], ..., [sentence n]])
    labels = np.array([label1, label2, ..., label n])
    """
    texts, labels = np.array([0] * max_length), np.array([])
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            sentence = line.strip().split()
            new_text = np.asarray([word2token.get(word, 1) for word in sentence[1:]])[:max_length]
            new_text = np.pad(new_text, (0, max(max_length - len(new_text), 0)), mode='constant', constant_values=0)
            labels = np.append(labels, int(sentence[0]))
            texts = np.vstack((texts, new_text))
    texts = np.delete(texts, 0, 0)
    return texts, labels

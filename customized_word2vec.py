# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import codecs
import gensim
import numpy as np
from gensim.models import *

# Gloabal parameters for gensim word2vec are as follows:
g_embedding_dim = 300  # dimensionality of the feature vectors
g_embedding_model_learning_rate = 0.025  # learning rate, will linearly drop to `min_alpha` as training progresses
g_embedding_window_size = 5  # the maximum distance between the current and predicted word within a sentence.
g_embbeding_min_count_2_ignore = 0  # ignore all words with total frequency lower than this.
g_embbeding_model = 0  # By default (`sg=0`), CBOW is used. Otherwise (`sg=1`), skip-gram is employed.
g_embbeding_hs_setting = 0  # if 1, hierarchical softmax will be used for model training. If set to 0 (default), and `negative` is non-zero, negative sampling will be used.
g_iter_count = 5  # number of iterations (epochs) over the corpus


def load_tokenized_sentences(file_path):
    """
    Load tokenized sentences.
    :param file_path: the file should contain tokenized sentences line by line
    :return:
    """
    sentences = []
    with open(file_path) as f:
        lines = f.readlines()
        for ln in lines:
            sent = [x for x in ln.strip().split(' ')]
            sentences.append(sent)
    return sentences


def generate_word_dictionary(sentences, case_sensitive=True):
    dict_of_words = {}
    if case_sensitive:
        for sent in sentences:
            for word in sent:
                if word in dict_of_words:
                    dict_of_words[word] += 1
                else:
                    dict_of_words[word] = 1
    else:
        for sent in sentences:
            for word in sent:
                wd = word.lower()
                if wd in dict_of_words:
                    dict_of_words[wd] += 1
                else:
                    dict_of_words[wd] = 1

    return dict_of_words


class SimpleWordDictionary(object):
    def __init__(self, sentences=None):
        if not sentences:
            raise Exception("sentences should not be \"None\"")

        # this word dictionay contains items as {word:freq} format
        self.dict_of_words = generate_word_dictionary(sentences)

        # print(len(self.dict_of_words))
        self.indexDict_sorted_by_alphabet, \
        self.indexDict_sorted_by_freq = self.getInsideDictionaries()

    # return the NO. of total words in "WordDictionary"
    def getSizeOfDictionary(self):
        return len(self.dict_of_words)

    def getInsideDictionaries(self):
        # this list contains items as [(word, freq)] format
        list_sorted_by_alphabet = sorted([(word, freq) for (word, freq) in self.dict_of_words.items()])
        indexDict_sorted_by_alphabet = {}
        for (index, (word, freq)) in enumerate(list_sorted_by_alphabet):
            if word in indexDict_sorted_by_alphabet:
                raise Exception("the dictionary already has the word, please check the reason.")

            indexDict_sorted_by_alphabet[word] = index

        # this list contains items as [(freq, word)] format
        list_sorted_by_freq = sorted([(freq, word) for (word, freq) in self.dict_of_words.items()], reverse=True)
        indexDict_sorted_by_freq = {}
        for (index, (freq, word)) in enumerate(list_sorted_by_freq):
            if word in indexDict_sorted_by_freq:
                raise Exception("the dictionary already has the word, please check the reason.")

            indexDict_sorted_by_freq[word] = index

        return indexDict_sorted_by_alphabet, indexDict_sorted_by_freq

    # two kinds of word indices
    def getWordIndices(self, word):
        return self.indexDict_sorted_by_alphabet.get(word, -1), self.indexDict_sorted_by_freq.get(word, -1)


def preprocess_sentences(sentences, case_sensitive=True, adding_start_end_tags=False):
    """
    This function preprocess sentences before training a custom embedding model.

    :param sentences: every sentence is a token sequence
    :param case_sensitive: default=True (means distinguishing word by cases)
    :param adding_start_end_tags: add every sentence with '<START>' tag and '<END>' tag
    :return:
    """

    sentences_after_processing = []
    if case_sensitive:
        sentences_after_processing = sentences[:]
    else:
        for sent in sentences:
            sentences_after_processing.append([word.lower() for word in sent])

    if adding_start_end_tags:
        for sent in sentences_after_processing:
            sent.insert(0, '<START>')
            sent.insert(len(sent), '<END>')

    return sentences_after_processing


def update_existing_embedding_model(embedding_model, new_words_embds):
    """
    :param embedding_model:
    :param new_words_embds: a list, i.e. [[word_text1, embedding1], [word_text2, embedding2], ...]
    :return:
    """
    temp_embds = []
    for tp in new_words_embds:
        word_text = tp[0]
        word_embds = tp[1]
        if word_text in embedding_model.vocab:  # the word already exists in the embedding model
            idx = embedding_model.vocab.get(word_text).index
            embedding_model.syn0[idx] = word_embds
            # raise Exception("the word '%s' is already in the embedding model and cannot be appended." % word_text)
        else:
            word_id = len(embedding_model.vocab)
            embedding_model.vocab[word_text] = gensim.models.word2vec.Vocab(index=word_id, count=0)
            embedding_model.index2word.append(word_text)
            temp_embds.append(word_embds)

    temp_embds = np.array(temp_embds)
    embedding_model.syn0 = np.concatenate((embedding_model.syn0, temp_embds), axis=0)
    return embedding_model


def get_my_extended_word_embedding_model(external_large_word2vec_file, my_small_word2vec_file):
    """
        According to my small trained model, truncate the external large word2vec embedding model to fit a small size.
        Using the embeddings from large corpus to update the customized embedding model

    :param external_large_word2vec_file:
    :param my_small_word2vec_file:
    :return:
    """
    external_large_ebds = Word2Vec.load_word2vec_format(external_large_word2vec_file, binary=True)
    my_local_small_ebds = Word2Vec.load_word2vec_format(my_small_word2vec_file, binary=False)

    nb_of_difference = 0
    for x in my_local_small_ebds.vocab:
        if x in external_large_ebds:
            idx_in_my_small_ebds = my_local_small_ebds.vocab.get(x).index
            my_local_small_ebds.syn0[idx_in_my_small_ebds] = external_large_ebds[x]
        else:
            nb_of_difference += 1
    print("There is/are '%s' different words between the 2 different embedding models" % nb_of_difference)

    dim_size = my_local_small_ebds.syn0.shape[-1]

    # add '<PAD>' word to the embedding model
    padding_word_embedding = np.zeros(dim_size, dtype='float32')  # padding word embedding is 0.0
    if '<PAD>' not in my_local_small_ebds:
        update_existing_embedding_model(my_local_small_ebds, [['<PAD>', padding_word_embedding]])

    # add '<UNKNOWN>' word to the embedding model
    np.random.seed(2000)  # for reproducibility
    unknown_word_embedding = np.random.uniform(-1.0, 1.0, dim_size).astype('float32')  # unknown word embedding
    if '<UNKNOWN>' not in my_local_small_ebds:
        update_existing_embedding_model(my_local_small_ebds, [['<UNKNOWN>', unknown_word_embedding]])

    print('vocabulary size:', len(my_local_small_ebds.vocab))
    return my_local_small_ebds


def generate_embedding_model(vocab, dim=300, scale=1.0):
    vocabulary = list(set([wd for wd in vocab]))
    embd_model = Word2Vec()
    np.random.seed(23455)
    embd_model.syn0 = np.random.uniform(low=-scale, high=scale, size=(len(vocabulary), dim)).astype('float32')
    for word in vocabulary:
        word_id = len(embd_model.vocab)
        embd_model.vocab[word] = gensim.models.word2vec.Vocab(index=word_id, count=0)
        embd_model.index2word.append(word)

    if '<PAD>' not in embd_model:
        update_existing_embedding_model(embd_model, [['<PAD>', np.zeros(dim, dtype='float32')]])

    np.random.seed(2000)
    if '<UNKNOWN>' not in embd_model:
        update_existing_embedding_model(embd_model, [
            ['<UNKNOWN>', np.random.uniform(-scale, scale, dim).astype('float32')]])
    return embd_model


def transfer_text_seq_2_index_seq(text_seq, embedding_model):
    """
    :param text_seq: a list of texts inside the sequence.
    :param embedding_model:
    :return:
    """
    list_of_input_sequence = []
    for i in range(len(text_seq)):
        wd = text_seq[i]
        a_vocab = embedding_model.vocab.get(wd)
        if a_vocab:
            list_of_input_sequence.append(a_vocab.index)
        else:
            list_of_input_sequence.append(
                embedding_model.vocab.get('<UNKNOWN>').index)  # it's a '<UNKNOWN>' word.
    return list_of_input_sequence


def train_a_customized_embedding_model():
    sentences = load_tokenized_sentences("../data/trainingSet_tokenized_sentences.txt")
    sentences.extend(load_tokenized_sentences("../data/developmentSet_tokenized_sentences.txt"))
    sentences.extend(load_tokenized_sentences("../data/testSet_tokenized_sentences.txt"))
    lengths = [len(s) for s in sentences]
    maxlen = np.max(lengths)
    print('The maximum sentence length: ', maxlen)

    # train model
    model = gensim.models.Word2Vec(sentences, size=g_embedding_dim, alpha=g_embedding_model_learning_rate,
                                   window=g_embedding_window_size, min_count=g_embbeding_min_count_2_ignore, workers=4,
                                   sg=g_embbeding_model, hs=g_embbeding_hs_setting, iter=g_iter_count)
    model = update_existing_embedding_model(model, [['<PAD>', np.zeros(g_embedding_dim, dtype='float32')]])

    if g_embbeding_model == 0:
        embedding_file_suffix = 'cbow'
        print('Training mode: CBOW.')
    elif g_embbeding_model == 1:
        embedding_file_suffix = 'skipgram'
        print('Training mode: Skip-gram.')
    else:
        raise Exception('Unknown training mode!')

    model.save_word2vec_format('../data/embeddings.' + str(g_iter_count) + "_" + str(
        g_embedding_dim) + '.' + embedding_file_suffix)

    print('Total word number: ', len(model.vocab))

    # load model
    # mod = Word2Vec.load_word2vec_format(data_dir + 'word_embedings.gensim')

    # continue train the existing model
    # tokenized_sentences = MySentences('tokenizedSentence.txt')  # a memory-friendly iterator
    # model = gensim.models.Word2Vec()  # an empty model, no training
    # model.build_vocab(some_sentences)  # can be a non-repeatable, 1-pass generator
    # model.train(other_sentences)  # can be a non-repeatable, 1-pass generator


if __name__ == '__main__':
    # train_a_customized_embedding_model()

    # md = construct_customized_word_embedding_model('../data/wordvectors/glove.840B.300d.bin', '../data/embeddings.5_300.cbow',
    #                                     'part-of-speech.vocab', 'syntactic.vocab', 'dependency.vocab')

    pass

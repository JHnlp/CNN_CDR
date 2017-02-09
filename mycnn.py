# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals
import numpy as np
import os, sys, codecs, time, pickle, copy, shutil
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv
from gensim.models import *
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from keras import backend as K
from keras.utils import np_utils
from keras.objectives import categorical_crossentropy
from keras.optimizers import *

import basics
import optimizer
from customized_word2vec import *
from layers import EntityLookUpTableLayer, Seq2VecLookUpTableLayer, \
    HiddenLayer, SoftMaxLayer, dropout, LeNetConvPoolLayer, MatrixConvPoolLayer, SimpleConvPoolLayer, \
    Path2VecLookUpTableLayer, MaxMinMeanConvPoolLayer
from evaluation import *

g_sentence_maxlen = 250  
g_distance_vocab_size = g_sentence_maxlen * 2

g_ent_context_size = 5
g_word_represent_win_size = 3  

g_word_embedding_dim = g_embedding_dim
g_distance_embedding_dim = 300  
g_dep_embedding_dim = 100
g_max_len_words_in_between = 150  

g_batch_size = 64
g_nb_epochs = 100
g_nb_filters = 150  
g_patience = 5

g_train_instances_file = "trainingSet.instances"
g_dev_instances_file = "developmentSet.instances"
g_test_instances_file = "testSet.instances"

g_pos_vocab_file = 'part-of-speech.vocab'
g_syn_vocab_file = 'syntactic.vocab'
g_dep_vocab_file = 'dependency.vocab'

g_external_large_word2vec_file = '../data/wordvectors/glove.840B.300d.bin'
g_my_small_word2vec_file = '../data/embeddings.5_300.cbow'


def post_pad_text_seq(text_seq, maxlen_after_padding=150):
    original_length = len(text_seq)
    padded_seq = copy.deepcopy(text_seq)
    if maxlen_after_padding <= original_length:
        padded_seq = padded_seq[:maxlen_after_padding]
    else:
        for k in range(len(padded_seq), maxlen_after_padding):
            padded_seq.append('<PAD>')

    return padded_seq


def construct_customized_word_embedding_model(external_large_word2vec_file, local_small_word2vec_file,
                                              local_pos_vocab_file, local_syn_vocab_file, local_dep_vocab_file):
    external_embeddings = Word2Vec.load_word2vec_format(external_large_word2vec_file, binary=True)
    my_trained_wd_embeddings = Word2Vec.load_word2vec_format(local_small_word2vec_file, binary=False)
    dim = my_trained_wd_embeddings.syn0.shape[-1]

    supplement_vocab = set()
    with codecs.open(local_pos_vocab_file, encoding="utf-8") as f1, \
            codecs.open(local_syn_vocab_file, encoding="utf-8") as f2, \
            codecs.open(local_dep_vocab_file, encoding="utf-8") as f3:
        supplement_vocab.update([v.strip() for v in f1.readlines()])
        supplement_vocab.update([v.strip() for v in f2.readlines()])
        supplement_vocab.update([v.strip() for v in f3.readlines()])

    diff = supplement_vocab.difference(my_trained_wd_embeddings.vocab)
    np.random.seed(23455)
    supplement_embeddings = np.random.uniform(low=-0.5, high=0.5, size=(len(diff), dim)).astype('float32')
    for word in diff:
        word_id = len(my_trained_wd_embeddings.vocab)
        my_trained_wd_embeddings.vocab[word] = gensim.models.word2vec.Vocab(index=word_id, count=0)
        my_trained_wd_embeddings.index2word.append(word)
    my_trained_wd_embeddings.syn0 = np.concatenate((my_trained_wd_embeddings.syn0, supplement_embeddings), axis=0)

    for x in my_trained_wd_embeddings.vocab:
        if x in external_embeddings:
            idx_in_my_small_ebds = my_trained_wd_embeddings.vocab.get(x).index
            my_trained_wd_embeddings.syn0[idx_in_my_small_ebds] = external_embeddings[x]

    padding_word_embedding = np.zeros(dim, dtype='float32') 
    if '<PAD>' not in my_trained_wd_embeddings:
        update_existing_embedding_model(my_trained_wd_embeddings, [['<PAD>', padding_word_embedding]])
    else:
        pad_idx = my_trained_wd_embeddings.vocab.get('<PAD>').index
        my_trained_wd_embeddings.syn0[pad_idx] = np.zeros(dim, dtype='float32')

    np.random.seed(2000) 
    unknown_word_embedding = np.random.uniform(-1.0, 1.0, dim).astype('float32') 
    if '<UNKNOWN>' not in my_trained_wd_embeddings:
        update_existing_embedding_model(my_trained_wd_embeddings, [['<UNKNOWN>', unknown_word_embedding]])

    print('vocabulary size:', len(my_trained_wd_embeddings.vocab))
    return my_trained_wd_embeddings


def generateInputs(instances, word_embedding_model, sentence_maxlen, entity_context_size=3, word_represent_win_size=5):
    batch_of_sents = [] 
    batch_of_wd_pos_seqs = []
    batch_of_distances_2_chem = []
    batch_of_distances_2_dis = []

    batch_of_whether_sent_is_title = []
    batch_of_sent_dist_2_title = []
    batch_of_sent_dist_2_end = []

    entity_maxlen = 50
    nb_wds_beside_ent = entity_context_size // 2
    ent_using_mode = 'IN_ORIGINAL'

    batch_of_synPath_Chem2Root = []
    batch_of_synPath_Dis2Root = []
    batch_of_synPath_Chem2Dis = []

    batch_of_depPath_Root2Chem = []
    batch_of_depPath_Root2Dis = []
    batch_of_depPath_Chem2Dis = []
    batch_of_depPath_Dis2Chem = []

    batch_of_chemical_words_indices = []
    batch_of_disease_words_indices = []
    batch_of_chem_l_wds_indices = []
    batch_of_chem_l_tags_indices = []
    batch_of_chem_r_wds_indices = []
    batch_of_chem_r_tags_indices = []
    batch_of_dis_l_wds_indices = []
    batch_of_dis_l_tags_indices = []
    batch_of_dis_r_wds_indices = []
    batch_of_dis_r_tags_indices = []

    batch_of_in_between_wds_indices = []
    batch_of_in_between_tags_indices = []

    batch_of_in_between_verbs_indices = []

    Y_inputs = []
    for inst in instances:
        if inst.get_relation_type() == "CID":
            Y_inputs.append(1)  
        else:
            Y_inputs.append(0)

        batch_of_whether_sent_is_title.append(1 if inst.chemSentenceIndex == 0 else 0)
        sent_dist_2_title = inst.chemSentenceIndex
        batch_of_sent_dist_2_title.append(sent_dist_2_title)
        sent_dist_2_end = inst.total_sent_num - inst.chemSentenceIndex
        batch_of_sent_dist_2_end.append(sent_dist_2_end)

        batch_of_synPath_Chem2Root.append(
            transfer_text_seq_2_index_seq(post_pad_text_seq(
                inst.synPathFromChem2Root, maxlen_after_padding=110), word_embedding_model))

        batch_of_synPath_Dis2Root.append(
            transfer_text_seq_2_index_seq(post_pad_text_seq(
                inst.synPathFromDis2Root, maxlen_after_padding=110), word_embedding_model))

        batch_of_synPath_Chem2Dis.append(
            transfer_text_seq_2_index_seq(post_pad_text_seq(
                inst.synPathFromChem2Dis, maxlen_after_padding=110), word_embedding_model))

        batch_of_depPath_Root2Chem.append(
            transfer_text_seq_2_index_seq(post_pad_text_seq(
                inst.depPathFromRoot2Chem, maxlen_after_padding=110), word_embedding_model))
        batch_of_depPath_Root2Dis.append(
            transfer_text_seq_2_index_seq(post_pad_text_seq(
                inst.depPathFromRoot2Dis, maxlen_after_padding=110), word_embedding_model))
        batch_of_depPath_Chem2Dis.append(
            transfer_text_seq_2_index_seq(post_pad_text_seq(
                inst.depPathFromChem2Dis, maxlen_after_padding=110), word_embedding_model))

        depPathFromDis2Chem = copy.deepcopy(inst.depPathFromChem2Dis)
        depPathFromDis2Chem.reverse()
        for i in range(len(depPathFromDis2Chem)):
            if depPathFromDis2Chem[i] == '↑':
                depPathFromDis2Chem[i] = '↓'
            elif depPathFromDis2Chem[i] == '↓':
                depPathFromDis2Chem[i] = '↑'
            else:
                pass
        batch_of_depPath_Dis2Chem.append(
            transfer_text_seq_2_index_seq(post_pad_text_seq(depPathFromDis2Chem, maxlen_after_padding=110),
                                          word_embedding_model))

        inst.pad_sentence(sentence_maxlen, padding='post', truncating='post', padding_word='<PAD>')

        padded_tokens = inst.get_token_infos(mode='IN_PADDED')
        padded_words = [t[-1] for t in padded_tokens]
        padded_pos_tags = [t[-2] for t in padded_tokens]

        current_chem_start_tk_idx = inst.get_chemical_start_token_index(mode='IN_PADDED')
        current_chem_last_tk_idx = inst.get_chemical_last_token_index(mode='IN_PADDED')
        current_disease_start_token_index = inst.get_disease_start_token_index(mode='IN_PADDED')
        current_disease_last_token_index = inst.get_disease_last_token_index(mode='IN_PADDED')

        dist_2_chem = basics.get_token_distance_2_entity(padded_words, current_chem_start_tk_idx,
                                                         current_chem_last_tk_idx, sentence_maxlen)
        batch_of_distances_2_chem.append(dist_2_chem)

        dist_2_dis = basics.get_token_distance_2_entity(padded_words, current_disease_start_token_index,
                                                        current_disease_last_token_index, sentence_maxlen)
        batch_of_distances_2_dis.append(dist_2_dis)

        wd_idx_seq = transfer_text_seq_2_index_seq(padded_words, word_embedding_model)
        batch_of_sents.append(wd_idx_seq)

        pos_tag_seq = transfer_text_seq_2_index_seq(padded_pos_tags, word_embedding_model)
        batch_of_wd_pos_seqs.append(pos_tag_seq)

        assert len(dist_2_chem) == len(dist_2_dis) == len(wd_idx_seq) == len(pos_tag_seq)

        original_chem_start_tk_idx = inst.get_chemical_start_token_index(mode=ent_using_mode)
        original_chem_last_tk_idx = inst.get_chemical_last_token_index(mode=ent_using_mode)
        original_disease_start_token_index = inst.get_disease_start_token_index(mode=ent_using_mode)
        original_disease_last_token_index = inst.get_disease_last_token_index(mode=ent_using_mode)

        chem_padded_array = np.array([-1] * entity_maxlen, dtype='int32')
        chem_words_indices = transfer_text_seq_2_index_seq(inst.get_chemical_words(), word_embedding_model)
        chem_padded_array[:len(chem_words_indices)] = chem_words_indices
        batch_of_chemical_words_indices.append(chem_padded_array)

        dis_padded_array = np.array([-1] * entity_maxlen, dtype='int32')
        dis_indices = transfer_text_seq_2_index_seq(inst.get_disease_words(), word_embedding_model)
        dis_padded_array[:len(dis_indices)] = dis_indices
        batch_of_disease_words_indices.append(dis_padded_array)

        chem_l_wds_indices = inst.get_words_indices_beside_given_token(
            word_embedding_model, inst.get_chemical_start_token_index(mode=ent_using_mode), direction='left',
            nb_words=nb_wds_beside_ent, mode=ent_using_mode)
        batch_of_chem_l_wds_indices.append(chem_l_wds_indices)

        chem_l_tags_indices = inst.get_pos_tags_indices_beside_given_token(
            word_embedding_model, inst.get_chemical_start_token_index(mode=ent_using_mode), direction='left',
            nb_words=nb_wds_beside_ent, mode=ent_using_mode)
        batch_of_chem_l_tags_indices.append(chem_l_tags_indices)

        chem_r_wds_indices = inst.get_words_indices_beside_given_token(
            word_embedding_model, inst.get_chemical_last_token_index(mode=ent_using_mode), direction='right',
            nb_words=nb_wds_beside_ent, mode=ent_using_mode)
        batch_of_chem_r_wds_indices.append(chem_r_wds_indices)

        chem_r_tags_indices = inst.get_pos_tags_indices_beside_given_token(
            word_embedding_model, inst.get_chemical_last_token_index(mode=ent_using_mode), direction='right',
            nb_words=nb_wds_beside_ent, mode=ent_using_mode)
        batch_of_chem_r_tags_indices.append(chem_r_tags_indices)

        dis_l_wds_indices = inst.get_words_indices_beside_given_token(
            word_embedding_model, inst.get_disease_start_token_index(mode=ent_using_mode), direction='left',
            nb_words=nb_wds_beside_ent, mode=ent_using_mode)
        batch_of_dis_l_wds_indices.append(dis_l_wds_indices)

        dis_l_tags_indices = inst.get_pos_tags_indices_beside_given_token(
            word_embedding_model, inst.get_disease_start_token_index(mode=ent_using_mode), direction='left',
            nb_words=nb_wds_beside_ent, mode=ent_using_mode)
        batch_of_dis_l_tags_indices.append(dis_l_tags_indices)

        dis_r_wds_indices = inst.get_words_indices_beside_given_token(
            word_embedding_model, inst.get_disease_last_token_index(mode=ent_using_mode), direction='right',
            nb_words=nb_wds_beside_ent, mode=ent_using_mode)
        batch_of_dis_r_wds_indices.append(dis_r_wds_indices)

        dis_r_tags_indices = inst.get_pos_tags_indices_beside_given_token(
            word_embedding_model, inst.get_disease_last_token_index(mode=ent_using_mode), direction='right',
            nb_words=nb_wds_beside_ent, mode=ent_using_mode)
        batch_of_dis_r_tags_indices.append(dis_r_tags_indices)

        original_tokens = inst.get_token_infos(mode=ent_using_mode)
        original_words = [t[-1] for t in original_tokens]
        original_pos_tags = [t[-2] for t in original_tokens]
        if original_chem_start_tk_idx < original_disease_start_token_index:
            fst_ent_last_wd_position = original_chem_last_tk_idx
            snd_ent_start_wd_position = original_disease_start_token_index
        else:
            fst_ent_last_wd_position = original_disease_last_token_index
            snd_ent_start_wd_position = original_chem_start_tk_idx
        words_in_between = basics.get_words_between_entities(original_words, fst_ent_last_wd_position,
                                                             snd_ent_start_wd_position)
        words_in_between = post_pad_text_seq(words_in_between, maxlen_after_padding=g_max_len_words_in_between)
        pos_tags_in_between = basics.get_words_between_entities(
            original_pos_tags, fst_ent_last_wd_position, snd_ent_start_wd_position)
        pos_tags_in_between = post_pad_text_seq(pos_tags_in_between, maxlen_after_padding=g_max_len_words_in_between)
        batch_of_in_between_wds_indices.append(transfer_text_seq_2_index_seq(words_in_between, word_embedding_model))
        batch_of_in_between_tags_indices.append(
            transfer_text_seq_2_index_seq(pos_tags_in_between, word_embedding_model))

        batch_of_in_between_verbs_indices.append(
            transfer_text_seq_2_index_seq(post_pad_text_seq(inst.get_verbs_in_between(), maxlen_after_padding=10),
                                          word_embedding_model))

        assert len(chem_l_wds_indices) == len(chem_l_tags_indices) == len(chem_r_wds_indices) \
               == len(chem_r_tags_indices) == len(dis_l_wds_indices) == len(dis_l_tags_indices) \
               == len(dis_r_wds_indices) == len(dis_r_tags_indices)

    represented_sents = basics.represent_seqs_of_indices_by_context_window(
        word_embedding_model, batch_of_sents, word_represent_win_size)
    X_represented_sentences = theano.shared(np.array(represented_sents, dtype='int32'))
    X_sentences = theano.shared(np.array(batch_of_sents, dtype='int32'))
    X_sent_is_title = theano.shared(np.array(batch_of_whether_sent_is_title, dtype='int32'))
    X_sent_dist_2_title = theano.shared(np.array(batch_of_sent_dist_2_title, dtype='int32'))
    X_sent_dist_2_end = theano.shared(np.array(batch_of_sent_dist_2_end, dtype='int32'))
    X_pos = theano.shared(np.array(batch_of_wd_pos_seqs, dtype='int32'))
    X_dist_2_Chem = theano.shared(np.array(batch_of_distances_2_chem, dtype='int32'))
    X_dist_2_Dis = theano.shared(np.array(batch_of_distances_2_dis, dtype='int32'))

    X_syn_Chem2Root = theano.shared(np.array(batch_of_synPath_Chem2Root, dtype='int32'))
    X_syn_Dis2Root = theano.shared(np.array(batch_of_synPath_Dis2Root, dtype='int32'))
    X_syn_Chem2Dis = theano.shared(np.array(batch_of_synPath_Chem2Dis, dtype='int32'))
    X_dep_Root2Chem = theano.shared(np.array(batch_of_depPath_Root2Chem, dtype='int32'))
    X_dep_Root2Dis = theano.shared(np.array(batch_of_depPath_Root2Dis, dtype='int32'))
    X_dep_Chem2Dis = theano.shared(np.array(batch_of_depPath_Chem2Dis, dtype='int32'))
    X_dep_Dis2Chem = theano.shared(np.array(batch_of_depPath_Dis2Chem, dtype='int32'))

    represented_dep_Root2Chem = basics.represent_seqs_of_indices_by_context_window(
        word_embedding_model, batch_of_depPath_Root2Chem, word_represent_win_size)
    X_represented_dep_Root2Chem = theano.shared(np.array(represented_dep_Root2Chem, dtype='int32'))
    represented_dep_Root2Dis = basics.represent_seqs_of_indices_by_context_window(
        word_embedding_model, batch_of_depPath_Root2Dis, word_represent_win_size)
    X_represented_dep_Root2Dis = theano.shared(np.array(represented_dep_Root2Dis, dtype='int32'))
    represented_dep_Chem2Dis = basics.represent_seqs_of_indices_by_context_window(
        word_embedding_model, batch_of_depPath_Chem2Dis, word_represent_win_size)
    X_represented_dep_Chem2Dis = theano.shared(np.array(represented_dep_Chem2Dis, dtype='int32'))
    represented_dep_Dis2Chem = basics.represent_seqs_of_indices_by_context_window(
        word_embedding_model, batch_of_depPath_Dis2Chem, word_represent_win_size)
    X_represented_dep_Dis2Chem = theano.shared(np.array(represented_dep_Dis2Chem, dtype='int32'))

    X_chemicals = theano.shared(np.array(batch_of_chemical_words_indices, dtype='int32'))
    X_diseases = theano.shared(np.array(batch_of_disease_words_indices, dtype='int32'))
    X_chem_left_wds = theano.shared(np.array(batch_of_chem_l_wds_indices, dtype='int32'))
    X_chem_left_tags = theano.shared(np.array(batch_of_chem_l_tags_indices, dtype='int32'))
    X_chem_right_wds = theano.shared(np.array(batch_of_chem_r_wds_indices, dtype='int32'))
    X_chem_right_tags = theano.shared(np.array(batch_of_chem_r_tags_indices, dtype='int32'))
    X_dis_left_wds = theano.shared(np.array(batch_of_dis_l_wds_indices, dtype='int32'))
    X_dis_left_tags = theano.shared(np.array(batch_of_dis_l_tags_indices, dtype='int32'))
    X_dis_right_wds = theano.shared(np.array(batch_of_dis_r_wds_indices, dtype='int32'))
    X_dis_right_tags = theano.shared(np.array(batch_of_dis_r_tags_indices, dtype='int32'))
    X_in_between_wds = theano.shared(np.array(batch_of_in_between_wds_indices, dtype='int32'))
    X_in_between_tags = theano.shared(np.array(batch_of_in_between_tags_indices, dtype='int32'))
    X_in_between_verbs = theano.shared(np.array(batch_of_in_between_verbs_indices, dtype='int32'))

    Y_categorical = np_utils.to_categorical(Y_inputs, 2)  # categorical Y, for categorical-crossentropy

    Y_inputs = theano.shared(np.array(Y_inputs, dtype='int32'))
    Y_categorical = theano.shared(np.array(Y_categorical, dtype='int32'))

    assert X_sentences.get_value(borrow=True).shape[0] \
           == X_sent_is_title.get_value(borrow=True).shape[0] \
           == X_sent_dist_2_title.get_value(borrow=True).shape[0] \
           == X_sent_dist_2_end.get_value(borrow=True).shape[0] \
           == X_represented_sentences.get_value(borrow=True).shape[0] \
           == X_chemicals.get_value(borrow=True).shape[0] == X_diseases.get_value(borrow=True).shape[0] \
           == X_chem_left_wds.get_value(borrow=True).shape[0] == X_chem_right_wds.get_value(borrow=True).shape[0] \
           == X_dis_left_wds.get_value(borrow=True).shape[0] == X_dis_right_wds.get_value(borrow=True).shape[0] \
           == X_chem_left_tags.get_value(borrow=True).shape[0] == X_chem_right_tags.get_value(borrow=True).shape[0] \
           == X_dis_left_tags.get_value(borrow=True).shape[0] == X_dis_right_tags.get_value(borrow=True).shape[0] \
           == X_in_between_wds.get_value(borrow=True).shape[0] == X_in_between_tags.get_value(borrow=True).shape[0] \
           == X_in_between_verbs.get_value(borrow=True).shape[0] \
           == Y_inputs.get_value(borrow=True).shape[0] == Y_categorical.get_value(borrow=True).shape[0]

    X_represented_sentences = X_represented_sentences.flatten(2)
    X_represented_dep_Root2Chem = X_represented_dep_Root2Chem.flatten(2)
    X_represented_dep_Root2Dis = X_represented_dep_Root2Dis.flatten(2)
    X_represented_dep_Chem2Dis = X_represented_dep_Chem2Dis.flatten(2)
    X_represented_dep_Dis2Chem = X_represented_dep_Dis2Chem.flatten(2)

    return X_sentences, X_sent_is_title, X_sent_dist_2_title, X_sent_dist_2_end, X_pos, X_represented_sentences, X_dist_2_Chem, X_dist_2_Dis, \
           X_syn_Chem2Root, X_syn_Dis2Root, X_syn_Chem2Dis, X_dep_Root2Chem, X_dep_Root2Dis, X_dep_Chem2Dis, X_dep_Dis2Chem, \
           X_represented_dep_Root2Chem, X_represented_dep_Root2Dis, X_represented_dep_Chem2Dis, X_represented_dep_Dis2Chem, \
           X_chemicals, X_diseases, X_chem_left_wds, X_chem_left_tags, X_chem_right_wds, X_chem_right_tags, \
           X_dis_left_wds, X_dis_left_tags, X_dis_right_wds, X_dis_right_tags, X_in_between_wds, X_in_between_tags, X_in_between_verbs, \
           Y_inputs, Y_categorical


class CustomizedModel(object):
    def __init__(self, gold_tr_cid_lines, gold_dev_cid_lines, gold_te_cid_lines,
                 pretrained_wd_embedding_model, dep_embedding_model=None, batch_size=g_batch_size,
                 sent_maxlen=g_sentence_maxlen, dist_vocab_size=g_distance_vocab_size,
                 dist_embedding_dim=g_distance_embedding_dim, entity_context_size=g_ent_context_size,
                 wd_represent_win_size=g_word_represent_win_size, nb_epoch=g_nb_epochs, nb_filters=g_nb_filters,
                 patience=g_patience):
        self.gold_tr_cid_lines = gold_tr_cid_lines
        self.gold_dev_cid_lines = gold_dev_cid_lines
        self.gold_te_cid_lines = gold_te_cid_lines

        self.pretrained_word_embeddings = pretrained_wd_embedding_model
        self.dep_embedding_model = dep_embedding_model
        self.batch_size = batch_size
        self.sentence_maxlen = sent_maxlen
        self.dist_vocab_size = dist_vocab_size
        self.dist_embedding_dim = dist_embedding_dim
        self.ent_context_size = entity_context_size
        self.wd_represent_win_size = wd_represent_win_size
        self.nb_epoch = nb_epoch
        self.nb_filters = nb_filters
        self.patience = patience

        self.variables = None
        self.layers = None
        self.cost = None
        self.accuracy = None

        self.params = None
        self.updates = None

        self.built = False

        print('... building the model')

        wd_Embd_W = theano.shared(value=self.pretrained_word_embeddings.syn0, name='wd_Embd_W', borrow=False)

        self.TH_Phrase = T.iscalar('TH_Phrase')  
        self.TH_Idx = T.iscalar()  

        self.TH_Sent = T.imatrix('TH_Sentence')
        self.TH_SentIsTitle = T.ivector('TH_SentIsTitle')
        self.TH_SentDist2Title = T.ivector('TH_SentDist2Title')
        self.TH_SentDist2End = T.ivector('TH_SentDist2End')
        self.TH_POS = T.imatrix('TH_POS')
        self.TH_RPR_Sent = T.imatrix('TH_Represent_Sent')
        self.TH_WordDist2Chem = T.imatrix('TH_WordDist2Chem')
        self.TH_WordDist2Dis = T.imatrix('TH_Dist2Dis')

        self.TH_SynC2R = T.imatrix('TH_SynChem2Root')
        self.TH_SynD2R = T.imatrix('TH_SynDis2Root')
        self.TH_SynC2D = T.imatrix('TH_SynChem2Dis')
        self.TH_DepR2C = T.imatrix('TH_DepRoot2Chem')
        self.TH_DepR2D = T.imatrix('TH_DepRoot2Dis')
        self.TH_DepC2D = T.imatrix('TH_DepChem2Dis')
        self.TH_DepD2C = T.imatrix('TH_DepDisease2Chemical')

        self.TH_RPR_DepR2C = T.imatrix('TH_Represent_DepR2C')
        self.TH_RPR_DepR2D = T.imatrix('TH_Represent_DepR2D')
        self.TH_RPR_DepC2D = T.imatrix('TH_Represent_DepC2D')
        self.TH_RPR_DepD2C = T.imatrix('TH_Represent_DepD2C')

        self.TH_ChemEnt = T.imatrix('TH_ChemEntity')
        self.TH_DisEnt = T.imatrix('TH_DisEntity')
        self.TH_ChemLWds = T.imatrix('TH_ChemLeftWords')
        self.TH_ChemLTags = T.imatrix('TH_ChemLeftTags')
        self.TH_ChemRWds = T.imatrix('TH_ChemRightWords')
        self.TH_ChemRTags = T.imatrix('TH_ChemRightTags')
        self.TH_DisLWds = T.imatrix('TH_DisLeftWords')
        self.TH_DisLTags = T.imatrix('TH_DisLeftTags')
        self.TH_DisRWds = T.imatrix('TH_DisRightWords')
        self.TH_DisRTags = T.imatrix('TH_DisRightTags')
        self.TH_InBetweenWds = T.imatrix('TH_InBetweenWds')
        self.TH_InBetweenTags = T.imatrix('TH_InBetweenTags')
        self.TH_InBetweenVerbs = T.imatrix('TH_InBetweenVerbs')

        self.TH_Y = T.ivector('TH_Y')
        self.TH_Y_Catigorical = T.imatrix('TH_Y_Catigorical')

        layer_EntLookUp = EntityLookUpTableLayer(inputs=[self.TH_ChemEnt, self.TH_DisEnt], weights=wd_Embd_W)
        ly_WordLookUp = Seq2VecLookUpTableLayer(
            inputs=[self.TH_ChemLWds, self.TH_ChemRWds, self.TH_DisLWds, self.TH_DisRWds,
                    # self.TH_DepR2C, self.TH_DepR2D, self.TH_DepC2D],
                    self.TH_RPR_DepR2C, self.TH_RPR_DepR2D, self.TH_RPR_DepC2D, self.TH_InBetweenVerbs],
            weights=wd_Embd_W)
        ent_info_tensor = T.concatenate(
            [ly_WordLookUp.outputs[0].flatten(2), layer_EntLookUp.outputs[0], ly_WordLookUp.outputs[1].flatten(2),
             ly_WordLookUp.outputs[2].flatten(2), layer_EntLookUp.outputs[1],
             ly_WordLookUp.outputs[3].flatten(2), ly_WordLookUp.outputs[-1].flatten(2)], axis=1)
        tensor_depR2C = ly_WordLookUp.outputs[4]
        tensor_depR2C = tensor_depR2C.reshape((-1, 110, self.wd_represent_win_size * g_word_embedding_dim))
        tensor_depR2D = ly_WordLookUp.outputs[5]
        tensor_depR2D = tensor_depR2D.reshape((-1, 110, self.wd_represent_win_size * g_word_embedding_dim))
        tensor_depC2D = ly_WordLookUp.outputs[6]
        tensor_depC2D = tensor_depC2D.reshape((-1, 110, self.wd_represent_win_size * g_word_embedding_dim))
        dep_tensor = T.concatenate([tensor_depR2C, tensor_depR2D, tensor_depC2D], axis=-1)
        ly_Conv1 = SimpleConvPoolLayer(
            input=dep_tensor, filter_shape=(
                self.nb_filters, 3 * self.wd_represent_win_size * g_word_embedding_dim), activation=T.tanh)
        input_tensor = T.concatenate([ent_info_tensor, ly_Conv1.output.flatten(2)], axis=-1)
        ly_H1 = HiddenLayer(
            inputs=input_tensor,
            n_in=self.nb_filters + g_word_embedding_dim * 2 + self.ent_context_size // 2 * 4 * g_word_embedding_dim + 10 * g_word_embedding_dim,
            n_out=1500, activation=T.tanh)
        input_tensor = T.switch(self.TH_Phrase > 0, dropout(ly_H1.output, n_in=1500, p=0.3), ly_H1.output)
        ly_SoftMax = SoftMaxLayer(input=input_tensor, n_in=1500, n_out=2)
        self.layers = [ly_SoftMax, ly_H1, ly_Conv1, ly_WordLookUp, layer_EntLookUp]
        self.params = ly_SoftMax.params + ly_H1.params + ly_Conv1.params + ly_WordLookUp.params

        L1 = theano.shared(0.)
        L2 = theano.shared(0.)
        for i in range(len(self.params)):
            L1 += T.sum(abs(self.params[i]))  # L1
            L2 += T.sum((self.params[i] ** 2))  # L2
        # self.cost = ly_SoftMax.negative_log_likelihood(TH_Y)
        # self.cost = -T.mean(T.log(ly_SoftMax.p_y_given_x)[T.arange(TH_Y.shape[0]), TH_Y])
        # self.cost = T.mean(K.binary_crossentropy(ly_SoftMax.p_y_given_x[T.arange(TH_Y.shape[0]), TH_Y], TH_Y))
        self.cost = T.mean(K.categorical_crossentropy(
            ly_SoftMax.p_y_given_x, self.TH_Y_Catigorical)) + L2 * 1e-4 / 2.
        self.accuracy = ly_SoftMax.accuracy(self.TH_Y)

        # self.updates = optimizer.sgd(loss=cost, params=params)
        # self.updates = optimizer.adadelta(loss=self.cost, params=self.params)
        # self.updates = optimizer.momentum_grad(loss=self.cost, params=self.params)
        # self.updates = optimizer.adagrad_update(loss=self.cost, params=self.params) 
        # self.updates = optimizer.RMSprop(lr=0.001, loss=self.cost, params=self.params)  
        # self.updates = optimizer.adagrad(lr=0.002, loss=self.cost, params=self.params)  
        # self.updates = SGD(lr=0.01, momentum=0., nesterov=False).get_updates(params=self.params, constraints={}, loss=self.cost)
        # self.updates = Adadelta(lr=0.01, rho=0.95).get_updates(params=self.params, constraints={}, loss=self.cost)
        # self.updates = Adam(lr=0.001).get_updates(params=params, constraints={}, loss=cost)
        # self.updates = Adamax(lr=0.002).get_updates(params=params, constraints={}, loss=cost)
        # self.updates = Nadam(lr=0.002).get_updates(params=params, constraints={}, loss=cost)  
        # self.updates = RMSprop(lr=0.002).get_updates(loss=self.cost, params=self.params, constraints={})  
        self.updates = Adagrad(0.002).get_updates(loss=self.cost, params=self.params, constraints={})  
        self.built = True


    def train_and_test(self, tr_instances_file, dev_instances_file, te_instances_file,
                       our_dev_result_file, our_te_result_file):
        if not self.built:
            raise Exception('Model not ready!')

        training_instances = basics.load_candidate_instances(tr_instances_file)
        np.random.seed(2000)
        np.random.shuffle(training_instances) 
        tr_X_sents, tr_X_sent_is_title, tr_X_sent_dist_2_title, tr_X_sent_dist_2_end, tr_X_pos, tr_X_represented_sentences, tr_X_dist_2_Chem, tr_X_dist_2_Dis, \
        tr_X_syn_Chem2Root, tr_X_syn_Dis2Root, tr_X_syn_Chem2Dis, tr_X_dep_Root2Chem, tr_X_dep_Root2Dis, tr_X_dep_Chem2Dis, tr_X_dep_Dis2Chem, \
        tr_X_rpr_depR2C, tr_X_rpr_depR2D, tr_X_rpr_depC2D, tr_X_rpr_depD2C, \
        tr_X_chemicals, tr_X_diseases, tr_X_chem_lwds, tr_X_chem_ltags, tr_X_chem_rwds, tr_X_chem_rtags, \
        tr_X_dis_lwds, tr_X_dis_ltags, tr_X_dis_rwds, tr_X_dis_rtags, tr_X_in_between_wds, tr_X_in_between_tags, tr_X_in_between_verbs, \
        tr_Y, tr_Y_catigorical = generateInputs(
            training_instances, self.pretrained_word_embeddings, self.sentence_maxlen,
            entity_context_size=self.ent_context_size, word_represent_win_size=self.wd_represent_win_size)

        nb_tr_instances = tr_Y.get_value(borrow=True).shape[0]
        print('training instances number: ', nb_tr_instances)
        tr_batch_size = self.batch_size
        nb_tr_batches = nb_tr_instances // tr_batch_size
        if nb_tr_instances % tr_batch_size > 0:
            nb_tr_batches += 1
        print('There are %d training batches' % nb_tr_batches)

        dev_instances = basics.load_candidate_instances(dev_instances_file)
        dev_X_sentences, dev_X_sent_is_title, dev_X_sent_dist_2_title, dev_X_sent_dist_2_end, dev_X_pos, dev_X_represented_sentences, dev_X_dist_2_Chem, dev_X_dist_2_Dis, \
        dev_X_syn_Chem2Root, dev_X_syn_Dis2Root, dev_X_syn_Chem2Dis, dev_X_dep_Root2Chem, dev_X_dep_Root2Dis, dev_X_dep_Chem2Dis, dev_X_dep_Dis2Chem, \
        dev_X_rpr_depR2C, dev_X_rpr_depR2D, dev_X_rpr_depC2D, dev_X_rpr_depD2C, \
        dev_X_chemicals, dev_X_diseases, dev_X_chem_lwds, dev_X_chem_ltags, dev_X_chem_rwds, dev_X_chem_rtags, \
        dev_X_dis_lwds, dev_X_dis_ltags, dev_X_dis_rwds, dev_X_dis_rtags, dev_X_in_between_wds, dev_X_in_between_tags, dev_X_in_between_verbs, \
        dev_Y, dev_Y_catigorical = generateInputs(
            dev_instances, self.pretrained_word_embeddings, self.sentence_maxlen,
            entity_context_size=self.ent_context_size, word_represent_win_size=self.wd_represent_win_size)

        nb_dev_size = dev_Y.get_value(borrow=True).shape[0]
        print('development instances number: ', nb_dev_size)
        dev_batch_size = self.batch_size
        nb_valid_batches = nb_dev_size // dev_batch_size
        if nb_dev_size % dev_batch_size > 0:
            nb_valid_batches += 1
        print('There are %d valid batches' % nb_valid_batches)

        test_instances = basics.load_candidate_instances(te_instances_file)
        te_X_sentences, te_X_sent_is_title, te_X_sent_dist_2_title, te_X_sent_dist_2_end, te_X_pos, te_X_represented_sentences, te_X_dist_2_Chem, te_X_dist_2_Dis, \
        te_X_syn_Chem2Root, te_X_syn_Dis2Root, te_X_syn_Chem2Dis, te_X_dep_Root2Chem, te_X_dep_Root2Dis, te_X_dep_Chem2Dis, te_X_dep_Dis2Chem, \
        te_X_rpr_depR2C, te_X_rpr_depR2D, te_X_rpr_depC2D, te_X_rpr_depD2C, \
        te_X_chemicals, te_X_diseases, te_X_chem_lwds, te_X_chem_ltags, te_X_chem_rwds, te_X_chem_rtags, \
        te_X_dis_lwds, te_X_dis_ltags, te_X_dis_rwds, te_X_dis_rtags, te_X_in_between_wds, te_X_in_between_tags, te_X_in_between_verbs, \
        te_Y, te_Y_catigorical = generateInputs(
            test_instances, self.pretrained_word_embeddings, self.sentence_maxlen,
            entity_context_size=self.ent_context_size, word_represent_win_size=self.wd_represent_win_size)

        nb_test_size = te_Y.get_value(borrow=True).shape[0]
        print('test instances number: ', nb_test_size)
        te_batch_size = self.batch_size
        nb_test_batches = nb_test_size // te_batch_size
        if nb_test_size % te_batch_size > 0:
            nb_test_batches += 1
        print('There are %d test batches' % nb_test_batches)

        train_model = theano.function(
            inputs=[self.TH_Idx, self.TH_Phrase], outputs=[self.cost, self.accuracy], updates=self.updates,
            givens={
                self.TH_Sent: tr_X_sents[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_SentIsTitle: tr_X_sent_is_title[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_SentDist2Title: tr_X_sent_dist_2_title[
                                        self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_SentDist2End: tr_X_sent_dist_2_end[
                                      self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_POS: tr_X_pos[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_RPR_Sent: tr_X_represented_sentences[
                                  self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_WordDist2Chem: tr_X_dist_2_Chem[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_WordDist2Dis: tr_X_dist_2_Dis[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_SynC2R: tr_X_syn_Chem2Root[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_SynD2R: tr_X_syn_Dis2Root[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_SynC2D: tr_X_syn_Chem2Dis[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_DepR2C: tr_X_dep_Root2Chem[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_DepR2D: tr_X_dep_Root2Dis[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_DepC2D: tr_X_dep_Chem2Dis[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_DepD2C: tr_X_dep_Dis2Chem[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_RPR_DepR2C: tr_X_rpr_depR2C[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_RPR_DepR2D: tr_X_rpr_depR2D[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_RPR_DepC2D: tr_X_rpr_depC2D[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_RPR_DepD2C: tr_X_rpr_depD2C[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_ChemEnt: tr_X_chemicals[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_ChemLWds: tr_X_chem_lwds[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_ChemLTags: tr_X_chem_ltags[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_ChemRWds: tr_X_chem_rwds[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_ChemRTags: tr_X_chem_rtags[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_DisEnt: tr_X_diseases[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_DisLWds: tr_X_dis_lwds[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_DisLTags: tr_X_dis_ltags[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_DisRWds: tr_X_dis_rwds[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_DisRTags: tr_X_dis_rtags[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_InBetweenWds: tr_X_in_between_wds[
                                      self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_InBetweenTags: tr_X_in_between_tags[
                                       self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_InBetweenVerbs: tr_X_in_between_verbs[
                                        self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_Y: tr_Y[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size],
                self.TH_Y_Catigorical: tr_Y_catigorical[self.TH_Idx * tr_batch_size: (self.TH_Idx + 1) * tr_batch_size]
            }, allow_input_downcast=True, on_unused_input='ignore')

        predict_on_dev = theano.function(
            inputs=[self.TH_Idx, self.TH_Phrase],
            outputs=[self.cost, self.accuracy, self.layers[0].y_pred, self.layers[0].y_pred_prob],
            givens={
                self.TH_Sent: dev_X_sentences[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_SentIsTitle: dev_X_sent_is_title[
                                     self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_SentDist2Title: dev_X_sent_dist_2_title[
                                        self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_SentDist2End: dev_X_sent_dist_2_end[
                                      self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_POS: dev_X_pos[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_RPR_Sent: dev_X_represented_sentences[
                                  self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_WordDist2Chem: dev_X_dist_2_Chem[
                                       self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_WordDist2Dis: dev_X_dist_2_Dis[
                                      self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_SynC2R: dev_X_syn_Chem2Root[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_SynD2R: dev_X_syn_Dis2Root[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_SynC2D: dev_X_syn_Chem2Dis[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_DepR2C: dev_X_dep_Root2Chem[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_DepR2D: dev_X_dep_Root2Dis[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_DepC2D: dev_X_dep_Chem2Dis[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_DepD2C: dev_X_dep_Dis2Chem[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_RPR_DepR2C: dev_X_rpr_depR2C[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_RPR_DepR2D: dev_X_rpr_depR2D[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_RPR_DepC2D: dev_X_rpr_depC2D[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_RPR_DepD2C: dev_X_rpr_depD2C[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_ChemEnt: dev_X_chemicals[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_ChemLWds: dev_X_chem_lwds[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_ChemLTags: dev_X_chem_ltags[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_ChemRWds: dev_X_chem_rwds[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_ChemRTags: dev_X_chem_rtags[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_DisEnt: dev_X_diseases[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_DisLWds: dev_X_dis_lwds[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_DisLTags: dev_X_dis_ltags[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_DisRWds: dev_X_dis_rwds[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_DisRTags: dev_X_dis_rtags[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_InBetweenWds: dev_X_in_between_wds[
                                      self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_InBetweenTags: dev_X_in_between_tags[
                                       self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_InBetweenVerbs: dev_X_in_between_verbs[
                                        self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_Y: dev_Y[self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size],
                self.TH_Y_Catigorical: dev_Y_catigorical[
                                       self.TH_Idx * dev_batch_size: (self.TH_Idx + 1) * dev_batch_size]
            }, allow_input_downcast=True, on_unused_input='ignore')

        predict_on_test = theano.function(
            inputs=[self.TH_Idx, self.TH_Phrase],
            outputs=[self.cost, self.accuracy, self.layers[0].y_pred, self.layers[0].y_pred_prob],
            givens={
                self.TH_Sent: te_X_sentences[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_SentIsTitle: te_X_sent_is_title[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_SentDist2Title: te_X_sent_dist_2_title[
                                        self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_SentDist2End: te_X_sent_dist_2_end[
                                      self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_POS: te_X_pos[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_RPR_Sent: te_X_represented_sentences[
                                  self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_WordDist2Chem: te_X_dist_2_Chem[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_WordDist2Dis: te_X_dist_2_Dis[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_SynC2R: te_X_syn_Chem2Root[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_SynD2R: te_X_syn_Dis2Root[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_SynC2D: te_X_syn_Chem2Dis[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_DepR2C: te_X_dep_Root2Chem[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_DepR2D: te_X_dep_Root2Dis[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_DepC2D: te_X_dep_Chem2Dis[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_DepD2C: te_X_dep_Dis2Chem[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_RPR_DepR2C: te_X_rpr_depR2C[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_RPR_DepR2D: te_X_rpr_depR2D[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_RPR_DepC2D: te_X_rpr_depC2D[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_RPR_DepD2C: te_X_rpr_depD2C[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_ChemEnt: te_X_chemicals[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_ChemLWds: te_X_chem_lwds[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_ChemLTags: te_X_chem_ltags[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_ChemRWds: te_X_chem_rwds[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_ChemRTags: te_X_chem_rtags[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_DisEnt: te_X_diseases[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_DisLWds: te_X_dis_lwds[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_DisLTags: te_X_dis_ltags[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_DisRWds: te_X_dis_rwds[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_DisRTags: te_X_dis_rtags[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_InBetweenWds: te_X_in_between_wds[
                                      self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_InBetweenTags: te_X_in_between_tags[
                                       self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_InBetweenVerbs: te_X_in_between_verbs[
                                        self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_Y: te_Y[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size],
                self.TH_Y_Catigorical: te_Y_catigorical[self.TH_Idx * te_batch_size: (self.TH_Idx + 1) * te_batch_size]
            }, allow_input_downcast=True, on_unused_input='ignore')

        print('... training')

        best_models = []
        best_F = 0.
        valid_loss_when_best_F = np.inf
        valid_acc_when_best_F = np.inf

        last_valid_loss = np.inf

        epoch = 0
        done_looping = False
        continous_nb_unimprovement_threshold = self.patience
        unimprovement_accumulation = 0
        last_loop_is_no_good = False

        nb_iters = 0
        max_iters = 250
        while (nb_iters < max_iters) and (not done_looping):
            epoch = epoch + 1

            for minibatch_index in range(nb_tr_batches):
                nb_iters += 1
                if (nb_iters >= max_iters):
                    break

                cost_acc = train_model(minibatch_index, 1)
                # print('training records:["cost": %s, "accuracy": %s]' % (cost_acc[0], cost_acc[1]))

                dev_predictions = [predict_on_dev(i, 0) for i in range(nb_valid_batches)]
                dev_cost = np.mean([v[0] for v in dev_predictions])
                dev_acc = np.mean([v[1] for v in dev_predictions])
                dev_labels = np.concatenate([v[2] for v in dev_predictions])
                dev_probs = np.concatenate([v[3] for v in dev_predictions])
                dev_results = get_CID_prediction_texts(dev_instances, dev_labels, dev_probs)
                dev_eval = evaluate_cid_relations(self.gold_dev_cid_lines, dev_results)
                this_valid_loss = dev_cost
                this_valid_acc = dev_acc

                te_predictions = [predict_on_test(i, 0) for i in range(nb_test_batches)]
                te_labels = np.concatenate([v[2] for v in te_predictions])
                te_probs = np.concatenate([v[3] for v in te_predictions])
                te_results = get_CID_prediction_texts(test_instances, te_labels, te_probs)
                te_eval = evaluate_cid_relations(self.gold_te_cid_lines, te_results)

                print('epoch %i, minibatch %i/%i, validation records:["cost": %f, "accuracy": %f]' %
                      (epoch, minibatch_index + 1, nb_tr_batches, dev_cost, dev_acc))
                print('\t\t... on development: [TP:%d, FP:%d, FN:%d, P:%f, R:%f, F:%f]' % dev_eval)
                print('\t\t... on test: [TP:%d, FP:%d, FN:%d, P:%f, R:%f, F:%f]' % te_eval)

                if this_valid_loss > last_valid_loss * 0.9995:  # * improvement_threshold:
                    print('@@@@@@@@@@@ NO IMPROVEMENT IN VALIDATION @@@@@@@@@@@\n')
                    if last_loop_is_no_good:
                        unimprovement_accumulation += 1
                    else:
                        unimprovement_accumulation = 1
                        last_loop_is_no_good = True
                else:
                    if dev_eval[-1] > best_F:
                        print('. . . saving the best params\n')
                        best_F = dev_eval[-1]
                        valid_loss_when_best_F = this_valid_loss
                        valid_acc_when_best_F = this_valid_acc
                        best_params = copy.deepcopy([x.get_value(borrow=True) for x in self.params])
                        best_models.append([valid_loss_when_best_F, valid_acc_when_best_F, best_F, best_params])
                        if len(best_models) > 10:
                            best_models.pop(0)
                    last_loop_is_no_good = False
                last_valid_loss = this_valid_loss
                last_valid_acc = this_valid_acc

                if unimprovement_accumulation == continous_nb_unimprovement_threshold:
                    done_looping = True
                    break

        print('best training ["cost": %f, "accuracy": %f]' % (valid_loss_when_best_F, valid_acc_when_best_F))
        best_performance = 0
        idx_when_best_performance = 0
        for i in range(len(best_models)):
            if best_models[i][2] > best_performance:
                best_performance = best_models[i][2]
                idx_when_best_performance = i
        best_params = best_models[idx_when_best_performance][3]
        print('choosing model ["cost": %f, "accuracy": %f, "F1": %f]' % (
            best_models[idx_when_best_performance][0], best_models[idx_when_best_performance][1],
            best_models[idx_when_best_performance][2]))

        if len(best_params) > 0:
            for i in range(len(self.params)):
                self.params[i].set_value(best_params[i], borrow=True)

        predictions = [predict_on_dev(i, 0) for i in range(nb_valid_batches)]
        labels = np.concatenate([p[2] for p in predictions])
        probs = np.concatenate([p[3] for p in predictions])
        dev_final_results = get_CID_prediction_texts(dev_instances, labels, probs)
        with codecs.open(our_dev_result_file, mode='w', encoding="utf-8")as out:
            for x in dev_final_results:
                out.write(x + "\n")
        dev_eval = evaluate_cid_relations(self.gold_dev_cid_lines, dev_final_results)
        print('TP: %d\nFP: %d\nFN: %d\nP: %f\nR: %f\nF: %f\n\n' % dev_eval)

        predictions = [predict_on_test(i, 0) for i in range(nb_test_batches)]
        labels = np.concatenate([p[2] for p in predictions])
        probs = np.concatenate([p[3] for p in predictions])
        te_final_results = get_CID_prediction_texts(test_instances, labels, probs)
        with codecs.open(our_te_result_file, mode='w', encoding="utf-8")as out:
            for x in te_final_results:
                out.write(x + "\n")
        te_eval = evaluate_cid_relations(self.gold_te_cid_lines, te_final_results)
        print('TP: %d\nFP: %d\nFN: %d\nP: %f\nR: %f\nF: %f\n\n' % te_eval)


def get_CID_prediction_texts(instances, labels, probs):
    """

    :param instances:
    :param labels: numpy array
    :param probs: numpy array
    :return:
    """
    assert len(instances) == len(labels) == len(probs)

    docID_chemID_disID = [[ist.get_document_id(), ist.get_chemical_concept_id(), ist.get_disease_concept_id()]
                          for ist in instances]
    cid_labels = []
    for lb, prob in zip(labels, probs):
        if lb == 0:
            label = 'UN'
        else:
            label = 'CID'
        cid_labels.append((label, prob))

    rel_prob = dict()
    for ((lb, prob), (docID, chemID, disID)) in zip(cid_labels, docID_chemID_disID):
        if lb == 'CID':
            key = docID + '\t' + lb + '\t' + chemID + "\t" + disID
            if key in rel_prob:
                if prob > rel_prob[key]:
                    rel_prob[key] = prob
            else:
                rel_prob[key] = prob
    results = set()
    for k, v in rel_prob.items():
        results.add(k + "\t" + str(v))
    return results


def load_gold_cid_annotation(gold_tr_file):
    with codecs.open(gold_tr_file, encoding="utf-8") as gf_tr:
        gold_tr_lines = gf_tr.readlines()
        gold_tr_cid_lines = set()
        for ln in gold_tr_lines:
            if len(ln.split("\t")) == 4:
                gold_tr_cid_lines.add(ln.strip())
    print('Gold training set CIDs: %d' % len(gold_tr_cid_lines))
    return gold_tr_cid_lines


def evaluate_cid_relations(gold_CIDs, our_CIDs):
    gold = set(gold_CIDs)
    ours = set(our_CIDs)

    if len(ours) == 0:
        return 0, 0, 0, 0., 0., np.nan

    ours_without_prob = set()
    for r in ours:
        if len(r.split('\t')) == 5:
            ours_without_prob.add(r[:r.strip().rfind('\t')])
        else:
            ours_without_prob.add(r)

    nb_TP = len(ours_without_prob.intersection(gold))
    nb_FP = len(ours_without_prob) - nb_TP
    nb_FN = len(gold) - nb_TP
    p = float(nb_TP) / float(len(ours_without_prob))
    r = float(nb_TP) / float(len(gold))

    if p + r == 0.:
        return nb_TP, nb_FP, nb_FN, p, r, np.nan

    f = 2 * p * r / (p + r)
    # print(nb_TP, nb_FP, nb_FN, p, r, f)
    return nb_TP, nb_FP, nb_FN, p, r, f


if __name__ == '__main__':
    start_time = time.time()
    gd_tr = load_gold_cid_annotation('../data/Corpus/CDR_TrainingSet.PubTator.txt')
    gd_dev = load_gold_cid_annotation('../data/Corpus/CDR_DevelopmentSet.PubTator.txt')
    gd_te = load_gold_cid_annotation('../data/Corpus/CDR_TestSet.PubTator.txt')

    pretrained_word_embeddings = construct_customized_word_embedding_model(
        g_external_large_word2vec_file, g_my_small_word2vec_file, g_pos_vocab_file, g_syn_vocab_file, g_dep_vocab_file)
    my_model = CustomizedModel(gd_tr, gd_dev, gd_te,
                               pretrained_word_embeddings, batch_size=g_batch_size, sent_maxlen=250,
                               dist_vocab_size=g_distance_vocab_size,
                               dist_embedding_dim=300, entity_context_size=5, wd_represent_win_size=9, nb_epoch=10,
                               nb_filters=300, patience=5)
    my_model.train_and_test(g_train_instances_file, g_dev_instances_file, g_test_instances_file,
                            '../Different_Trials/cid_results.dev.txt', '../Different_Trials/cid_results.test.txt')

    end_time = time.time()
    print(end_time - start_time)

    print('*********** End **********')
    pass

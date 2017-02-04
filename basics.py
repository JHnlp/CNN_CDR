# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import codecs
import copy
import time

from customized_word2vec import *


class CandidateInstance(object):
    def __init__(self, rel, documentID, total_sent_num, chem_concept_id, chem_sentence_idx,
                 chem_start_offset_in_doc, chem_end_offset_in_doc, chem_first_token_idx, chem_last_token_idx, chem_text,
                 dis_concept_id, dis_sentence_idx, dis_start_offset_in_doc, dis_end_offset_in_doc, dis_first_token_idx,
                 dis_last_token_idx, dis_text, original_tokenized_sentence_text, original_token_infos,
                 mention_anonymous_parsing_result, synPathFromChem2Root, synPathFromDis2Root, synPathFromChem2Dis,
                 depPathFromRoot2Chem, depPathFromRoot2Dis, depPathFromChem2Dis):
        """
        :param rel:
        :param documentID:
        :param total_sent_num: total number of sentences in the document
        :param chem_concept_id:
        :param chem_sentence_idx:
        :param chem_start_offset_in_doc:
        :param chem_end_offset_in_doc:
        :param chem_first_token_idx:
        :param chem_last_token_idx:
        :param dis_concept_id:
        :param dis_sentence_idx:
        :param dis_start_offset_in_doc:
        :param dis_end_offset_in_doc:
        :param dis_first_token_idx:
        :param dis_last_token_idx:
        :param dis_text:
        :param original_tokenized_sentence_text: There is one sentence for cooccurrence relation candidates.
        :param original_token_infos: The sentence would have a sequence of token infos.
        """
        # ********** in original
        self.relation = rel
        self.documentID = documentID
        self.total_sent_num = int(total_sent_num)

        self.chemConceptID = chem_concept_id
        self.chemSentenceIndex = int(chem_sentence_idx)
        self.chemStartOffInDoc = int(chem_start_offset_in_doc)
        self.chemEndOffInDoc = int(chem_end_offset_in_doc)
        self.chemicalStartTokenIndex = int(chem_first_token_idx)
        self.chemicalLastTokenIndex = int(chem_last_token_idx)
        self.chemicalText = chem_text

        self.disConceptID = dis_concept_id
        self.disStartOffInDoc = int(dis_start_offset_in_doc)
        self.disSentenceIndex = int(dis_sentence_idx)
        self.disEndOffInDoc = int(dis_end_offset_in_doc)
        self.diseaseStartTokenIndex = int(dis_first_token_idx)
        self.diseaseLastTokenIndex = int(dis_last_token_idx)
        self.diseaseText = dis_text
        self.originalTokenizedSentenceText = original_tokenized_sentence_text

        # a token info means: [docid, sent_idx, start_off_in_sent, end_off_in_sent, token_idx, token_pos, token_text]
        self.originalTokenInfos = original_token_infos

        self.mentionAnonymousParsingResult = mention_anonymous_parsing_result
        self.synPathFromChem2Root = synPathFromChem2Root
        self.synPathFromDis2Root = synPathFromDis2Root
        self.synPathFromChem2Dis = synPathFromChem2Dis
        self.depPathFromRoot2Chem = depPathFromRoot2Chem
        self.depPathFromRoot2Dis = depPathFromRoot2Dis
        self.depPathFromChem2Dis = depPathFromChem2Dis

        self.has_been_padded = False
        self.padded_token_infos = None  # token infos after padding
        self.chem_start_token_idx_after_padding = None
        self.chem_last_token_idx_after_padding = None
        self.dis_start_token_idx_after_padding = None
        self.dis_last_token_idx_after_padding = None

    def get_document_id(self):
        return self.documentID

    def get_relation_type(self):
        return self.relation

    def get_chemical_concept_id(self):
        return self.chemConceptID

    def get_disease_concept_id(self):
        return self.disConceptID

    def get_chemical_sentence_index(self):
        return self.chemSentenceIndex

    def get_disease_sentence_index(self):
        return self.disSentenceIndex

    def get_chemical_words(self):
        chemical_token_infos = self.originalTokenInfos[
                               self.chemicalStartTokenIndex:self.chemicalLastTokenIndex + 1]
        chemical_words = [tk_info[-1] for tk_info in chemical_token_infos]
        return chemical_words

    def get_chemical_start_token_index(self, mode='IN_PADDED'):
        if mode not in {'IN_ORIGINAL', 'IN_PADDED'}:
            raise Exception('Mode cannot be understood!')
        if mode == 'IN_PADDED':
            if not self.has_been_padded:
                raise Exception('This instance has not been padded!')
            return self.chem_start_token_idx_after_padding
        else:
            return self.chemicalStartTokenIndex

    def get_chemical_last_token_index(self, mode='IN_PADDED'):
        if mode not in {'IN_ORIGINAL', 'IN_PADDED'}:
            raise Exception('Mode cannot be understood!')
        if mode == 'IN_PADDED':
            if not self.has_been_padded:
                raise Exception('This instance has not been padded!')
            return self.chem_last_token_idx_after_padding
        else:
            return self.chemicalLastTokenIndex

    def get_disease_start_token_index(self, mode='IN_PADDED'):
        if mode not in {'IN_ORIGINAL', 'IN_PADDED'}:
            raise Exception('Mode cannot be understood!')
        if mode == 'IN_PADDED':
            if not self.has_been_padded:
                raise Exception('This instance has not been padded!')
            return self.dis_start_token_idx_after_padding
        else:
            return self.diseaseStartTokenIndex

    def get_disease_last_token_index(self, mode='IN_PADDED'):
        if mode not in {'IN_ORIGINAL', 'IN_PADDED'}:
            raise Exception('Mode cannot be understood!')
        if mode == 'IN_PADDED':
            if not self.has_been_padded:
                raise Exception('This instance has not been padded!')
            return self.dis_last_token_idx_after_padding
        else:
            return self.diseaseLastTokenIndex

    def get_disease_words(self):
        disease_token_infos = self.originalTokenInfos[
                              self.diseaseStartTokenIndex:self.diseaseLastTokenIndex + 1]
        disease_words = [tk_info[-1] for tk_info in disease_token_infos]
        return disease_words

    def get_chemical_entity_embedding(self, gensim_embedding_model):
        chemical_words = self.get_chemical_words()
        chem_embedding = np.zeros(g_embedding_dim, dtype='float32')
        for tk in chemical_words:
            chem_embedding += gensim_embedding_model[tk]
        return chem_embedding / float(len(chemical_words))

    def get_disease_entity_embedding(self, gensim_embedding_model):
        disease_words = self.get_disease_words()
        dis_embedding = np.zeros(g_embedding_dim, dtype='float32')
        for tk in disease_words:
            dis_embedding += gensim_embedding_model[tk]
        return dis_embedding / float(len(disease_words))

    def get_tokenized_sentence_text(self, mode='IN_PADDED'):
        if mode not in {'IN_ORIGINAL', 'IN_PADDED'}:
            raise Exception('Mode cannot be understood!')
        if mode == 'IN_PADDED':
            if not self.has_been_padded:
                raise Exception('This instance has not been padded!')
            return [t[-1] for t in self.padded_token_infos]
        else:
            return self.originalTokenizedSentenceText.split(' ')

    def get_token_infos(self, mode='IN_PADDED'):
        if mode not in {'IN_ORIGINAL', 'IN_PADDED'}:
            raise Exception('Mode cannot be understood!')
        if mode == 'IN_PADDED':
            if not self.has_been_padded:
                raise Exception('This instance has not been padded!')
            return self.padded_token_infos
        else:
            return self.originalTokenInfos

    def get_verbs_in_between(self):
        if self.chemicalLastTokenIndex < self.diseaseStartTokenIndex:
            start = self.chemicalLastTokenIndex + 1
            end = self.diseaseStartTokenIndex
        elif self.diseaseLastTokenIndex < self.chemicalStartTokenIndex:
            start = self.diseaseLastTokenIndex + 1
            end = self.chemicalStartTokenIndex
        else:
            raise Exception('Mentions may overlap!')

        verbs_in_between = [self.originalTokenInfos[i][-1] for i in range(start, end) if
                            self.originalTokenInfos[i][-2].startswith('VB')]

        return verbs_in_between

    def pad_sentence(self, maxlen_after_padding, padding='post', truncating='post', padding_word='<PAD>'):
        """
        Pad sentences within one candidate instance.
        A token sequence means one sentence, consisting of multiple token infos.
        A token info means: [docid, sent_idx, start_off_in_sent, end_off_in_sent, token_idx, pos, token_text]

        :param maxlen_after_padding:
        :param padding:
        :param truncating:
        :param padding_word:
        :return: return words list after padding.
        """
        # Notes: cooccurrence means there is only one sentence inside the candidate instance

        if self.has_been_padded: return

        chem_start_token_idx_before_padding = int(self.chemicalStartTokenIndex)
        chem_last_token_idx_before_padding = int(self.chemicalLastTokenIndex)
        dis_start_token_idx_before_padding = int(self.diseaseStartTokenIndex)
        dis_last_token_idx_before_padding = int(self.diseaseLastTokenIndex)
        token_infos = self.get_token_infos(mode='IN_ORIGINAL')
        length = len(token_infos)

        # a deep copy
        padded_token_infos_sequence = copy.deepcopy(token_infos)

        self.chem_start_token_idx_after_padding = None
        self.chem_last_token_idx_after_padding = None
        self.dis_start_token_idx_after_padding = None
        self.dis_last_token_idx_after_padding = None

        if maxlen_after_padding <= length:
            if truncating == 'pre':
                padded_token_infos_sequence = padded_token_infos_sequence[-maxlen_after_padding:]
            elif truncating == 'post':
                padded_token_infos_sequence = padded_token_infos_sequence[:maxlen_after_padding]
            else:
                raise ValueError('Truncating type "%s" cannot be understood!' % truncating)
        else:
            if padding == 'post':
                for k in range(len(padded_token_infos_sequence), maxlen_after_padding):
                    a_end_tag_token = [self.get_document_id(), self.get_chemical_sentence_index(),
                                       - 1, -1, -1, padding_word, padding_word]
                    padded_token_infos_sequence.append(a_end_tag_token)
            elif padding == 'pre':
                for k in range(len(padded_token_infos_sequence), maxlen_after_padding):
                    a_end_tag_token = [self.get_document_id(), self.get_chemical_sentence_index(),
                                       - 1, -1, -1, padding_word, padding_word]
                    padded_token_infos_sequence.insert(0, a_end_tag_token)
            else:
                raise ValueError('Padding type "%s" not understood' % padding)

                # update the current indices for all tokens
        for idx, t_info in enumerate(padded_token_infos_sequence):
            token_idx = int(t_info[-3])
            if token_idx < 0:  # this means some padded tokens.
                continue

            if int(token_idx) == chem_start_token_idx_before_padding:
                self.chem_start_token_idx_after_padding = int(idx)
            if int(token_idx) == chem_last_token_idx_before_padding:
                self.chem_last_token_idx_after_padding = int(idx)
            if int(token_idx) == dis_start_token_idx_before_padding:
                self.dis_start_token_idx_after_padding = int(idx)
            if int(token_idx) == dis_last_token_idx_before_padding:
                self.dis_last_token_idx_after_padding = int(idx)

        if self.chem_start_token_idx_after_padding is None \
                or self.chem_last_token_idx_after_padding is None \
                or self.dis_start_token_idx_after_padding is None \
                or self.dis_last_token_idx_after_padding is None:
            raise Exception('Entity is missing after padding!')
        self.has_been_padded = True
        self.padded_token_infos = padded_token_infos_sequence

    def has_been_padded(self):
        return self.has_been_padded

    def get_words_beside_given_token(self, wd_pos_idx, direction='left', nb_words=1, mode='IN_PADDED'):
        """
        :param wd_pos_idx: the index of a given word in the sequence.
        :param direction: left or right
        :param nb_words: the number of words needed beside the given token
        :param mode:
        :return: return n left or right word text beside the given word, padding the empty position with '<PAD>'
        """

        if mode not in {'IN_ORIGINAL', 'IN_PADDED'}:
            raise Exception('Mode cannot be understood!')
        if mode == 'IN_PADDED':
            if not self.has_been_padded:
                raise Exception('This instance has not been padded!')
            token_infos = self.padded_token_infos
        else:
            token_infos = self.originalTokenInfos

        if wd_pos_idx < 0 or wd_pos_idx > len(token_infos):
            raise ValueError('the given word index "%s" is out of range!' % wd_pos_idx)

        wds = []
        if direction == 'left':
            for i in range(1, nb_words + 1):
                if wd_pos_idx - i > 0:
                    wds.append(token_infos[wd_pos_idx - i][-1])
                else:
                    wds.append('<PAD>')
            wds.reverse()
        elif direction == 'right':
            for i in range(1, nb_words + 1):
                if wd_pos_idx + i < len(token_infos):
                    wds.append(token_infos[wd_pos_idx + i][-1])
                else:
                    wds.append('<PAD>')
        else:
            raise ValueError('Direction type "%s" not understood' % direction)
        return wds

    def get_words_indices_beside_given_token(self, word_embedding_model, wd_pos_idx,
                                             direction='left', nb_words=1, mode='IN_PADDED'):
        words = self.get_words_beside_given_token(wd_pos_idx, direction=direction, nb_words=nb_words, mode=mode)
        return transfer_text_seq_2_index_seq(words, word_embedding_model)

    def get_pos_tags_beside_given_token(self, wd_pos_idx, direction='left', nb_words=1, mode='IN_PADDED'):
        """
        :param wd_pos_idx: the index of a given word in the sequence.
        :param direction: left or right
        :param nb_words: the number of words needed
        :param mode:
        :return: return n left or right word tags beside the given word, padding the empty position with '<PAD>'
        """

        if mode not in {'IN_ORIGINAL', 'IN_PADDED'}:
            raise Exception('Mode cannot be understood!')
        if mode == 'IN_PADDED':
            if not self.has_been_padded:
                raise Exception('This instance has not been padded!')
            token_infos = self.padded_token_infos
        else:
            token_infos = self.originalTokenInfos

        if wd_pos_idx < 0 or wd_pos_idx > len(token_infos):
            raise ValueError('the given word index "%s" is out of range!' % wd_pos_idx)

        tags = []
        if direction == 'left':
            for i in range(1, nb_words + 1):
                if wd_pos_idx - i > 0:
                    tags.append(token_infos[wd_pos_idx - i][-2])
                else:
                    tags.append('<PAD>')
            tags.reverse()
        elif direction == 'right':
            for i in range(1, nb_words + 1):
                if wd_pos_idx + i < len(token_infos):
                    tags.append(token_infos[wd_pos_idx + i][-2])
                else:
                    tags.append('<PAD>')
        else:
            raise ValueError('Direction type "%s" not understood' % direction)
        return tags

    def get_pos_tags_indices_beside_given_token(self, tag_embedding_model, wd_pos_idx,
                                                direction='left', nb_words=1, mode='IN_PADDED'):
        tags = self.get_pos_tags_beside_given_token(wd_pos_idx, direction=direction, nb_words=nb_words, mode=mode)
        return transfer_text_seq_2_index_seq(tags, tag_embedding_model)


def get_token_distance_2_entity(text_seq, entity_start_token_idx, entity_last_token_idx, max_distance):
    """
    The input sentence should be padded first.
    Get token distance according to the given word sequence.
    :param text_seq: words sequence
    :param entity_start_token_idx: current start token index of a entity in the padded sequence
    :param entity_last_token_idx: current last token index of a entity in the padded sequence
    :param max_distance:
    :return:
    """
    sentence = text_seq
    if max_distance < 0:
        raise Exception("The unreachable distance '%s' should be larger than 0!" % max_distance)

    if not (0 <= entity_start_token_idx <= entity_last_token_idx < len(sentence)):
        raise Exception("The entity token indices are wrong!")

    list_of_token_distance = []
    for idx in range(len(sentence)):
        # the word is inside the entity
        if entity_start_token_idx <= idx <= entity_last_token_idx:
            list_of_token_distance.append(0)

        # the word is on the left side of the entity
        if idx < entity_start_token_idx:
            dist = idx - int(entity_start_token_idx)
            if sentence[idx] == '<PAD>':  # given the max_distance to the padding word (on the left).
                dist = -max_distance
            list_of_token_distance.append(dist)

        if entity_last_token_idx < idx:
            dist = idx - int(entity_last_token_idx)
            if sentence[idx] == '<PAD>':  # given the max_distance to the padding word (on the right).
                dist = max_distance
            list_of_token_distance.append(dist)

    return list_of_token_distance


def get_words_between_entities(text_seq, fst_ent_last_tk_idx, snd_ent_start_tk_idx):
    """
    Get words in between.
    :param text_seq: words sequence
    :param fst_ent_last_tk_idx: the last word position of the first entity in the seq
    :param snd_ent_start_tk_idx: the start word position of the second entity in the seq
    :return:
    """
    seq = copy.deepcopy(text_seq)
    if not (0 <= fst_ent_last_tk_idx <= snd_ent_start_tk_idx < len(seq)):
        raise Exception("The entity token indices are wrong!")

    return seq[fst_ent_last_tk_idx + 1: snd_ent_start_tk_idx]


def load_candidate_instances(instances_file):
    insts = []

    with codecs.open(instances_file, encoding="utf-8") as f1:
        inst_lines = f1.readlines()

        for i in range(len(inst_lines)):
            parts = inst_lines[i].strip('\n').split("\t")
            relTag = parts[0]
            chem_dis_men_infos = [p for p in parts[1].split("_")]
            tokenized_sentence_text = parts[2]

            tokens_infos = []  # corresponding to a sentence.
            for part in parts[3].split("  "):
                x = part[6:-1]  # a token format: 'Token[10027919 16 0 11 0 NNS CONCLUSIONS]'
                component = x.split(' ')

                # token information format: docid, sent_idx, start_off_in_sent, end_off_in_sent, token_idx, part-of-speech, token_text
                token = [component[0], component[1], component[2], component[3],
                         component[4], component[5], component[6]]
                tokens_infos.append(token)
            bllipparsing_result = parts[4]
            synChem2Root = parts[5].split(' ')
            synDis2Root = parts[6].split(' ')
            synChem2Dis = parts[7].split(' ')
            depRoot2Chem = parts[8].split(' ')
            depRoot2Dis = parts[9].split(' ')
            depChem2Dis = parts[10].split(' ')

            instance = CandidateInstance(relTag, *chem_dis_men_infos,
                                         original_tokenized_sentence_text=tokenized_sentence_text,
                                         original_token_infos=tokens_infos,
                                         mention_anonymous_parsing_result=bllipparsing_result,
                                         synPathFromChem2Root=synChem2Root,
                                         synPathFromDis2Root=synDis2Root,
                                         synPathFromChem2Dis=synChem2Dis,
                                         depPathFromRoot2Chem=depRoot2Chem,
                                         depPathFromRoot2Dis=depRoot2Dis,
                                         depPathFromChem2Dis=depChem2Dis)
            insts.append(instance)

    return insts


def represent_seqs_of_indices_by_context_window(gensim_model, seqs_of_indices, word_context_window_size):
    index_of_padding_word = gensim_model.vocab.get('<PAD>').index  # pad the position before 'start' and at 'end'
    wd_dist = word_context_window_size // 2

    seqs_after_process = []
    for seq in seqs_of_indices:
        a_context_representation = []
        for i in range(len(seq)):
            l_dist = wd_dist
            context = []
            while l_dist >= 0:
                if i - l_dist < 0:
                    context.append(index_of_padding_word)
                    l_dist -= 1
                else:
                    context.append(seq[i - l_dist])
                    l_dist -= 1

            r_dist = 1
            while r_dist <= wd_dist:
                if i + r_dist >= len(seq):
                    context.append(index_of_padding_word)
                    r_dist += 1
                else:
                    context.append(seq[i + r_dist])
                    r_dist += 1

            a_context_representation.append(context)

        seqs_after_process.append(a_context_representation)

    return seqs_after_process


if __name__ == '__main__':
    embedding_model = gensim.models.Word2Vec.load_word2vec_format('../data/word_embeddings.5_20.gensim')
    # embedding_model = construct_customized_word_embedding_model(
    #     '../data/wordvectors/glove.840B.300d.bin', '../data/embeddings.5_300.cbow', 'part-of-speech.vocab',
    #     'syntactic.vocab', 'dependency.vocab')
    instances = load_candidate_instances('trainingSet.instances')
    instances.extend(load_candidate_instances('developmentSet.instances'))
    instances.extend(load_candidate_instances('testSet.instances'))
    print(instances[12].originalTokenizedSentenceText)
    print(len(instances[12].originalTokenInfos))
    print(instances[12].chemicalStartTokenIndex, instances[12].chemicalLastTokenIndex,
          instances[12].diseaseStartTokenIndex, instances[12].diseaseLastTokenIndex)
    instances[12].pad_sentence(150, padding='pre')
    words = [t[-1] for t in instances[12].get_token_infos(mode='IN_PADDED')]
    print(u'\u2191')
    print(instances[12].get_chemical_last_token_index())
    print(instances[12].get_chemical_last_token_index(mode='IN_ORIGINAL'))
    print(instances[12].chem_start_token_idx_after_padding,
          instances[12].chem_last_token_idx_after_padding,
          instances[12].dis_start_token_idx_after_padding,
          instances[12].dis_last_token_idx_after_padding)
    print(len(instances[12].get_token_infos(mode='IN_PADDED')))
    print(len(instances[12].get_token_infos(mode='IN_ORIGINAL')))
    print(get_token_distance_2_entity(words, instances[12].get_chemical_start_token_index(mode='IN_PADDED'),
                                      instances[12].get_chemical_last_token_index(mode='IN_PADDED'), 150))

    print(instances[12].get_verbs_in_between())
    print(instances[12].chemicalText,instances[12].diseaseText)


    pass

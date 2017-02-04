# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals
import codecs, gensim, os, copy


def count_file_number(dir_path, file_filter=None):
    """
    count the number of files in a 'dir_path' recursively, the default file suffix is given by 'file_filter'.

    :param dir_path:
    :param file_filter:
    :return:
    """
    file_nb = 0
    g = os.walk(dir_path)
    for t in g:
        files = t[-1]
        for x in files:
            if file_filter is not None:
                if x.endswith(file_filter):
                    # print(x)
                    file_nb += 1
            else:
                file_nb += 1

    if file_filter is not None:
        print('The total number of "%s" file is: %d' % (file_filter, file_nb))
    else:
        print('The total number of distinct files is: %d' % (file_nb,))
    pass


if __name__ == '__main__':
    count_file_number('/home/gjh/PubMedFiles', file_filter='.txt')

    pass

# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import, unicode_literals

import os
import platform


def bc5_cid_evaluation(evalName, evalType, gold_annotation_file, our_result_file):
    if not (os.path.exists(gold_annotation_file) and os.path.isfile(gold_annotation_file)):
        raise Exception("The gold annotation file does not exist!")

    if not (os.path.exists(our_result_file) and os.path.isfile(our_result_file)):
        raise Exception("The input result file does not exist!")
    print("\n**** Customized Method is Ending ****")
    print("\n**** BC5 Evaluation Starting... ****")
    # String evalArgs[] = new String[] { "relation", "CID", "PubTator", goldFilePath, resultFilePath };

    if 'Windows' in platform.system():
        cmd_prefix = 'java -Xms2g -Xmx10g -cp "./*;../*:%JAVA_HOME%/lib;%JAVA_HOME%/lib/dt.jar;%JAVA_HOME%/lib/tools.jar" ncbi.bc5cdr_eval.Evaluate'
    elif 'Linux' in platform.system():
        cmd_prefix = 'java -Xms2g -Xmx10g -cp "./*:../*:%JAVA_HOME%/lib:%JAVA_HOME%/lib/dt.jar:%JAVA_HOME%/lib/tools.jar" ncbi.bc5cdr_eval.Evaluate'
    else:
        raise Exception('Unknown System Platform! Please recheck it!')

    cmd_param = evalName + ' ' + evalType + ' ' + "PubTator" + ' ' + gold_annotation_file + ' ' + our_result_file
    cmd = cmd_prefix + ' ' + cmd_param
    # print(cmd)

    print('CMD Exit State: ', os.system(cmd))


if __name__ == '__main__':
    bc5_cid_evaluation("relation", "CID", '../CDR_TestSet.PubTator.txt', '../cid_results.txt')

    pass

import sys
sys.path.append('./src/common')
sys.path.append('./src/common/image_processor')
sys.path.append('./src/common/model_preparator')
sys.path.append('./src/net')
sys.path.append('./experiment_settings')
from mini_batch_loader import DatasetPreProcessor
from sample import get_args
from model_loader import prepare_model

import chainer
from chainer import serializers, Variable

import sys, os, math, inspect
import numpy as np
from collections import defaultdict
from gensim import corpora


def test(model, args):
    sum_accuracy = 0
    sum_loss     = 0
    dictionary = corpora.Dictionary.load(args.token_args.dic_load_path)
    id2token = {_id: token for token, _id in dictionary.token2id.items()}
    print(id2token)

    # start_idx = 4
    val_it, test_data_size = prepare_dataset(args)
    # test_data_size = 1
    print("------test data size")
    print(test_data_size)
    for batch in val_it:
        raw_x, raw_t, raw_tokens = np.array([batch[0][0]], dtype=np.float32), \
                            np.array([batch[0][1]],dtype=np.int32), \
                            np.array([batch[0][2]],dtype=np.int32)
        gt_label = int(batch[0][1])
        if args.debug_mode:
            cv.imshow('input image', raw_x[0].transpose(1, 2, 0))
            cv.waitKey(0)
            cv.destroyAllWindows()

        if args.gpu>-1:
            chainer.cuda.get_device(args.gpu).use()
            x = chainer.Variable(chainer.cuda.to_gpu(raw_x), volatile=True)
            t = chainer.Variable(chainer.cuda.to_gpu(raw_t), volatile=True)
            tokens = chainer.Variable(chainer.cuda.to_gpu(raw_tokens), volatile=True)
        else:
            x = chainer.Variable(raw_x, volatile=True)
            t = chainer.Variable(raw_t, volatile=True)
            tokens = chainer.Variable(raw_tokens, volatile=True)

        model(x, t, tokens)
        # result_sentence_ids = chainer.cuda.to_cpu(model.sentence_ids)
        sum_loss += model.loss.data
        sum_accuracy += model.accuracy.data

        # caption
        print("gt---------:")
        print(id2sentence(id2token, raw_tokens[0]))
        print("infer------:")
        print(id2sentence(id2token, model.sentence_ids))  # result_sentence_ids


def id2sentence(id2token, ids):
    result = []
    for _id in ids:
        try:
            word = id2token[int(_id)]
        except Exception as e:
            word = '<unk>'
        result.append(word)
        if word=='<EOS>':
            break
    return ' '.join(result)


def prepare_dataset(args):
    # load dataset
    test_mini_batch_loader = DatasetPreProcessor(args)
    val_it = chainer.iterators.SerialIterator( \
                            test_mini_batch_loader, \
                            1, repeat=False, shuffle=False)
    return val_it, test_mini_batch_loader.__len__()


def main(args):
    _, model_eval = prepare_model(get_args('train'))
    test(model_eval, args)


if __name__ == '__main__':
    args = get_args('test')
    main(args)

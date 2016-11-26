import sys, os
sys.path.append('./src/common')
sys.path.append('./src/common/image_processor')
sys.path.append('./src/common/model_preparator')
sys.path.append('./src/net')
sys.path.append('./src/net/RAM')
sys.path.append('./src/net/caption_generator')
sys.path.append('./experiment_settings')
from mini_batch_loader import DatasetPreProcessor
from important_serial_iterator import ImportantSerialIterator
from comment_gen_settings import get_args
from model_loader import prepare_model

import chainer
import chainer.functions as F
from chainer import serializers
from chainer import cuda, optimizers, Variable
from chainer import training
from chainer.training import extensions
from chainer import Reporter, report, report_scope

import cv2
import importlib
import numpy as np


def select_optimizer(args):
    if args.training_params.optimizer=='RMSpropGraves':
        return chainer.optimizers.RMSpropGraves(args.training_params.lr)
    elif args.training_params.optimizer=='Adam':
        return chainer.optimizers.Adam()
    elif args.training_params.optimizer=='NesterovAG':
        return chainer.optimizersNesterovAG(args.training_params.lr)


def prepare_optimizer(model, args):
    optimizer = select_optimizer(args)
    optimizer.setup(model)
    if args.training_params.weight_decay:
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.training_params.weight_decay))
    if args.training_params.lasso:
        optimizer.add_hook(chainer.optimizer.Lasso(args.training_params.weight_decay))
    if args.training_params.clip_grad:
        optimizer.add_hook(chainer.optimizer.GradientClipping(args.training_params.clip_value))
    return optimizer


def prepare_dataset():
    train_args = get_args('train')
    # load dataset
    train_mini_batch_loader = DatasetPreProcessor(train_args)
    test_mini_batch_loader = DatasetPreProcessor(get_args('test'))
    print("---set mini_batch----------")

    if train_args.importance_sampling:
        print("importance----------")
        train_it = ImportantSerialIterator( \
                                train_mini_batch_loader, \
                                train_args.training_params.batch_size, \
                                shuffle=train_args.shuffle, \
                                p=np.loadtxt(train_args.weights_file_path))
    else:
        if train_args.training_params.iter_type=='multi':
            iterator = chainer.iterators.MultiprocessIterator
        else:
            iterator = chainer.iterators.SerialIterator
        train_it = iterator( \
                        train_mini_batch_loader, \
                        train_args.training_params.batch_size, \
                        shuffle=train_args.shuffle)

    val_it = iterator( \
                test_mini_batch_loader, \
                1, repeat=False, shuffle=False)
    return train_it, val_it, train_mini_batch_loader.__len__()


def main(args):
    # load model
    model, model_for_eval = prepare_model(args)
    print("---set model----------")

    # Setup optimizer
    optimizer = prepare_optimizer(model, args)
    print("---set optimzer----------")

    # load data
    train_it, val_it, train_data_length = prepare_dataset()
    print("---set data----------")

    updater = training.StandardUpdater(train_it, optimizer, device=args.gpu)
    print("---set updater----------")

    evaluator_interval = args.training_params.report_epoch, 'epoch'
    snapshot_interval = args.training_params.snapshot_epoch, 'epoch'
    log_interval = args.training_params.report_epoch, 'epoch'

    trainer = training.Trainer( \
        updater, (args.training_params.epoch, 'epoch'), out=args.output_path)
    trainer.extend( \
        extensions.Evaluator(val_it, model_for_eval, device=args.gpu), \
        trigger=evaluator_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object( \
        model, 'model_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.ExponentialShift( \
        'lr', args.training_params.decay_factor), \
        trigger=(args.training_params.decay_epoch, 'epoch'))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([ \
        'epoch', 'iteration', 'main/loss', 'validation/main/loss', \
        'main/accuracy', 'validation/main/accuracy', \
        ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=1))
    print("---set trainer----------")

    if os.path.exists(args.resume):
        print('resume trainer:{}'.format(args.resume))
        # Resume from a snapshot
        serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    print("-------traing")
    args = get_args('train')
    main(args)

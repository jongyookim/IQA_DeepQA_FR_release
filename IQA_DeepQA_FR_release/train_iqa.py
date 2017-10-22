from __future__ import absolute_import, division, print_function

import os
import timeit
from importlib import import_module

import numpy as np
import theano
import theano.tensor as T

from .config_parser import config_parser, dump_config
from .data_load.data_loader_IQA import DataLoader
from .trainer import Trainer


def train_iqa(config_file, section, snap_path,
              output_path=None, snap_file=None, tr_te_file=None):
    """
    Imagewise training of an IQA model using both reference and
    distorted images.
    """
    db_config, model_config, train_config = config_parser(
        config_file, section)

    # Check snapshot file
    if snap_file is not None:
        assert os.path.isfile(snap_file), \
            'Not existing snap_file: %s' % snap_file

    # Initialize patch step
    init_patch_step(db_config, int(model_config.get('ign', 0)),
                    int(model_config.get('ign_scale', 1)))

    # Load data
    data_loader = DataLoader(db_config)
    train_data, test_data = data_loader.load_data_tr_te(tr_te_file)
    # train_data, test_data = data_loader.load_toy_data_tr_te()

    # Create model
    model = create_model(model_config,
                         train_data.patch_size, train_data.num_ch)
    if snap_file is not None:
        model.load(snap_file)

    # Create trainer
    trainer = Trainer(train_config, snap_path, output_path)

    # Store current configuration file
    dump_config(os.path.join(snap_path, 'config.yaml'),
                db_config, model_config, train_config)

    ###########################################################################
    # Train the model
    epochs = train_config.get('epochs', 100)
    batch_size = train_config.get('batch_size', 4)

    score = run_iqa_iw(
        train_data, test_data, model, trainer, epochs, batch_size)
    print("Best SRCC: {:.3f}, PLCC: {:.3f} ({:d})".format(
        score[0], score[1], score[2]))


def run_iqa_iw(train_data, test_data, model, trainer, epochs, n_batch_imgs,
               x_c=None, x=None, mos_set=None, bat2img_idx_set=None,
               prefix2='iqa_'):
    """
    @type model: .models.model_basis.ModelBasis
    @type train_data: .data_load.dataset.Dataset
    @type test_data: .data_load.dataset.Dataset
    """
    te_n_batch_imgs = 1

    # Make dummy shared dataset
    max_num_patch = np.max(np.asarray(train_data.npat_img_list)[:, 0])
    n_pats_dummy = max_num_patch * n_batch_imgs
    sh = model.input_shape
    np_set_r = np.zeros((n_pats_dummy, sh[2], sh[3], sh[1]), dtype='float32')
    np_set_d = np.zeros((n_pats_dummy, sh[2], sh[3], sh[1]), dtype='float32')
    shared_set_r = theano.shared(np_set_r, borrow=True)
    shared_set_d = theano.shared(np_set_d, borrow=True)

    train_data.set_imagewise()
    test_data.set_imagewise()

    print('\nCompile theano function: Regress on MOS', end='')
    print(' (imagewise / low GPU memory)')
    start_time = timeit.default_timer()
    if x is None:
        x = T.ftensor4('x')
    if x_c is None:
        x_c = T.ftensor4('x_c')
    if mos_set is None:
        mos_set = T.vector('mos_set')
    if bat2img_idx_set is None:
        bat2img_idx_set = T.imatrix('bat2img_idx_set')

    print(' (Make training model)')
    model.set_training_mode(True)
    cost, updates, rec_train = model.cost_updates_iqa(
        x, x_c, mos_set, n_batch_imgs, bat2img_idx_set)
    outputs = [cost] + rec_train.get_function_outputs(train=True)

    train_model = theano.function(
        [mos_set, bat2img_idx_set],
        [output for output in outputs],
        updates=updates,
        givens={
            x: shared_set_r,
            x_c: shared_set_d
        },
        on_unused_input='warn'
    )

    print(' (Make testing model)')
    model.set_training_mode(False)
    cost, rec_test = model.cost_iqa(
        x, x_c, mos_set, te_n_batch_imgs, bat2img_idx_set=bat2img_idx_set)
    outputs = [cost] + rec_test.get_function_outputs(train=False)

    test_model = theano.function(
        [mos_set, bat2img_idx_set],
        [output for output in outputs],
        givens={
            x: shared_set_r,
            x_c: shared_set_d
        },
        on_unused_input='warn'
    )

    minutes, seconds = divmod(timeit.default_timer() - start_time, 60)
    print(' - Compilation took {:02.0f}:{:05.2f}'.format(minutes, seconds))

    def get_train_outputs():
        res = train_data.next_batch(n_batch_imgs)
        np_set_r[:res['n_data']] = res['ref_data']
        np_set_d[:res['n_data']] = res['dis_data']
        shared_set_r.set_value(np_set_r)
        shared_set_d.set_value(np_set_d)
        return train_model(res['score_set'], res['bat2img_idx_set'])

    def get_test_outputs():
        res = test_data.next_batch(te_n_batch_imgs)
        np_set_r[:res['n_data']] = res['ref_data']
        np_set_d[:res['n_data']] = res['dis_data']
        shared_set_r.set_value(np_set_r)
        shared_set_d.set_value(np_set_d)
        return test_model(res['score_set'], res['bat2img_idx_set'])

    # Main training routine
    return trainer.training_routine(
        model, get_train_outputs, rec_train, get_test_outputs, rec_test,
        n_batch_imgs, te_n_batch_imgs, train_data, test_data,
        epochs, prefix2, check_mos_corr=True)


def init_patch_step(db_config, ign_border, ign_scale=8):
    """
    Initialize patch_step:
    patch_step = patch_size - ign_border * ign_scale.
    """
    patch_size = db_config.get('patch_size', None)
    patch_step = db_config.get('patch_step', None)
    random_crops = int(db_config.get('random_crops', 0))

    if (patch_size is not None and patch_step is None and
            random_crops == 0):
        db_config['patch_step'] = (
            patch_size[0] - ign_border * ign_scale,
            patch_size[1] - ign_border * ign_scale)
        print(' - Set patch_step according to patch_size and ign: (%d, %d)' % (
            db_config['patch_step'][0], db_config['patch_step'][1]
        ))


def create_model(model_config, patch_size=None, num_ch=None):
    """
    Create a model using a model_config.
    Set input_size and num_ch according to patch_size and num_ch.
    """
    model_module_name = model_config.get('model', None)
    assert model_module_name is not None
    model_module = import_module(model_module_name)

    # set input_size and num_ch according to dataset information
    if patch_size is not None:
        model_config['input_size'] = patch_size
    if num_ch is not None:
        model_config['num_ch'] = num_ch

    return model_module.Model(model_config)

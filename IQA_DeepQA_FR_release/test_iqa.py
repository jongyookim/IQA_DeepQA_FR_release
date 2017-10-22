from __future__ import absolute_import, division, print_function

import os
import timeit
from importlib import import_module

import numpy as np
import theano
import theano.tensor as T

from .config_parser import config_parser
from .data_load.data_loader_IQA import DataLoader
from .trainer import Trainer


def check_dist_list(testing_dist_list, db_config):
    if not isinstance(testing_dist_list, (list, tuple)):
        testing_dist_list = (testing_dist_list, )

    dist_list = []
    for testing_dist in testing_dist_list:
        if testing_dist == 'each':
            if db_config['sel_data'] == 'LIVE':
                from .data_load import LIVE
                dist_list += [[dist] for dist in LIVE.ALL_DIST_TYPES]
            elif db_config['sel_data'] == 'TID2008':
                from .data_load import TID2008
                dist_list += [[dist] for dist in TID2008.ALL_DIST_TYPES]
            elif db_config['sel_data'] == 'TID2013':
                from .data_load import TID2013
                dist_list += [[dist] for dist in TID2013.ALL_DIST_TYPES]
            elif db_config['sel_data'] == 'CSIQ':
                from .data_load import CSIQ
                dist_list += [[dist] for dist in CSIQ.ALL_DIST_TYPES]
            else:
                raise NotImplementedError
        else:
            dist_list.append(testing_dist)
    return dist_list


def test_iqa(config_file, section, testing_dist_list=('each', 'all'),
             output_path=None, snap_file=None, load_keys=None,
             tr_te_file=None, use_ref_for_nr=True):
    db_config, model_config, train_config = config_parser(
        config_file, section)

    # Check snapshot file
    if snap_file is not None:
        assert os.path.isfile(snap_file), \
            'Not existing snap_file: %s' % snap_file

    testing_dist_list = check_dist_list(testing_dist_list, db_config)

    # Initialize patch step
    init_patch_step(db_config, int(model_config.get('ign', 0)),
                    int(model_config.get('ign_scale', 1)))

    x_c = T.ftensor4('x_c')
    x = T.ftensor4('x')
    mos_set = T.vector('mos_set')
    bat2img_idx_set = T.imatrix('bat2img_idx_set')

    batch_size = train_config.get('batch_size', 1)

    # Write log
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    with open(os.path.join(output_path, 'results.txt'), 'a') as f_log:
        data = 'Dist. type, SRCC, PLCC\n'
        f_log.write(data)

    # Test for each testing distortion set in testing_dist_list
    made_model = False
    for idx, testing_dist in enumerate(testing_dist_list):
        print('\n##### %d/%d #####' % (idx + 1, len(testing_dist_list)))
        prefix2 = 'dist_%d' % idx

        # Load data
        db_config['dist_types'] = testing_dist
        data_loader = DataLoader(db_config)
        _, test_data = data_loader.load_data_tr_te(tr_te_file)

        if not made_model:
            # Create model
            model = create_model(model_config,
                                 test_data.patch_size, test_data.num_ch)

            if load_keys is None:
                model.load(snap_file)
            else:
                model.load_load_keys(load_keys, snap_file)
                # model.load_load_keys(['sens_map', 'reg_mos'], snap_file)
            made_model = True

        # Create trainer
        trainer = Trainer(train_config, output_path=output_path)

        score = run_iqa_iw(
            test_data, model, trainer, batch_size,
            x=x, x_c=x_c, mos_set=mos_set, bat2img_idx_set=bat2img_idx_set,
            prefix2=prefix2)

        # Write log
        with open(os.path.join(output_path, 'results.txt'), 'a') as f_log:
            data = '{:s}, {:.4f}, {:.4f}\n'.format(
                str(testing_dist), score[0], score[1])
            f_log.write(data)


def run_iqa_iw(test_data, model, trainer,
               n_batch_imgs, x=None, x_c=None, mos_set=None,
               bat2img_idx_set=None, prefix2=''):
    """
    @type model: .models.model_basis.ModelBasis
    @type test_data: .data_load.dataset.Dataset
    """
    # Make dummy shared dataset
    max_num_patch = np.max(np.asarray(test_data.npat_img_list)[:, 0])
    n_pats_dummy = max_num_patch * n_batch_imgs
    sh = model.input_shape
    np_set_r = np.zeros((n_pats_dummy, sh[2], sh[3], sh[1]), dtype='float32')
    np_set_d = np.zeros((n_pats_dummy, sh[2], sh[3], sh[1]), dtype='float32')
    shared_set_r = theano.shared(np_set_r, borrow=True)
    shared_set_d = theano.shared(np_set_d, borrow=True)

    test_data.set_imagewise()

    print('\nCompile theano function: IQA using reference images', end=' ')
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

    print(' (Make testing model)')
    model.set_training_mode(False)
    cost, rec_test = model.cost_iqa(
        x, x_c, mos_set, n_img=n_batch_imgs, bat2img_idx_set=bat2img_idx_set)
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

    def get_test_outputs():
        res = test_data.next_batch(n_batch_imgs)
        np_set_r[:res['n_data']] = res['ref_data']
        np_set_d[:res['n_data']] = res['dis_data']
        shared_set_r.set_value(np_set_r)
        shared_set_d.set_value(np_set_d)
        return test_model(res['score_set'], res['bat2img_idx_set'])

    # Main testing routine
    return trainer.testing_routine(
        get_test_outputs, rec_test, n_batch_imgs, test_data,
        prefix2, check_mos_corr=True)


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

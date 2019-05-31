from absl import app
from absl import flags

import copy
import dnnlib
from dnnlib import EasyDict
import config
from metrics import metric_base

FLAGS = flags.FLAGS
flags.DEFINE_integer('size', 128, 'The size of the training images.')
flags.DEFINE_string('dataset_dir', 'eboy-dataset',
                    'The directory of the training dataset.')


def main(_):
    # Adapted from: https://github.com/NVlabs/stylegan/blob/master/train.py
    desc = 'sgan'
    train = EasyDict(run_func_name='training.training_loop.training_loop')
    G = EasyDict(func_name='training.networks_stylegan.G_style')
    D = EasyDict(func_name='training.networks_stylegan.D_basic')
    G_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)
    D_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)
    G_loss = EasyDict(func_name='training.loss.G_logistic_nonsaturating')
    D_loss = EasyDict(func_name='training.loss.D_logistic_simplegp',
                      r1_gamma=10.0)
    dataset = EasyDict()
    sched = EasyDict()
    grid = EasyDict(size='4k', layout='random')
    metrics = [metric_base.fid50k]
    submit_config = dnnlib.SubmitConfig()
    tf_config = {'rnd.np_random_seed': 1000}

    desc += '-eboy'
    dataset = EasyDict(tfrecord_dir=FLAGS.dataset_dir, resolution=FLAGS.size)
    train.mirror_augment = True

    desc += '-1gpu'
    submit_config.num_gpus = 1
    sched.minibatch_base = 4
    sched.minibatch_dict = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16,
                            256: 8, 512: 4}

    train.total_kimg = 25000
    sched.lod_initial_resolution = 8
    sched.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
    sched.D_lrate_dict = EasyDict(sched.G_lrate_dict)

    kwargs = EasyDict(train)
    kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt,
                  G_loss_args=G_loss, D_loss_args=D_loss)
    kwargs.update(dataset_args=dataset, sched_args=sched, grid_args=grid,
                  metric_arg_list=metrics, tf_config=tf_config)
    kwargs.submit_config = copy.deepcopy(submit_config)
    kwargs.submit_config.run_dir_root = \
        dnnlib.submission.submit.get_template_from_path(config.result_dir)
    kwargs.submit_config.run_dir_ignore += config.run_dir_ignore
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)


if __name__ == '__main__':
    app.run(main)

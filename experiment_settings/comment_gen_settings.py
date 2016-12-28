import os, sys
sys.path.append('./experiment_settings')
from easydict import EasyDict as edict
import platform
from base_settings import get_base_params, calculate_in_ch


# common params
base_params = get_base_params()


# hand making params
debug_mode = False
converse_gray = False
detect_edge = False
in_ch = calculate_in_ch(converse_gray, detect_edge)
generate_comment = True
crop = True
resize = False
gpu = -1 if platform.system()==base_params.local_os_name else 1
use_net = 'attention_caption_net_mod'
n_class = 19  # number of class is 2, if you use ne_class classifier.
crop_size = 352
normalize_type = 'LCN'  # 'ZCA', 'LCN'
zca_eps = 1e-5  # Restoration coefficient of values in image
im_norm_type = base_params.image_normalize_types_dir[normalize_type]
base_params.image_normalize_types_dir['ZCA']['opts'] = zca_eps
model_module = base_params.net_dir[use_net]
experiment_criteria = ''
output_path = os.path.join(base_params.data_root_path+'/results', use_net+experiment_criteria)
initial_model = os.path.join( \
    base_params.data_root_path+'/results'+'/'+use_net+experiment_criteria, \
    'model_iter_xxx')
resume = os.path.join( \
    base_params.data_root_path+'/results'+'/'+use_net+experiment_criteria, \
    'snapshot_iter_xxx')
aug_flags = {'do_scale':False, 'do_flip':True,
             'change_britghtness':False, 'change_contrast':False,
             'do_shift':False, 'do_rotate':False}


# reset training params
base_params.trainig_params.lr = 1e-3
base_params.trainig_params.snapshot_epoch = 30
base_params.trainig_params.clip_grad = True
base_params.trainig_params.clip_value = 5.
# base_params.trainig_params.iter_type = 'multi'


# a body of args
train_args = \
    {
        'train': True,
        'active_learn': False,
        'generate_comment': generate_comment,
        'debug_mode': debug_mode,
        'gpu': gpu,
        'n_class': n_class,
        'in_ch': in_ch,
        'image_dir_path': base_params.data_root_path+'/train_data_path',
        'image_pointer_path': base_params.data_root_path+'/train_image_pointer',
        'labels_file_path': base_params.data_root_path+'/train_labels',
        'weights_file_path': base_params.data_root_path+'/train_class_weights',
        'output_path': output_path,
        'initial_model': initial_model,
        'resume': resume,
        'im_norm_type': im_norm_type,
        'archtecture': model_module,
        'converse_gray': converse_gray,
        'do_resize': resize,
        'crop_params': {'flag':crop, 'size': crop_size},
        'multiple': base_params.mult_dir[use_net],  # total stride multiple
        'aug_params': {'do_augment':True,
                       'params': dict(base_params.augmentation_params, **aug_flags),
                      },
        'importance_sampling': False,
        'shuffle': True,  # data shuffle in SerialIterator
        'training_params': base_params.trainig_params,
        'token_args': base_params.token_args,
    }

test_args = \
    {
        'train': False,
        'generate_comment': generate_comment,
        'debug_mode': debug_mode,
        'gpu': gpu,
        'n_class': n_class,
        'in_ch': in_ch,
        'image_dir_path': base_params.data_root_path+'/test_data_path',
        'image_pointer_path': base_params.data_root_path+'/test_image_pointer',
        'labels_file_path': base_params.data_root_path+'/test_labels',
        'output_path': output_path,
        'initial_model': initial_model,
        'im_norm_type': im_norm_type,
        'archtecture': model_module,
        'converse_gray': converse_gray,
        'do_resize': resize,
        'crop_params': {'flag':crop, 'size': crop_size},
        'multiple': base_params.mult_dir[use_net],  # total stride multiple
        'aug_params': {'do_augment':False},
        'token_args': base_params.token_args,
    }


def get_args(args_type='train'):
    if args_type=='train':
        return edict(train_args)
    else:
        return edict(test_args)

import os
from math import sqrt
from easydict import EasyDict as edict
import platform
import inflection


def calculate_in_ch(converse_gray, detect_edge=False):
    in_ch = 1 if converse_gray else 3
    in_ch = in_ch+1 if detect_edge else in_ch
    return in_ch


# immortal params
local_os_name = 'Darwin'
data_root_path = './data' if platform.system()==local_os_name else '/data'

# mult_dir's key is module name
mult_dir = {'n_i_n': 32,
            'r_a_m': 8,
            'r_a_m_cnn': 8,
            'res_net50': 32,
            'dense_net': 4,  # equal 2^(block-1) which is number of aberage pooling(stride=2)
            'squeeze_net': 16,
            'squeeze_net_dilate': 8,
            'squeeze_net_dilate_recog_one_class': 8,
            'squeeze_net_dilate_sr0075': 8,
            'simple_bypass_squeeze': 8,
            'review_net': 8,
            'caption_net': 8,
            'attention_caption_net': 8,
            'attention_caption_net_mod': 8}

augmentation_params = {
                       'scale':[0.5, 0.75, 1.25],
                       'ratio':[sqrt(1/2), 1, sqrt(2)],  # 1/sqrt(2)だと2ratioが2倍, 逆だと0.5倍
                       'lr_shift':[-64, -32, -16, 16, 32, 64],
                       'ud_shift':[-64, -32, -16, 16, 32, 64],
                       'rotation_angle': list(range(5,360,5))
                      }

net_dir = {net_name:{'module_name':net_name, \
                    'class_name':inflection.camelize(net_name)} \
                        for net_name, _ in mult_dir.items()}

image_normalize_types_dir = {'ZCA': {'method':'zca_whitening', 'opts':{'eps':1e-5}},
                             'LCN': {'method':'local_contrast_normalization', 'opts':None},
                             'GCN': {'method':'global_contrast_normalization', 'opts':None}
                            }

dic_name = 'sysm_pathological.dict'
token_args = {
        'lang': 'ja',
        'max_len': 50,
        'tagger': '-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd -Owakati',
        'dic_load_path': os.path.join(data_root_path, dic_name),
        'texts_path': os.path.join(data_root_path, 'pathological_comments'),
        'dic_save_path': os.path.join(data_root_path, dic_name),
        'tokens_path': os.path.join(data_root_path, 'sysm_tokens.npz')
    }

trainig_params = {
        'optimizer': 'RMSpropGraves',
        'lr': 1e-5,
        'batch_size': 20,
        'epoch': 200,
        'decay_factor': 0.05,  # as lr time decay
        'decay_epoch': 50,
        'snapshot_epoch': 5,
        'report_epoch': 1,
        'weight_decay': True,
        'lasso': False,
        'clip_grad': False,
        'weight_decay': 0.0005,
        'clip_value': 5.,
        'iter_type': 'serial',
    }


def get_base_params():
    base_params = {
                    'local_os_name': local_os_name,
                    'data_root_path': data_root_path,
                    'mult_dir': mult_dir,
                    'augmentation_params': augmentation_params,
                    'net_dir': net_dir,
                    'image_normalize_types_dir': image_normalize_types_dir,
                    'trainig_params': trainig_params,
                    'token_args': token_args,
                }
    return edict(base_params)

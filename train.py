import train_involution
import train_mutual_learning
import utils
import time
import wandb

def train_(model_str, proj_name, is_augmentation=False):
    SAVE_PATH = utils.get_save_path()
    print('SAVE_PATH: ', SAVE_PATH)
    train_involution.involution_(model_str, 'involution_' + proj_name, SAVE_PATH, is_augmentation=is_augmentation)
    train_mutual_learning.mutual_(model_str, 'mutual_' + proj_name, SAVE_PATH, is_augmentation=is_augmentation)


if __name__ == '__main__':

    model_str = ['resnet32, resnet32, resnet32, resnet32',
                 'resnet32, resnet32, resnet32, mobilenetV1',
                 'resnet32, resnet32, mobilenetV1, mobilenetV1',
                 'resnet32, resnet32',
                 'resnet32, resnet32, resnet32',
                 'resnet32, mobilenetV1',
                 'resnet32, resnet32, mobilenetV1',

                ]
    proj_name = ['r32_r32_r32_r32_12no',
                 'r32_r32_r32_mv1_11',
                 'r32_r32_mv1_mv1_12no',
                 'r32_r32_11',
                 'r32_r32_r32_',
                 'r32_mv1_',
                 'r32_r32_mv1_',
                 ]

    for i in range(0, len(proj_name)):
        print('*' * 100)
        train_(model_str[i], proj_name[i])
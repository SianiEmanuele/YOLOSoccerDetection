# flake8: noqa
import os.path as osp
import sys

from basicsr.train import train_pipeline

import realesrgan.archs
import realesrgan.data
import realesrgan.models

if __name__ == '__main__':
    # Set the root path relative to this script
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))

    # Simulate CLI arguments (you can change this YAML path if needed)
    sys.argv = [
        'train.py',
        '-opt', r'src\models\esrgan\options\finetune_realesr-general-x4v3.yml',
        '--auto_resume'
    ]

    # Start training
    train_pipeline(root_path)


    

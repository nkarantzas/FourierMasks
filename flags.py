import argparse

parser = argparse.ArgumentParser(
    description="Understanding Robustness and Generalization through Fourier Masks"
)



# Imagenette Data set hyperparameters
parser.add_argument(
    '--training_data', 
    type=str,
    required=False,
    default='/mnt/scratch07/datasets/imagenette/train'
)
parser.add_argument(
    '--test_data', 
    type=str,
    required=False,
    default='/mnt/scratch07/datasets/imagenette/val'
)
parser.add_argument(
    '--data_stats',
    type=tuple,
    required=False,
    default=((0.449,), (0.226,))
)
parser.add_argument(
    '--img_size',
    type=tuple,
    required=False,
    default=(128, 128)
)



# Model hyperparameters
parser.add_argument(
    '--architecture', 
    type=str,
    required=False,
    help="either 'vgg' or 'resnet'"
)



# Model Training hyperparameters
parser.add_argument(
    '--model_save_folder', 
    type=str,
    required=False,
    help="path to where all the models should be saved"
)
parser.add_argument(
    '--model_epochs', 
    type=int,
    required=False,
    default=50
)
parser.add_argument(
    '--batch_size', 
    type=int,
    required=False,
    default=128
)
parser.add_argument(
    '--lr', 
    type=float,
    required=False,
    default=0.001
)
parser.add_argument(
    '--schedule', 
    type=float,
    required=False,
    default=0.1
)
parser.add_argument(
    '--weight_decay', 
    type=float,
    required=False,
    default=0.
)
parser.add_argument(
    '--seed', 
    type=int,
    required=False,
    default=31
)
parser.add_argument(
    '--adv_eps', 
    type=float,
    required=False,
    default=0.06
)



# Mask Training Hyperparameters
parser.add_argument(
    '--mask_save_folder', 
    type=str,
    required=False,
    help="path to where all the masks should be saved"
)
parser.add_argument(
    '--mask_epochs', 
    type=int,
    required=False,
    default=100
)
parser.add_argument(
    '--class_portion', 
    type=float,
    required=False,
    default=1.
)
parser.add_argument(
    '--mask_lr', 
    type=float,
    required=False,
    default=0.001
)
parser.add_argument(
    '--mask_schedule', 
    type=float,
    required=False,
    default=0.1
)
parser.add_argument(
    '--mask_decay', 
    type=float,
    required=False,
    default=0.2
)
parser.add_argument(
    '--mask_patience', 
    type=int,
    required=False,
    default=7
)



# Single Img Mask training hyperparameters
parser.add_argument(
    '--single_img_mask_save_folder', 
    type=str,
    required=False,
    help="Path to where single img data and masks should be saved"
)
parser.add_argument(
    '--single_img_mask_epochs', 
    type=int,
    required=False,
    default=20000
)
parser.add_argument(
    '--single_img_mask_lr', 
    type=float,
    required=False,
    default=0.2
)
parser.add_argument(
    '--single_img_mask_schedule', 
    type=float,
    required=False,
    default=0.1
)
parser.add_argument(
    '--single_img_mask_decay', 
    type=float,
    required=False,
    default=0.07
)
parser.add_argument(
    '--single_img_mask_patience', 
    type=int,
    required=False,
    default=300
)
parser.add_argument(
    '--num_images',  
    default=None, 
    required=False, 
    help='num images to train single masks for'
)

args, _ = parser.parse_known_args()
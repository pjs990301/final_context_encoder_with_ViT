import argparse
import ast

def parse_comma_separated_list(arg_value):
    return [int(item) for item in arg_value.split(',')]

def merge_args_with_train_config(train_config, args, input_args):
    """
    Updates args with train_config, giving priority to command-line arguments.
    
    Args:
        train_config (dict): Configuration loaded from train_args.json.
        args (Namespace): Parsed command-line arguments.
        input_args (list): Original command-line arguments for reference.
    
    Returns:
        Namespace: Updated arguments.
    """
    # Create a set of keys explicitly passed through command-line arguments
    explicit_keys = set(
        key.lstrip('-') for key in input_args if key.startswith('--')
    )

    # Merge train_config into args, respecting explicit command-line inputs
    for key, value in train_config.items():
        if key not in explicit_keys:  # Only update if the key wasn't passed in CLI
            setattr(args, key, value)
    
    return args

class Parser:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Argument parser for configuring the model and dataset settings.")
        
        # General
        parser.add_argument("--output_dir", type=str, default="context_vit", help="output directory")
        
        # CUDA
        parser.add_argument("--gpu_id", type=str, default="0", help="GPU id")
        
        # Dataset
        parser.add_argument("--dataset", type=str, default="dataset1", help="name of the dataset")
        parser.add_argument("--enviroment", type=int, default=1, help="enviroment")
        parser.add_argument("--csi_type", type=str, default="amplitude", help="type of csi data")
        parser.add_argument("--RPI", type=parse_comma_separated_list, default="1,2,3", help="Using RPI")
        parser.add_argument("--img_size", type=int, default=128, help="size of the input image")
        parser.add_argument("--window_size", type=int, default=30, help="size of the window")
        parser.add_argument("--batch_size", type=int, default=16, help="batch size")
        parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
        
        # Preprocessing
        # Sampling
        parser.add_argument("--is_sampling", type=bool, default=False, help="sampling")
        parser.add_argument("--sampling_rate", type=int, default=1, help="sampling rate")
        
        # Exclude
        parser.add_argument("--is_exclude", type=bool, default=False, help="exclude null subcarriers and pilot subcarriers")
        
        # Preprocess
        parser.add_argument("--is_preprocess", type=bool, default=False, help="preprocess")
        parser.add_argument("--preprocess_type", type=str, nargs='+', default=["PCA", "lowpass"], help="preprocess type")
        
        
        # PCA
        parser.add_argument("--is_PCA", type=bool, default=False, help="PCA")
        parser.add_argument("--PCA_components", type=int, default=10, help="PCA components")
        
        # Low pass
        parser.add_argument("--low_cut_off", type=int, default=10, help="low pass cut off")
        parser.add_argument("--low_order", type=int, default=4, help="low pass order")
        parser.add_argument("--low_fs", type=int, default=100, help="low pass fs")
        
        # Proposed scheduling Method
        parser.add_argument("--is_split", type=bool, default=False, help="split")
        parser.add_argument("--is_zero_padding", type=bool, default=False, help="zero padding")
        
        # Round Robin scheduling        
        parser.add_argument("--round_robin_size", type=int, default=5, help="round robin size")
        parser.add_argument("--is_round_robin_order", type=bool, default=False, help="round robin")
        parser.add_argument("--round_robin_order", type=parse_comma_separated_list, default="0,1,2", help="round robin order")
        
        # Trainer
        parser.add_argument("--early_stopping", type=bool, default=True, help="early stopping")
        parser.add_argument("--early_stopping_target", type=str, default="train_accuracy", help="early stopping target")
        parser.add_argument("--early_stopping_patience", type=int, default=10, help="early stopping patience")
        parser.add_argument("--early_stopping_mode", type=str, default="max", help="early stopping mode")
        parser.add_argument("--early_change_rate", type=float, default=0.1, help="early stopping change rate")
        parser.add_argument("--metric", type=str, nargs='+', default=["accuracy"], help="metric")
        parser.add_argument("--seed", type=int, default=42, help="seed")
        parser.add_argument("--epochs", type=int, default=200, help="epochs")
        
        # Classifier
        parser.add_argument("--cls_model", type=str, default="ViT", help="classifier_model")
        parser.add_argument("--cls_loss", type=str, default="crossentropy", help="classifier_loss")
        parser.add_argument("--cls_optimizer", type=str, default="adam", help="classifier_optimizer")
        parser.add_argument("--cls_lr", type=float, default=0.0001, help="classifier lr")
        parser.add_argument("--cls_step_size", type=int, default=10, help="classifier_step size")
        parser.add_argument("--cls_gamma", type=float, default=0.9, help="classifier_gamma")
        parser.add_argument("--vit_patch_size", type=int, default=16, help="vit_patch size")
        
        # Inpainting
        parser.add_argument("--is_inpainting", type=bool, default=False, help="inpainting")
        parser.add_argument("--inpainting_lr", type=float, default=0.0002, help="inpainting_lr")
        parser.add_argument("--inpainting_b1", type=float, default=0.5, help="inpainting_b1")
        parser.add_argument("--inpainting_b2", type=float, default=0.999, help="inpainting_b2")
        parser.add_argument("--inpainting_latent_dim", type=int, default=100, help="inpainting_latent_dim")
        parser.add_argument("--use_mask", type=bool, default=False, help="use_mask")
        parser.add_argument("--mask_h", type=int, default=64, help="mask_h")
        parser.add_argument("--mask_w", type=int, default=64, help="mask_w")
        
        # Log
        parser.add_argument("--debug", type=bool, default=False, help="debug")
        parser.add_argument("--wandb", type=bool, default=False, help="wandb")
        parser.add_argument("--log_interval", type=int, default=10, help="log_interval")
        parser.add_argument("--save_interval", type=int, default=10, help="save_interval")
        parser.add_argument("--plot", type=str, nargs='+', default=["loss", "accuracy"], help="plot")

        # To inference
        parser.add_argument("--train_serial", type=str, default=None, help="train_serial")
        parser.add_argument("--train_dir", type=str, default=None, help="train_dir")
        parser.add_argument("--test_mask", type=bool, default=False, help="test mask")
        parser.add_argument("--test_mask_type", type=str, default="center", help="test mask")
        parser.add_argument('--test_interpolation', type=str, default="linear", help='test interpolation')
        
        self.opt = parser.parse_args()
    
    def __getitem__(self, item):
        return getattr(self.opt, item)

    
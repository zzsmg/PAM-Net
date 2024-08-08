import os, argparse

import warnings
from datetime import datetime


now = datetime.now()
time_string = now.strftime("%Y%m%d")
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='PAM-DWT Dehaze')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=4, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=10000, type=int)
parser.add_argument('--model_dir', type=str, default='./logs/best_model/best.pkl')
parser.add_argument('--model_save_dir', type=str, default='./logs/logs_points/')
parser.add_argument('--best_model', type=str, default='./logs/best_model/')
parser.add_argument('--plot_path', type=str, default='./logs/numpy_files/')
parser.add_argument('--crop', action='store_true')
parser.add_argument('--crop_size', type=int, default=384, help='Takes effect when using --crop ')
# --- Parse hyper-parameters test --- #
parser.add_argument('--predict_result', type=str, default='./logs/test_result/')
args = parser.parse_args()

if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir, exist_ok=True)
if not os.path.exists(args.best_model):
    os.makedirs(args.best_model, exist_ok=True)
if not os.path.exists(args.plot_path):
    os.makedirs(args.plot_path, exist_ok=True)
if not os.path.exists(args.predict_result):
    os.makedirs(args.predict_result, exist_ok=True)


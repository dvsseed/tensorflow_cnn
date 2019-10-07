from __future__ import print_function

import argparse

# Parse input arguments

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=False, help="path to input dataset of house images")
ap.add_argument('--num_epochs', default=40)
ap.add_argument('--batch_size', default=384)
ap.add_argument('--num_classes', default=4)
ap.add_argument('--plot', default=False, help='disables plotting')
ap.add_argument('--learning_rate', default=1e-3)
# ap.add_argument('--learning_rate', default=1e-4)
ap.add_argument("--debug", action="store_true", default=False)
ap.add_argument("--show_tf_cpp_log", action="store_true", default=False)

args = ap.parse_args()
epochs = args.num_epochs
num_classes = args.num_classes
batch_size = args.batch_size
plotting = args.plot
learning_rate = args.learning_rate
# input_shape = (256, 256, 3)    # RGB
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_DEPTH = 1  # grayscale
input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)  # height, width, channel

# NUM_GPUS = 2
# BS_PER_GPU = 128
# NUM_EPOCHS = 60

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
# NUM_CLASSES = 10
# NUM_TRAIN_SAMPLES = 50000

# BASE_LEARNING_RATE = 0.1
# LR_SCHEDULE = [(0.1, 30), (0.01, 45)]

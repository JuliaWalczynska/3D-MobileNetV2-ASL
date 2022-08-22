# training configurations
BATCH_SIZE = 2
IMG_SIZE = 112
INIT_LR = 0.0001
WEIGHT_DECAY = 0.001
EPOCHS = 300
CLASSES = 100

# path to folder with videos
ROOT = 'Z:/Thesis/Video dataset/videos'

# file provided by WLASL authors with information about videos from WLASL100 subset
SPLIT_FILE = 'Z:/Thesis/Video dataset/nslt_100.json'

# saving name and paths for csv file and model
MODEL_NAME = '3DmobileNetV2'
CSV_PATH = MODEL_NAME + '.csv'
SAVE_PATH = MODEL_NAME + '.t7'

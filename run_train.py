
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from train import *
from data import *
from model import *

# Select GPU device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

# data configuration
data = data()
data.data_root = '.\\Data'
data.volume_folder = 'echocardiography'
#data.view = 'A2C' # A2C 학습시  ./Data/echocardiogrphy/train/A2C/에 .png이미지와 .npy라벨 모두 위치시키기
#data.view = 'A4C' # A4C 학습시 ./Data/echocardiogrphy/train/A4C/에 .png이미지와 .npy라벨 모두 위치시키기
data.view = 'ALL' # A2C와 A4C 분류없이 모두 학습시 ./Data/echocardiogrphy/train/ALL/에 .png이미지와 .npy라벨 모두 위치시키기
data.inputA_size = (256,256)
data.inputA_channel = 1
data.inputB_size = (256,256)
data.inputB_channel = 1
data.load_datalist()

# model configuration
model = SegmentationModel()
model.input_shape_A = data.inputA_size + (data.inputA_channel, )
model.input_shape_B = data.inputB_size + (data.inputB_channel, )
model.build_model()

# train configuration
train = train()
train.model_dir = '.\\Model\\echocardiography' #해당 폴더에 모델 저장
train.iter_perEpoch = 1#1600 # 한 에퐄의 배치의 개수
train.make_folders()
train.load_data(data)
train.load_model(model)
train.train()


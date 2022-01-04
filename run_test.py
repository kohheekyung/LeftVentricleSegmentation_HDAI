import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from train import *
from data import *
from model import *

# Select GPU device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

# data configuration (data.view 외 수정 금지)
data = data()
data.data_root = '.\\Data'
data.volume_folder = 'echocardiography'
data.view = 'A2C' # A2C 평가시  ./Data/echocardiogrphy/validation/A2C/에 .png이미지와 .npy라벨 모두 위치시키기
#data.view = 'A4C' # A4C 평가시 ./Data/echocardiogrphy/validation/A4C/에 .png이미지와 .npy라벨 모두 위치시키기
#data.view = 'ALL' # A2C와 A4C 분류없이 모두 평가시 ./Data/echocardiogrphy/validation/ALL/에 .png이미지와 .npy라벨 모두 위치시키기
data.inputA_size = (256,256)
data.inputA_channel = 1
data.inputB_size = (256,256)
data.inputB_channel = 1
data.load_datalist()

#model configuration (모델 구조 재구현, 전부 수정 금지)
model = SegmentationModel()
model.input_shape_A = data.inputA_size + (data.inputA_channel, )
model.input_shape_B = data.inputB_size + (data.inputB_channel, )
model.build_model()

# train configuration (모델 테스트, 전부 수정 금지)
train = train()
train.model_dir = '.\\Model\\echocardiography' # 모델의 최상위 디렉토리
train.test_path = '20211204-112646' # 모델 버전 폴더명
train.test_epoch = 360 # 테스트할 모델의 에퐄   '.\\Model\\echocardiography\\20211204-112646\\models\\model_1_epoch_360.hdf5'
train.make_folders()
train.load_data(data)
train.load_model(model)
train.test()

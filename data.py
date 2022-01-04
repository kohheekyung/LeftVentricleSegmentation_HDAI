import os
import numpy as np
import glob
import cv2

class data():

    def __init__(self):
        # data location
        self.data_root = 'G:/data/Dental/'
        self.volume_folder = 'augmented'
        self.view = 'ALL' # 'A4C' 'BOTH'

        self.inputA_min = 0
        self.inputA_max = 2500
        self.inputB_min = -1000
        self.inputB_max = 2000
        self.tanh_norm = False

        self.inputA_size = (434, 636)
        self.inputA_channel = 1
        self.inputB_size = (434, 636)
        self.inputB_channel = 1

        self.trainA_path = None
        self.trainA_path = None
        self.testA_path = None
        self.testB_path = None

        self.trainA_file_names = None
        self.trainB_file_names = None
        self.testA_file_names = None
        self.testB_file_names = None

    def load_datalist(self):

        # file paths
        self.trainA_path = os.path.join(self.data_root, self.volume_folder, 'train', self.view)
        self.trainB_path = os.path.join(self.data_root, self.volume_folder,'train', self.view)
        self.testA_path = os.path.join(self.data_root, self.volume_folder, 'validation', self.view)
        self.testB_path = os.path.join(self.data_root, self.volume_folder, 'validation',self.view)

        # file names
        self.trainA_file_names = sorted(glob.glob(os.path.join(self.trainA_path, '*.png')))
        self.trainB_file_names = sorted(glob.glob(os.path.join(self.trainB_path, '*.npy')))
        self.testA_file_names = sorted(glob.glob(os.path.join(self.testA_path, '*.png')))
        self.testB_file_names = sorted(glob.glob(os.path.join(self.testB_path, '*.npy')))

    def load_trainset(self, listA, listB):

        source_slices = []
        target_slices = []

        for idx in range(len(listA)):
            source_slice = self.read_png(listA[idx])
            target_slice = self.read_npy(listB[idx])

            source_slice = cv2.resize(source_slice, (self.inputA_size[0], self.inputA_size[1]), interpolation=cv2.INTER_AREA)
            target_slice = cv2.resize(target_slice, (self.inputB_size[0], self.inputB_size[1]), interpolation=cv2.INTER_AREA)

            source_slice = source_slice - np.min(source_slice)
            source_slice = source_slice / (np.max(source_slice) - np.min(source_slice))

            if self.inputA_channel == 1:
                source_slice = np.array(source_slice[:, :,  np.newaxis])
            if self.inputB_channel == 1:
                target_slice = np.array(target_slice[:, :,  np.newaxis])

            source_slices.append(source_slice)
            target_slices.append(target_slice)

        return (np.array(source_slices), np.array(target_slices))

    def load_valset(self, listA, listB):

        source_slices = []
        target_slices = []

        for idx in range(len(listA)):
            #load data
            source_slice = self.read_png(listA[idx])
            target_slice = self.read_npy(listB[idx])

            #resizing
            source_slice = cv2.resize(source_slice, (self.inputA_size[0], self.inputA_size[1]), interpolation=cv2.INTER_AREA)
            target_slice = cv2.resize(target_slice, (self.inputB_size[0], self.inputB_size[1]),  interpolation=cv2.INTER_AREA)

            #normalization
            source_slice = source_slice - np.min(source_slice)
            source_slice = source_slice / (np.max(source_slice) - np.min(source_slice))

            if self.inputA_channel == 1:
                source_slice = np.array(source_slice[:, :,  np.newaxis])
            if self.inputB_channel == 1:
                target_slice = np.array(target_slice[:, :,  np.newaxis])

            source_slices.append(source_slice)
            target_slices.append(target_slice)

        return np.array(source_slices), np.array(target_slices)


    def load_testset(self, listA):

        source_slices = []
        source_shapes = []

        for idx in range(len(listA)):
            source_slice = self.read_png(listA[idx])
            source_shape = source_slice.shape
            print(source_shape)

            source_slice = cv2.resize(source_slice, (self.inputA_size[0], self.inputA_size[1]), interpolation=cv2.INTER_AREA)
            source_shapes.append(source_shape)

            # normalization
            source_slice = source_slice - np.min(source_slice)
            source_slice = source_slice / (np.max(source_slice) - np.min(source_slice))

            if self.inputA_channel == 1:
                source_slice = np.array(source_slice[:, :,  np.newaxis])

            source_slices.append(source_slice)


        return np.array(source_slices), np.array(source_shapes)

    def read_png(self, inputImageFileName):
        return cv2.imread(inputImageFileName)[:,:,0]

    def read_npy(self, inputImageFileName):
        return np.load(inputImageFileName)
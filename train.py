import os
import numpy as np
import glob

import sys
import datetime
import csv
import time
import matplotlib.image as mpimage
from sklearn.utils import shuffle
import cv2
import tensorflow.python.keras.backend as K
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import model

class train():

    def __init__(self):


        self.test_path = False
        self.test_epoch = False
        self.model_dir = 'G:\Model\HDAI'
        self.retrain_path = False  #
        self.retrain_epoch = False  #

        self.augmentation = True

        self.epochs = 400  # choose multiples of 20 since the models are saved each 20th epoch
        self.use_linear_decay = True  #False # Linear decay of learning rate, for both discriminators and generators
        self.decay_epoch = 100 #False #11#101 # The epoch where the linear decay of the learning rates start
        self.batch_size = 1  # Number of volumes per batch
        self.iter_perEpoch = 1000 # 220

        self.model = None

        self.save_training_vol = True  # Save or not example training results or only tmp.png
        self.save_models = True  # Save or not the generator and discriminator models
        self.tmp_vol_update_frequency = 3
        self.train_iterations = 1

    def make_folders(self):

        # ===== Folders and configuration =====
        self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        self.out_dir = os.path.join(self.model_dir, self.date_time)


        if self.retrain_path :
            self.out_dir =  os.path.join(self.model_dir,self.retrain_path)

        if self.test_path:
            self.out_dir =  os.path.join(self.model_dir, self.test_path)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        if self.save_training_vol:
            self.out_dir_volumes = os.path.join(self.out_dir, 'training_volumes')
            if not os.path.exists(self.out_dir_volumes):
                os.makedirs(self.out_dir_volumes)

        if self.save_models:
            self.out_dir_models = os.path.join(self.out_dir, 'models')
            self.out_dir_train_losses = os.path.join(self.out_dir, 'train_losses')
            if not os.path.exists(self.out_dir_models):
                os.makedirs(self.out_dir_models)
            if not os.path.exists(self.out_dir_train_losses):
                os.makedirs(self.out_dir_train_losses)

    def load_data(self, data):
        self.data = data

    def load_model(self, model):
        self.model = model

    def train(self):

        def run_training_batch():
            # ======= Generator training ==========
            synthetic_volumes_B= self.model.base_model.predict(real_volumes_A)

            target_data = []
            for _ in range(len(self.model.compile_weights)) :
                target_data.append(real_volumes_B)

            # Train on batch
            for _ in range(self.train_iterations):
                loss.append(self.model.model.train_on_batch(x=[real_volumes_A], y=target_data))

            # =====================================
            
            # Update learning rates
            if self.use_linear_decay and epoch >= self.decay_epoch:
                self.update_lr(self.model.model, decay, loop_index, epoch)


            losses.append(loss[-1])

            # Print training status
            print('\n')
            print('Epoch ---------------------', epoch, '/', self.epochs)
            print('Loop index ----------------', loop_index_idx, '/',  # ) * loop_index_idx
                  nr_vol_per_epoch)  # * len(self.trainA_file_names))
            print('  Summary:')
            print('lr', K.get_value(self.model.model.optimizer.lr))
            print('loss: ', loss[-1][0])

            idx = 0
            for compile_loss in self.model.compile_losses :
                idx = idx + 1
                try:
                    print(compile_loss.name, loss[-1][idx])
                except:
                    print('additional_loss' + str(idx + 1) + ':', loss[-1][idx])


            self.print_ETA(start_time, epoch, nr_vol_per_epoch, loop_index_idx)
            sys.stdout.flush()

            if loop_index % self.tmp_vol_update_frequency * self.batch_size == 0:
                # Save temporary images continously
                self.save_tmp_images(real_volumes_A[0][:,:,0], synthetic_volumes_B[0][:,:,0], real_volumes_B[0][:,:,0])

        # ======================================================================
        # Begin training
        # ======================================================================
        if self.save_training_vol:
            if not os.path.exists(os.path.join(self.out_dir_volumes, 'train_A')):
                os.makedirs(os.path.join(self.out_dir_volumes, 'train_A'))
                os.makedirs(os.path.join(self.out_dir_volumes, 'test_A'))


        losses = []

        # Start stopwatch for ETAs
        start_time = time.time()
        timer_started = False

        if self.retrain_epoch:
            start = self.retrain_epoch + 1
        else:
            start = 1

        for epoch in range(start, self.epochs + 1):

            loss = []
            loop_index_idx = 1

            trainA_file_name, trainB_file_name = shuffle(self.data.trainA_file_names, self.data.trainB_file_names)
            trainA_file_name = trainA_file_name[: self.iter_perEpoch]
            trainB_file_name = trainB_file_name[: self.iter_perEpoch]

            A_train, B_train = self.data.load_trainset(trainA_file_name, trainB_file_name)

            if self.augmentation:
                A_train, B_train = self.data_augmentation2D(A_train, B_train, epoch)

            # Linear learning rate decay
            if self.use_linear_decay:
                decay = self.get_lr_linear_decay_rate()

            nr_train_vol = self.iter_perEpoch
            nr_vol_per_epoch = int(np.ceil(nr_train_vol / self.batch_size) * self.batch_size)

            random_order = np.concatenate((np.random.permutation(nr_train_vol),
                                           np.random.randint(nr_train_vol, size=nr_vol_per_epoch - nr_train_vol)))

            # Train on volume batch
            for loop_index in range(0, nr_vol_per_epoch, self.batch_size):

                indices = random_order[loop_index:loop_index + self.batch_size]

                real_volumes_A = A_train[indices]
                real_volumes_B = B_train[indices]

                # Train on volume batch
                run_training_batch()

                loop_index_idx += self.batch_size

                # Start timer after first (slow) iteration has finished
                if not timer_started:
                    start_time = time.time()
                    timer_started = True

            # Save training volumes
            print('\n', '\n', '-------------------------Saving volumes for epoch', epoch,
                  '-------------------------',
                  '\n', '\n')
            self.save_epoch_images(epoch)
            self.save_model(self.model.model, epoch)


            print('..........save train loss...........')
            train_losses = {}
            idx = 0
            train_losses['total_loss'] = loss[-1][idx].mean()
            for compile_loss in self.model.compile_losses:
                idx = idx + 1
                try:
                    train_losses[compile_loss.name] = loss[-1][idx].mean()
                except:
                    train_losses['additional_loss' + str(idx + 1)] = loss[-1][idx].mean()



            self.write_loss_data_by_epoch(train_losses, epoch, self.out_dir_train_losses)



    # ===============================================================================
    # Learning rates
    def get_lr_linear_decay_rate(self):
        # Calculate decay rates
        nr_batches_per_epoch = int(np.ceil(self.iter_perEpoch / self.batch_size))

        updates_per_epoch = nr_batches_per_epoch
        nr_decay_updates = (self.epochs - self.decay_epoch + 1) * updates_per_epoch
        decay = self.model.learning_rate / nr_decay_updates
        return decay
    
    def update_lr(self, model, decay, loop_index, epoch):
        new_lr = K.get_value(model.optimizer.lr) - decay
        if new_lr < 0:
            new_lr = 0
        # print(K.get_value(model.optimizer.lr))
        K.set_value(model.optimizer.lr, new_lr)

        if loop_index == 0:
            lr_path = '{}/test_A/epoch{}_lr_{}.npy'.format(self.out_dir_volumes, epoch, loop_index)
            f = open(lr_path, mode='wt', encoding='utf-8')
            f.write(str(new_lr))
            f.close()

    # ===============================================================================
    # save datas
    def join_and_save(self, images, save_path):
        # Join images
        image = np.hstack(images)
        # Save images
        #mpimage.imsave(save_path, image, vmin=-1, vmax=1, cmap='gray')
        mpimage.imsave(save_path, image, cmap='gray')

    def save_tmp_images(self, real_image_A, synthetic_image_B, real_image_B ):
        try:
            save_path = '{}/tmp.png'.format(self.out_dir)
            self.join_and_save((real_image_A, synthetic_image_B, real_image_B), save_path)
        except: # Ignore if file is open
            pass

    def save_epoch_images(self, epoch, num_saved_images=1):

        def jaccard(x, y):
            y = np.where(y<0.5, 0 ,1)

            x = np.asarray(x, np.bool)  # Not necessary, if you keep your data
            y = np.asarray(y, np.bool)  # in a boolean array already!
            return np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())

        def dice(im1, im2):
            im2 = np.where(im2 < 0.5, 0, 1)
            im1 = np.asarray(im1).astype(np.bool)
            im2 = np.asarray(im2).astype(np.bool)
            # Compute Dice coefficient
            intersection = np.logical_and(im1, im2)
            return 2. * intersection.sum() / (im1.sum() + im2.sum())

        # Save training images
        testA_file_name = self.data.testA_file_names  # shuffle(self.data.testA_file_names)
        #testA_file_name = testA_file_name[: 1]
        testB_file_name = self.data.testB_file_names  # shuffle(self.data.testA_file_names)
        #testB_file_name = testB_file_name[: 1]
        A_test, B_test = self.data.load_valset(testA_file_name, testB_file_name)

        nr_train_im = A_test.shape[0]

        A2C_dsc_list= []
        A2C_ji_list = []

        A4C_dsc_list = []
        A4C_ji_list = []

        #
        #rand_ind = np.random.randint(nr_train_im)

        save_path = '{}/test_A/epoch{}'.format(self.out_dir_volumes, epoch)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        i = 0
        for nr_ind in range( nr_train_im ):
            #

            real_image_A = A_test[nr_ind][:,:,0]
            real_image_B = B_test[nr_ind][:,:,0]
            synthetic_image_B = self.model.base_model.predict(A_test)[nr_ind][:,:,0]

            if self.data.testA_file_names[nr_ind].split('\\')[-1].split(".")[0].split("_")[-1] == 'A2C':
                A2C_dsc_list.append(dice(real_image_B, synthetic_image_B))
                A2C_ji_list.append(jaccard(real_image_B, synthetic_image_B))
            else:
                A4C_dsc_list.append(dice(real_image_B, synthetic_image_B))
                A4C_ji_list.append(jaccard(real_image_B, synthetic_image_B))

            img_path = save_path +'\\{}'.format(self.data.testA_file_names[nr_ind].split('\\')[-1])

            if i == 0 :
                self.join_and_save((real_image_A, synthetic_image_B, real_image_B), img_path)

            i = i + 1



        f = open(save_path+'/A2Cdsc.txt', 'w')
        f.write("%f %f" % (np.mean(np.array(A2C_dsc_list)), np.std(np.array(A2C_dsc_list)) ))
        f.close()

        f = open(save_path+'/A2CJAC.txt', 'w')
        f.write("%f %f" %  ( np.mean(np.array(A2C_ji_list)), np.std(np.array(A2C_ji_list)) ))
        f.close()

        f = open(save_path + '/A4Cdsc.txt', 'w')
        f.write("%f %f" % (np.mean(np.array(A4C_dsc_list)), np.std(np.array(A4C_dsc_list))))
        f.close()

        f = open(save_path + '/A4CJAC.txt', 'w')
        f.write("%f %f" % (np.mean(np.array(A4C_ji_list)), np.std(np.array(A4C_ji_list))))
        f.close()


    def save_model(self, model, epoch):

        weights_path = '{}/{}_epoch_{}.hdf5'.format(self.out_dir_models, model.name, epoch)
        model.save_weights(weights_path)

        model_path = '{}/{}_epoch_{}.json'.format(self.out_dir_models, model.name, epoch)
        model_json_string = model.to_json()
        with open(model_path, 'w') as outfile:
            outfile.write(model_json_string)
        print('{} has been saved in saved_models/{}/'.format(model.name, self.date_time))

    def write_loss_data_by_epoch(self, losses, epoch , path):
        keys = losses.keys()
        with open(path + '/' + '{:03d}.csv'.format(epoch), 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = keys)
            writer.writeheader()
            writer.writerow(losses)
            #$writer.writerow(*[losses[key] for key in keys])


    def test(self):

        def jaccard(x, y):
            y = np.where(y<0.5, 0 ,1)

            x = np.asarray(x, np.bool)  # Not necessary, if you keep your data
            y = np.asarray(y, np.bool)  # in a boolean array already!
            return np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())

        def dice(im1, im2):
            im2 = np.where(im2 < 0.5, 0, 1)
            im1 = np.asarray(im1).astype(np.bool)
            im2 = np.asarray(im2).astype(np.bool)
            # Compute Dice coefficient
            intersection = np.logical_and(im1, im2)
            return 2. * intersection.sum() / (im1.sum() + im2.sum())

        # load generator A to B
        path_to_weights = glob.glob(os.path.join(self.model_dir, self.test_path, 'models', '*'+'{}.hdf5'.format(self.test_epoch)) )[0]
        self.model.model.load_weights(path_to_weights)

        real_path = os.path.join(self.data.data_root, self.data.volume_folder, 'realLabel')
        if not os.path.exists(real_path):
            os.mkdir(real_path)

        syn_path = os.path.join(self.data.data_root, self.data.volume_folder, 'expectedLabel')
        if not os.path.exists(syn_path):
            os.mkdir(syn_path)

        all_path = os.path.join(self.data.data_root, self.data.volume_folder, 'comparedLabel')
        if not os.path.exists(all_path):
            os.mkdir(all_path)


        testA_file_name = self.data.testA_file_names
        #testB_file_name = self.data.testB_file_names
        A_test, A_shape = self.data.load_testset(testA_file_name)

        nr_train_im = A_test.shape[0]

        # dsc_list = []
        # ji_list = []
        #
        # A2C_dsc_list= []
        # A2C_ji_list = []
        #
        # A4C_dsc_list = []
        # A4C_ji_list = []

        for nr_ind in range(nr_train_im):

            print('predict...', self.data.testA_file_names[nr_ind].split('\\')[-1])
            real_image_A = A_test[nr_ind][:, :, 0]
            #real_image_B = B_test[nr_ind][:, :, 0]
            synthetic_image_B = self.model.base_model.predict(A_test)[nr_ind][:, :, 0]

            #dsc_list.append(dice(real_image_B, synthetic_image_B))
            #ji_list.append(jaccard(real_image_B, synthetic_image_B))

            # if self.data.testA_file_names[nr_ind].split('\\')[-1].split(".")[0].split("_")[-1] == 'A2C':
            #     A2C_dsc_list.append(dice(real_image_B, synthetic_image_B))
            #     A2C_ji_list.append(jaccard(real_image_B, synthetic_image_B))
            # else:
            #     A4C_dsc_list.append(dice(real_image_B, synthetic_image_B))
            #     A4C_ji_list.append(jaccard(real_image_B, synthetic_image_B))
            synthetic_image_B = np.where(synthetic_image_B > 0.5, 1, 0.0)
            # if nr_ind == nr_train_im -1 :
            #     from scipy import ndimage
            #     # Get largest continuous image
            #
            #     label_im, nb_labels = ndimage.label(synthetic_image_B)
            #     sizes = ndimage.sum(synthetic_image_B, label_im, range(nb_labels + 1))
            #     mask = sizes == np.max(sizes)
            #     mask_ = mask[label_im]
            #
            #     synthetic_image_B = np.where(mask_, synthetic_image_B , 0.0)

            realImg_path =  real_path + '\\{}'.format(self.data.testA_file_names[nr_ind].split('\\')[-1])
            synImg_path = syn_path + '\\{}'.format(self.data.testA_file_names[nr_ind].split('\\')[-1])
            #allImg_path = all_path + '\\{}'.format(self.data.testA_file_names[nr_ind].split('\\')[-1])

            real_image_A = cv2.resize(real_image_A, (A_shape[nr_ind][1],A_shape[nr_ind][0]) , cv2.INTER_LANCZOS4)
            synthetic_image_B = cv2.resize(synthetic_image_B, (A_shape[nr_ind][1],A_shape[nr_ind][0]), cv2.INTER_LANCZOS4)


            mpimage.imsave(realImg_path, real_image_A, cmap='gray')
            mpimage.imsave(synImg_path, synthetic_image_B, cmap='gray')
            #self.join_and_save((real_image_A, synthetic_image_B, real_image_B), allImg_path)

            #realImg_path = real_path + '\\{}'.format(self.data.testB_file_names[nr_ind].split('\\')[-1])
            synImg_path = syn_path + '\\{}'.format(self.data.testA_file_names[nr_ind].split('\\')[-1].split('.png')[0])
            #np.save(realImg_path, real_image_B)
            np.save(synImg_path, synImg_path)

        # f = open(os.path.join(self.data.data_root, self.data.volume_folder, 'evaluationDSC.txt'), 'w')
        # f.write("%f %f" % (np.mean(np.array(dsc_list)), np.std(np.array(dsc_list))))
        # f.close()
        # print('총 ' + str(nr_train_im)+'개의 데이터 평가 결과')
        # print("DSC mean: ", str(np.mean(np.array(dsc_list))), "std: ", str(np.std(np.array(dsc_list))))
        #
        # f = open(os.path.join(self.data.data_root, self.data.volume_folder, 'evaluationJAC.txt'), 'w')
        # f.write("%f %f" % (np.mean(np.array(ji_list)), np.std(np.array(ji_list))))
        # f.close()
        #
        # print("JI mean: ", str(np.mean(np.array(ji_list))), "std: ", str(np.std(np.array(ji_list))))




    def retrain(self):

        # load generator A to B
        path_to_weights = glob.glob( os.path.join(self.model_dir, self.retrain_path, 'models','*'+'{}.hdf5'.format(self.retrain_epoch)))[0]
        self.model.model.load_weights(path_to_weights)

        self.train()

    # ===============================================================================
    def data_augmentation2D(self, A_train, B_train, epoch):

        #source = A_train
        #target = B_train

        augmented_source_list = []
        augmented_target_list = []

        # create image data augmentation generator
        datagen = ImageDataGenerator(rotation_range=4.5, zoom_range=0.3, shear_range=0.04, horizontal_flip=True)
        # shear_range=0.03=
        for i in range(A_train.shape[0]):

            itA = datagen.flow(A_train[i][np.newaxis, :, :,:], batch_size=1, seed=epoch)
            itB = datagen.flow(B_train[i][np.newaxis, :, :,:], batch_size=1, seed=epoch)

            # generate batch of images
            batchA = itA.next()
            batchB = itB.next()

            imageA = batchA[0, :, :, 0]
            imageB = batchB[0, :, :, 0]

            augmented_source_list.append(imageA[:, :, np.newaxis])
            augmented_target_list.append(imageB[:, :, np.newaxis])

        return (np.array(augmented_source_list), np.array(augmented_target_list))

    # ==============================================================================
    # Other output
    def print_ETA(self, start_time, epoch, nr_vol_per_epoch, loop_index):
        passed_time = time.time() - start_time

        iterations_so_far = ((epoch - 1) * nr_vol_per_epoch + loop_index) / self.batch_size
        iterations_total = self.epochs * nr_vol_per_epoch / self.batch_size
        iterations_left = iterations_total - iterations_so_far
        eta = round(passed_time / (iterations_so_far + 1e-5) * iterations_left)

        passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))

        try:
            eta_string = str(datetime.timedelta(seconds=eta))
            print('Elapsed time', passed_time_string, ': ETA in', eta_string)
        except:
            print('Too long elapsed time')

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))

            else:  # 50% chance that we replace an old synthetic image
                p = np.random.rand()
                if p > 0.5:
                    random_id = np.random.randint(0, self.pool_size)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))

        return return_images

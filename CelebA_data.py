import os
import cv2
import numpy as np
from keras.utils import to_categorical

def deprocess(x):
    return np.reshape((x * 255.0), (-1, 64, 64, 3)).astype(np.uint8)

def preprocess(x):
    return x.astype(np.float) / 255.0

def crop_and_resize(img, bbox=(40, 218 - 30, 15, 178 - 15), target_size=(64, 64)):
    img_crop = img[bbox[0]:bbox[1], bbox[2]:bbox[3], :]
    img_resize = cv2.resize(img_crop, target_size, interpolation=cv2.INTER_AREA)
    return img_resize

def split_and_generate_npy(dataset_root_dir='/path/to/your/dataset/CelebA/', output_root_dir='/path/to/your/dataset/CelebA/GenderSplit'):
    list_eval_partition_path = os.path.join(dataset_root_dir, 'Eval/list_eval_partition.txt')
    list_attr_path = os.path.join(dataset_root_dir, 'Anno/list_attr_celeba.txt')
    img_dir = os.path.join(dataset_root_dir, 'Img/img_align_celeba/')

    partition_file = open(list_eval_partition_path, mode='r')
    partitions = partition_file.readlines()
    partition_file.close()

    attr_file = open(list_attr_path, mode='r')
    attrs = attr_file.readlines()
    attr_file.close()
    attr_header = attrs[1]
    GENDER_INDEX = attr_header.split().index('Male') + 1
    attrs = attrs[2:]

    # train val test buffer
    train_imgs, train_labels = [], []
    val_imgs, val_labels = [], []
    test_imgs, test_labels = [], []

    num_of_records = len(partitions)
    num_of_train, num_of_test, num_of_val = 0, 0, 0
    num_of_male, num_of_female = 0, 0

    for idx, partition in enumerate(partitions):
        if idx % 500 == 0:
            print('Processing {} / {}'.format(idx, num_of_records))
        partition = partition.split()
        attr = attrs[idx].split()

        assert len(partition) == 2, 'Partition format Error, you must down load the list_eval_partition.txt file from the CelebA website.'
        assert partition[0] == attr[0], 'Image name doesn\'t match between partition file and attribute file.'

        img_name = partition[0]
        flag = int(partition[1])
        gender_flag = int(attr[GENDER_INDEX])

        img = cv2.imread(os.path.join(img_dir, img_name))
        img_crop_and_resize = crop_and_resize(img)

        if flag == 0:  # training image
            train_labels.append((0 if gender_flag == -1 else 1))
            train_imgs.append(img_crop_and_resize)
            num_of_train += 1

        elif flag == 1:  # validating image
            val_labels.append((0 if gender_flag == -1 else 1))
            val_imgs.append(img_crop_and_resize)
            num_of_val += 1

        elif flag == 2:  # testing image
            test_labels.append((0 if gender_flag == -1 else 1))
            test_imgs.append(img_crop_and_resize)
            num_of_test += 1
        else:
            raise ValueError('Partition flag must to be 0(train), 1(validation) or 2(test), given {}, '
                             'you must down load the list_eval_partition.txt file from the CelebA website.'.format(flag))
        if gender_flag == -1:
            num_of_female += 1
        else:
            num_of_male += 1

    print('Total {} images'.format(num_of_records))
    print('Traing: {}, Validating: {}, Testing: {}'.format(num_of_train, num_of_val, num_of_test))
    print('Female: {}, Male: {}'.format(num_of_female, num_of_male))

    np.save(os.path.join(output_root_dir, 'train_imgs.npy'), train_imgs)
    np.save(os.path.join(output_root_dir, 'val_imgs.npy'), val_imgs)
    np.save(os.path.join(output_root_dir, 'test_imgs.npy'), test_imgs)
    np.save(os.path.join(output_root_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(output_root_dir, 'val_labels.npy'), val_labels)
    np.save(os.path.join(output_root_dir, 'test_labels.npy'), test_labels)

# split_and_generate_npy()

def load_celebA_Gender(data_dir='/path/to/your/dataset/CelebA/', onehot=False, test_only=False, prep=True):
    if not test_only:
        x_train = np.load(os.path.join(data_dir, 'train_imgs.npy'))
        y_train = np.load(os.path.join(data_dir, 'train_labels.npy'))
    else:
        x_train = None
        y_train = None

    x_test = np.load(os.path.join(data_dir, 'val_imgs.npy'))
    y_test = np.load(os.path.join(data_dir, 'val_labels.npy'))

    if not test_only:
        x_train = x_train[:, :, :, ::-1].transpose((0, 3, 1, 2))

    x_test = x_test[:, :, :, ::-1].transpose((0, 3, 1, 2))

    if onehot:
        y_train = to_categorical(y_train, num_classes=2) if not test_only else None
        y_test = to_categorical(y_test, num_classes=2)
    else:
        y_train = np.reshape(y_train, newshape=(y_train.shape[0], 1)) if not test_only else None
        y_test = np.reshape(y_test, newshape=(y_test.shape[0], 1))

    if prep:
        x_train = preprocess(x_train) if not test_only else None
        x_test = preprocess(x_test)

    return x_train, y_train, x_test, y_test
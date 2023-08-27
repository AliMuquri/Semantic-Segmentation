import argparse
from VOC import *
import os

def main():

    parser = argparse.ArgumentParser(prog='VOC',
                                    description='Help Program for using labelme and converting to VOC',
                                     )

    parser.add_argument('-V', nargs=2, help='''Use for creating a VOC structure out of labeled files .JSON
                                            ''', required=True)


    label_2_voc = VOC()
    args = parser.parse_args()
    input_path = args.V[0]
    output_path = args.V[1]
    label_2_voc.VOC_multiple_folders(input_path, output_path)

    print("Checking if all files are accounted for.")
    train_jsons = [x[:-len('.json')] for x in os.listdir(os.path.join(input_path, 'train')) if x.endswith('.json')]
    test_jsons = [x[:-len('.json')] for x in os.listdir(os.path.join(input_path, 'test')) if x.endswith('.json')]
    val_jsons = [x[:-len('.json')] for x in os.listdir(os.path.join(input_path, 'val')) if x.endswith('.json')]
    train_jpgs = [x[:-len('.jpg')] for x in os.listdir(os.path.join(os.path.join(os.path.join(output_path, 'dataset_voc'), 'train'), 'JPEGImages')) if x.endswith('.jpg')]
    test_jpgs = [x[:-len('.jpg')] for x in os.listdir(os.path.join(os.path.join(os.path.join(output_path, 'dataset_voc'), 'test'), 'JPEGImages')) if x.endswith('.jpg')]
    val_jpgs = [x[:-len('.jpg')] for x in os.listdir(os.path.join(os.path.join(os.path.join(output_path, 'dataset_voc'), 'val'), 'JPEGImages')) if x.endswith('.jpg')]

    check_train_size = len(train_jpgs) == len(train_jsons)
    check_test_size = len(test_jpgs) == len(test_jsons)
    check_val_size = len(val_jpgs) == len(val_jsons)

    check_train_elements = set(train_jpgs) == set(train_jsons)
    check_test_elements = set(test_jpgs) == set(test_jsons)
    check_val_elements = set(val_jpgs) == set(val_jsons)

    if not check_train_size:
        print("TRAIN")
        print("Size different jpgs: {} and jsons: {}".format(len(train_jpgs), len(train_jsons)))
    else:
        print("Training set correct: {} images".format(len(train_jpgs)))

    if not check_test_size:
        print("TEST")
        print("Size different jpgs: {} and jsons: {}".format(len(test_jpgs), len(test_jsons)))
    else:
        print("Test set correct: {} images".format(len(test_jpgs)))

    if not check_val_size:
        print("VAL")
        print("Size different jpgs: {} and jsons: {}".format(len(val_jpgs), len(val_jsons)))
    else:
        print("Val set correct: {} images".format(len(val_jpgs)))

    if not check_train_elements:
        print("TRAIN elements not the same")
    else:
        print("Training correct elements")

    if not check_test_elements:
        print("Test elements not the same")
    else:
        print("Test correct elements")

    if not check_val_elements:
        print("Val elements not the same")
    else:
        print("Val correct elements")
if __name__ == '__main__':

    main()

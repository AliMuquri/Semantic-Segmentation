import os
import sys
import subprocess
from subprocess import DEVNULL
import shutil
import json
import time


class FolderStructureError(Exception):
    """This exception is raised when a path way is given that follows a different multi folder structure
    then what is expected (train, test, val)"""
    def __init__(self, message):
        self.message = "{} is not permitted FolderStructure. Please follow train, test, val ".format(message)
        super().__init__(self, message)
        self.message = message

class VOC():
    """ Labelme provides a way to handle multiple files in one single command using labelme2voc.py
        This class is customized to handle a voc structured input of images and json files and output
        a voc structured output of images and annotated images at a specific location. """

    def __init__(self):
        pass


    def filename_handler(self, path):

        base_name = None
        if path.endswith('train'):
            base_name = 'train_'
        elif path.endswith('test'):
            base_name = 'test_'
        elif path.endswith('val'):
            base_name = 'val_'
        else:
            raise FolderStructureError(path)

        indexer = 0

        for filename in os.listdir(path):
            if filename.endswith('.json'):
                if filename.startswith(base_name):
                    splitname = filename.split('_')[1]
                    index = splitname.split('.')[0]
                    if indexer < int(index):
                        indexer = int(index)

        indexer+=1
        for filename in os.listdir(path):
            if filename.endswith('.json'):
                if filename.startswith(base_name):
                    pass
                else:
                    file = open(os.path.join(path,filename), 'r')
                    data = json.load(file)
                    file.close()
                    data.update({'imagePath': base_name + str(indexer) + '.jpg'})

                    file = open(os.path.join(path,filename), 'w')
                    json.dump(data, file, ensure_ascii=False, indent=2)
                    file.close()


                    # with open(os.path.join(path,filename), 'r+') as f:
                    #     file = json.load(f)
                    #     file['imagePath'] = base_name + str(indexer) + '.jpg'
                    #     f.seek(0)
                    #     json.dump(file, f, ensure_ascii=False, indent=2)

                    os.rename(os.path.join(path, filename), os.path.join(path, base_name + str(indexer) + '.json'))

                    os.rename(os.path.join(path, filename[:-len('.json')]+ '.jpg'), os.path.join(path, base_name + str(indexer) + '.jpg'))

                    indexer+=1




    def VOC_folder(self, input_path, output_path):
        """
            This function process a folder with images and corresponding annotation in json files. 
            This function envokes subprocesses and labelme function to turn the json files to semantically annotation images.

            Arguments:
            input_path -- the folder path to images and corresponding json files, string
            output_path -- the folder path were images and corresponding annotation images will be saved, string
        """
        JPEG_path, SegmentationClass_path, SegmentationClassPNG_path,SegmenationClassVisualization_path = None, None, None, None

        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        JPEG_path = os.path.join(output_path,'JPEGImages')
        #SegmentationClass_path = os.path.join(output_path,'SegmentationClass')
        SegmentationClassPNG_path = os.path.join(output_path,'SegmentationClassPNG')
        SegmenationClassVisualization_path = os.path.join(output_path,'SegmenationClassVisualization')


        for final_paths in [JPEG_path, SegmentationClassPNG_path, SegmenationClassVisualization_path]:
            if not os.path.isdir(final_paths):
                os.mkdir(final_paths)

        processes = []

        self.filename_handler(input_path)

        tot = len([x for x in os.listdir(input_path) if x.endswith('.json')])

        for filename in os.listdir(input_path):
            if filename.endswith('.json'):
                new_file_name = filename[:-len('json')]
                new_output_path = os.path.join(output_path, new_file_name)
                if not os.path.isdir(new_output_path):
                    os.mkdir(new_output_path)
                #os.mkdir(os.path.join(output_path, filename.removesuffix('.json')))
                cmd =r'labelme_json_to_dataset "{}" -o "{}" '.format(os.path.join(input_path, filename), new_output_path)
                process = subprocess.Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
                processes.append(process)
                print("{} of {} jsons has been turned into VOC".format(len(processes), tot), end='\r')
                if len(processes)%40 == 0:
                    time.sleep(5)
        print('\n')

        for process in processes:
            process.wait()

        for filename in os.listdir(input_path):
            if filename.endswith('.json'):
                new_file_name = filename[:-len('json')]
                new_output_path = os.path.join(output_path, new_file_name)
                try:
                    os.rename(os.path.join(new_output_path, 'img.png'), os.path.join(JPEG_path, new_file_name + 'jpg'))
                    os.rename(os.path.join(new_output_path, 'label.png'), os.path.join(SegmentationClassPNG_path, new_file_name + '.png'))
                    os.rename(os.path.join(new_output_path, 'label_viz.png'), os.path.join(SegmenationClassVisualization_path, new_file_name + '.jpg'))
                    os.remove(os.path.join(new_output_path, 'label_names.txt'))
                    os.rmdir(new_output_path)
                except:
                    shutil.rmtree(new_output_path)


    def VOC_multiple_folders(self, input_path, output_path):

        """
            This function iterates over all folders inside a main folder following the VOC structure (test, train, val)

            Arguments:
            input_path -- Is the path to the main folder containing multiple folders containing image files and json file annotations, string
            output_path -- Is the path to where the main folder where all subfolders and their images, annotations images will be saved, string
        """
        output_path = os.path.join(output_path, 'dataset_voc')
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        for folder in os.listdir(input_path):
            print(f"for {folder} creating VOC".format(folder))
            folder_path = os.path.join(input_path, folder)
            if not os.path.isfile(folder_path):
                folder_output_path = os.path.join(output_path, folder)
                self.VOC_folder(folder_path, folder_output_path)

from torch.utils.data import Dataset

import os

import random

import torch

import numpy as np

from PIL import Image

import cv2

from diffusers.utils import load_image


CROP_x= 512
CROP_y= 320


NUM_BINS= 6

EV_CHANNELS= NUM_BINS



'''
For 2x 
'''

UPSAMPLE_SCALE= [2]


def get_views(panorama_height, panorama_width, overlap_ratio=0.5):
    
    panorama_height /= 8
    panorama_width /= 8

    print('panorama_height', panorama_height)
    print('panorama_width', panorama_width)

    window_size_x= CROP_x // 8
    window_size_y= CROP_y // 8


    stride_x = int(window_size_x * (1 - overlap_ratio))
    stride_y = int(window_size_y * (1 - overlap_ratio))


    # num_blocks_height = (panorama_height - window_size_y) // stride_y + 1
    # num_blocks_width = (panorama_width - window_size_x) // stride_x + 1

    # account for residual blocks
    num_blocks_height = (panorama_height - window_size_y) // stride_y + 1
    if (panorama_height - window_size_y) % stride_y != 0:
        num_blocks_height += 1 
    
    num_blocks_width = (panorama_width - window_size_x) // stride_x + 1
    if (panorama_width - window_size_x) % stride_x != 0:
        num_blocks_width += 1



    total_num_blocks = int(num_blocks_height * num_blocks_width)

    views = []

    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride_y)
        h_end = h_start + window_size_y
        if h_end > panorama_height:
            h_end = panorama_height
            h_start = h_end - window_size_y
        w_start = int((i % num_blocks_width) * stride_x)
        w_end = w_start + window_size_x
        if w_end > panorama_width:
            w_end = panorama_width
            w_start = w_end - window_size_x
        
        views.append((w_start, h_start, w_end, h_end))


    return views


def apply_crop(image, start_x, start_y, crop_size_x, crop_size_y):
    image_croped = image.crop((start_x, start_y, start_x + crop_size_x, start_y + crop_size_y))

    return image_croped  


def get_random_crop_idx(image, crop_size_x, crop_size_y):
    width, height = image.size

    start_x = random.randint(0, width - crop_size_x)
    start_y = random.randint(0, height - crop_size_y)

    return start_x, start_y

def divide_image(image, crop_size_x, crop_size_y):
    width, height = image.size

    num_parts_x = width // crop_size_x
    num_parts_y = height // crop_size_y
    
    residual_x = width % crop_size_x
    residual_y = height % crop_size_y

    image_parts = []

    for i in range(num_parts_x):
        for j in range(num_parts_y):
            start_x = i * crop_size_x
            start_y = j * crop_size_y

            image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

            image_parts.append(image_croped)
    
    if residual_x > 0:
        for j in range(num_parts_y):
            start_x = width - crop_size_x
            start_y = j * crop_size_y

            image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

            image_parts.append(image_croped)
    
    if residual_y > 0:
        for i in range(num_parts_x):
            start_x = i * crop_size_x
            start_y = height - crop_size_y

            image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

            image_parts.append(image_croped)
    
    if residual_x > 0 and residual_y > 0:
        start_x = width - crop_size_x
        start_y = height - crop_size_y

        image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

        image_parts.append(image_croped)


    return image_parts


def get_image_parts(image, crop_size_x, crop_size_y, idx):
    width, height = image.size

    num_parts_x = width // crop_size_x
    num_parts_y = height // crop_size_y
    
    residual_x = width % crop_size_x
    residual_y = height % crop_size_y

    image_parts = []

    for i in range(num_parts_x):
        for j in range(num_parts_y):
            start_x = i * crop_size_x
            start_y = j * crop_size_y

            image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

            image_parts.append(image_croped)
    
    if residual_x > 0:
        for j in range(num_parts_y):
            start_x = width - crop_size_x
            start_y = j * crop_size_y

            image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

            image_parts.append(image_croped)
    
    if residual_y > 0:
        for i in range(num_parts_x):
            start_x = i * crop_size_x
            start_y = height - crop_size_y

            image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

            image_parts.append(image_croped)
    
    if residual_x > 0 and residual_y > 0:
        start_x = width - crop_size_x
        start_y = height - crop_size_y

        image_croped = apply_crop(image, start_x, start_y, crop_size_x, crop_size_y)

        image_parts.append(image_croped)


    return image_parts[idx]






def check_all_files_exist(file_list, folder_path):

    return all([os.path.isfile(os.path.join(folder_path, file)) for file in file_list])



class DummyDataset(Dataset):
    def __init__(self, base_folder, num_samples=100000, width=1024, height=576, sample_frames=14, valid = False):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        self.num_samples = num_samples
        # Define the path to the folder containing video frames
        # self.base_folder =  '/fs/nexus-projects/DroneHuman/jxchen/data/04_ev/Control_SVD/Finetune_data/bdd100k/images/track/mini'

        # self.base_folder =  '/fs/nexus-projects/DroneHuman/jxchen/data/04_ev/Control_SVD/Finetune_data/ev_svd_mini_rgb'
        self.base_folder = base_folder
        self.folders = os.listdir(self.base_folder)
        self.channels = 3
        self.ev_channels = EV_CHANNELS
        self.width = width
        self.height = height
        self.sample_frames = sample_frames

        self.valid = valid

    def __len__(self):
        train_txt= os.path.join(self.base_folder, "../TRAINING.txt")

        with open(train_txt) as f:
            num_samples = f.readlines()
            num_samples= int(num_samples[0])
            num_samples= num_samples // self.sample_frames
            # print('num_samples', num_samples)

        return num_samples

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        # Randomly select a folder (representing a video) from the base folder

        chosen_folder = random.choice(self.folders)
        folder_path = os.path.join(self.base_folder, chosen_folder)




        # Get from rgb folder
        rgb_folder_path = os.path.join(folder_path, 'image')
        
        frames = os.listdir(rgb_folder_path)
        # Sort the frames by name
        frames.sort()

        
        # Ensure the selected folder has at least `sample_frames`` frames
        if len(frames) < self.sample_frames:
            raise ValueError(
                f"The selected folder '{chosen_folder}' contains fewer than `{self.sample_frames}` frames.")
        
        # Randomly select a start index for frame sequence
        # start_idx = random.randint(0, len(frames) - self.sample_frames)
        
        start_idx = random.randint(0, len(frames) - self.sample_frames)


        test_img= Image.open(os.path.join(rgb_folder_path, frames[start_idx]))



        # selected_frames = frames[start_idx:start_idx + self.sample_frames]
        selected_frames = frames[start_idx:start_idx + self.sample_frames]

        while check_all_files_exist(selected_frames, rgb_folder_path) == False:
            start_idx = random.randint(0, len(frames) - self.sample_frames)
            selected_frames = frames[start_idx:start_idx + self.sample_frames]

        
        # ger random crop idx
        rand_x, rand_y= get_random_crop_idx(test_img, CROP_x, CROP_y)

        # selected_frames.reverse()


        '''
        Random select a scale/ Fix scale
        '''
        # scale_idx= random.randint(0, len(UPSAMPLE_SCALE) - 1)

        scale= UPSAMPLE_SCALE[0]


        # print('scale', scale)


   

        # Initialize a tensor to store the pixel values
        # pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))
        pixel_values = torch.empty((self.sample_frames, self.channels, self.height, self.width))


        # Load and process each frame
        for i, frame_name in enumerate(selected_frames):
            frame_path = os.path.join(rgb_folder_path, frame_name)


            if i == 0:
                init_frame_path = frame_path
                # print('init_frame', init_frame_path)

            # print('frame', frame_path)

            with Image.open(frame_path) as img:
        

                '''
                Code block for scale image: start
                '''

                image_weigth, image_height = img.size

                # print('image_weigth, image_height', image_weigth, image_height)


                img= img.resize((int(image_weigth * scale), int(image_height * scale)))

                # print('img size', img.size)

                # exit(0)

                '''
                Code block for scale image: end
                '''

        

                img_resized= apply_crop(img, rand_x, rand_y, CROP_x, CROP_y)


                
                img_tensor = torch.from_numpy(np.array(img_resized)).float()

                # Normalize the image by scaling pixel values to [-1, 1]
                img_normalized = img_tensor / 127.5 - 1

                # Rearrange channels if necessary
                if self.channels == 3:
                    img_normalized = img_normalized.permute(
                        2, 0, 1)  # For RGB images
                elif self.channels == 1:
                    img_normalized = img_normalized.mean(
                        dim=2, keepdim=True)  # For grayscale images

                pixel_values[i] = img_normalized
        
        '''
        Modified by Jxchen:
        Add event frame
        '''
        ev_folder_path = os.path.join(folder_path, 'event')


        # Initialize a tensor to store the pixel values
        ev_pixel_values = torch.empty((self.sample_frames, self.ev_channels, self.height, self.width))



       # Load and process each frame
        for i, frame_name in enumerate(selected_frames):

            # old version: if the last frame is not exist, use zero frame
            if i == 0:
                zero_frame = torch.zeros((self.ev_channels, self.height, self.width))
                zero_frame = zero_frame.float()


                ev_pixel_values[i] = zero_frame
                continue


            frame_path = os.path.join(ev_folder_path, frame_name)

            # print('event', frame_path)


            for bins in range(NUM_BINS):
                # print('frame_name', frame_name)
                frame_path= os.path.join(ev_folder_path, frame_name.split('.')[0] + f'_{str(bins).zfill(2)}.png')

                # print('frame_path', frame_path)

                with Image.open(frame_path) as img:
                    # Resize the image and convert it to a tensor
                    # img_resized = img.resize((self.width, self.height))
                    # img= np.array(img)

                    '''
                    Code block for scale event image: start
                    '''

                    image_weigth, image_height = img.size

                    # print('event image_weigth, image_height', image_weigth, image_height)

                    img= img.resize((int(image_weigth * scale), int(image_height * scale)))

                    # print('event img size', img.size)

                    '''
                    Code block for scale event image: end
                    '''


                  
                    
                    img_resized= apply_crop(img, rand_x, rand_y, CROP_x, CROP_y)


                    img_tensor = torch.from_numpy(np.array(img_resized)).float()


                    img_tensor_1ch= img_tensor

                    # Normalize the image by scaling pixel values to [-1, 1]
                    img_normalized = img_tensor_1ch / 127.0 - 1

                    img_normalized = img_normalized.unsqueeze(0)


                    '''
                    bins! not i: ev_pixel_values[i][bins] !!!!!!
                    '''
                    ev_pixel_values[i][bins] = img_normalized

        # exit(0)
            

        return {'pixel_values': pixel_values, "event_values": ev_pixel_values, 'init_frame': init_frame_path}
    

def center_crop(img, new_width, new_height):
    width, height = img.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return img.crop((left, top, right, bottom))


def get_valid_image_bins(image_path, idx, width, height, num_frames, scale= UPSAMPLE_SCALE[0]):
    image_file_path= os.listdir(image_path)[idx]

    image_file_path= os.path.join(image_path, image_file_path)

    # print('image_file_path', image_file_path)


    # valid_image= load_image(image_file_path).resize((width, height))
    valid_image= load_image(image_file_path)

    '''
    Code block for scale image: start
    '''

    image_weigth, image_height = valid_image.size

    valid_image= valid_image.resize((int(image_weigth * scale), int(image_height * scale)))

    '''
    Code block for scale image: end
    '''

    valid_image= center_crop(valid_image, width, height)

    '''
    Load valid event frame: error here
    '''
    rgb_folder_path = image_path

    frames = os.listdir(rgb_folder_path)
    # Sort the frames by name
    frames.sort()


    # selected_frames = frames[start_idx:start_idx + self.sample_frames]
    selected_frames = frames[idx:idx + num_frames]



    event_path= image_path.replace('image', 'event')


    ev_pixel_values = torch.empty((num_frames, EV_CHANNELS, height, width))

    for i in range(num_frames):

        '''
        Load valid event frame: error here
        '''
        frame_name = selected_frames[i]
        frame_path= os.path.join(event_path, frame_name)



        if i == 0:
            zero_frame = torch.zeros((EV_CHANNELS, height, width))
            zero_frame = zero_frame.float()
 
            ev_pixel_values[i] = zero_frame
            continue

        for bins in range(NUM_BINS):
            
            '''
            Load valid event frame: error here
            '''
            frame_path= os.path.join(event_path, frame_name.split('.')[0] + f'_{str(bins).zfill(2)}.png')
            

            print('frame_path', frame_path)

            with Image.open(frame_path) as img:

                '''
                Code block for scale event image: start
                '''

                image_weigth, image_height = img.size

                img= img.resize((int(image_weigth * scale), int(image_height * scale)))

                '''
                Code block for scale event image: end
                '''
                
                img_resized= center_crop(img, width, height)

                img_tensor = torch.from_numpy(np.array(img_resized)).float()


                # single channel
                # img_tensor_1ch= img_tensor[:, :, 0] - img_tensor[:, :, 2]
                # 2 channels
                img_tensor_1ch= img_tensor

                # Normalize the image by scaling pixel values to [-1, 1]
                img_normalized = img_tensor_1ch / 127.0 - 1

                img_normalized = img_normalized.unsqueeze(0)


                '''
                bins! not i: ev_pixel_values[i][bins] !!!!!!
                '''
                ev_pixel_values[i][bins] = img_normalized
    

    return valid_image, ev_pixel_values
        




class ValidDataset(Dataset):
    def __init__(self, base_folder, num_samples=100000, width=1024, height=576, sample_frames=14):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        self.num_samples = num_samples
        # Define the path to the folder containing video frames
        # self.base_folder =  '/fs/nexus-projects/DroneHuman/jxchen/data/04_ev/Control_SVD/Finetune_data/bdd100k/images/track/mini'

        # self.base_folder =  '/fs/nexus-projects/DroneHuman/jxchen/data/04_ev/Control_SVD/Finetune_data/ev_svd_mini_rgb'
        self.base_folder = base_folder
        self.folders = sorted(os.listdir(self.base_folder))
        self.channels = 3
        self.width = width
        self.height = height
        self.sample_frames = sample_frames

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx= 0):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
   

        folder_idx= idx[0]
        frame_idx= idx[1]
        nparts= idx[2]

        skip= idx[3]

        # slecet a folder by index
        chosen_folder = self.folders[folder_idx]
        folder_path = os.path.join(self.base_folder, chosen_folder)


        # Get from rgb folder
        rgb_folder_path = os.path.join(folder_path, 'images')
        
        frames = os.listdir(rgb_folder_path)
        # Sort the frames by name
        frames.sort()

        
        # Ensure the selected folder has at least `sample_frames`` frames
        if len(frames) < self.sample_frames:
            raise ValueError(
                f"The selected folder '{chosen_folder}' contains fewer than `{self.sample_frames}` frames.")
        

        start_idx = frame_idx



        selected_frames = frames[start_idx:start_idx + self.sample_frames]


        while check_all_files_exist(selected_frames, rgb_folder_path) == False:
            start_idx = random.randint(0, len(frames) - self.sample_frames)
            selected_frames = frames[start_idx:start_idx + self.sample_frames]







        return {'rgb_folder_path': rgb_folder_path, 'start_idx': start_idx}
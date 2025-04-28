import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from .event_utils.lib.representations.voxel_grid import plot_voxel_grid, events_to_voxel, events_to_neg_pos_voxel
from .event_utils.lib.representations.image import events_to_image, events_to_timestamp_image
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms



from diffusers.utils import load_image

from PIL import Image


from.mStack_utils import events_to_voxel_grid, visualize_voxel_grid, voxel_norm, get_event_stacks


Num_BINS= 3



def count_image_files(folder_path):
    # Define the image file extensions you want to count
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

    # List all files in the folder and count how many are image files
    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith(image_extensions)]
    return len(image_files)


def check_event_sequence_npz(img, event_folder, idx, num_frames, scale=1):
    events_list= None



    event_npz_files= sorted(os.listdir(event_folder))

    event_timestamps= []

    for i in range(num_frames):
        cur_event_file= os.path.join(event_folder, event_npz_files[idx + i])

        # print('cur event file: ', cur_event_file)

        # print('cur event file: ', cur_event_file)

        events= load_event_npz(img, cur_event_file)

        if events is None:
            return False


        
    return True




def process_EventImage(events, h, w, num_bins,  reverse= False):
    """
    Main function
    """


    if len(events) == 0:
        voxel_img= np.zeros((num_bins, h, w))

        save_img_list= visualize_voxel_grid(voxel_img)

        return save_img_list
    
    events_stacks= get_event_stacks(events, num_bins, reverse= reverse)
    

    events_stack_voxels= []

    for i in range(num_bins):
        event_voxels_i= events_to_voxel_grid(events_stacks[i], 1, w, h)

        '''
        Normalize the voxel grid first!!!!
        '''

        event_voxels_i= voxel_norm(event_voxels_i)
        
        events_stack_voxels.append(event_voxels_i)

    event_voxels= np.stack(events_stack_voxels, axis=0)

    event_voxels= event_voxels.squeeze(1)


    # event_voxels= voxel_norm(event_voxels)


    save_img_list= visualize_voxel_grid(event_voxels)

    return save_img_list




def process_EventImage_bins(events, h, w, num_bins,  reverse= False):
    """
    Main function
    """


    if len(events) == 0:
        voxel_img= np.zeros((num_bins, h, w))

        save_img_list= visualize_voxel_grid(voxel_img)

        return save_img_list
    
    # events_stacks= get_event_stacks(events, num_bins, reverse= reverse)

    event_stacks= events_to_voxel_grid(events, num_bins, w, h)
    

    events_stack_voxels= []

    for i in range(num_bins):
        # event_voxels_i= events_to_voxel_grid(events_stacks[i], 1, w, h)
        event_voxels_i= event_stacks[i]

        '''
        Normalize the voxel grid first!!!!
        '''

        event_voxels_i= voxel_norm(event_voxels_i)
        
        events_stack_voxels.append(event_voxels_i)

    event_voxels= np.stack(events_stack_voxels, axis=0)

    # event_voxels= event_voxels.squeeze(1)


    # event_voxels= voxel_norm(event_voxels)


    save_img_list= visualize_voxel_grid(event_voxels)

    return save_img_list


def convert_stack_to_tensor(stack):
    img_tensor = torch.from_numpy(np.array(stack)).float()


    # single channel
    # img_tensor_1ch= img_tensor[:, :, 0] - img_tensor[:, :, 2]
    # 2 channels
    img_tensor_1ch= img_tensor

    # Normalize the image by scaling pixel values to [-1, 1]
    img_normalized = img_tensor_1ch / 127.0 - 1

    img_normalized = img_normalized.unsqueeze(0)


    return img_normalized



def check_event_sequence_npz(img, event_folder, idx, num_frames, scale=1):
    events_list= None

    event_npz_files= sorted(os.listdir(event_folder))

    event_timestamps= []

    # print('event folder: ', event_folder)
    # print('idx: ', idx)
    # print('num frames: ', num_frames)

    # print('len event npz files: ', len(event_npz_files))

    for i in range(num_frames):
        cur_event_file= os.path.join(event_folder, event_npz_files[idx + i])

        # print('cur event file: ', cur_event_file)

        # print('cur event file: ', cur_event_file)

        events= load_event_npz(img, cur_event_file)


        if events is None:
            return False


    return True



def load_event_sequence_npz(img, event_folder, idx, num_frames, scale=1, event_scale=32, timestamp_file= None):
    events_list= None


    event_npz_files= sorted(os.listdir(event_folder))

    event_timestamps= []

    


    for i in range(num_frames):



        if i == 0:
  
            continue
        else:
      
            # Real Video
            cur_event_file= os.path.join(event_folder, event_npz_files[idx + i - 1])
        


        print('cur event file: ', cur_event_file)


        events= load_event_npz(img, cur_event_file, event_scale= event_scale)



        event_timestamps.append(np.min(events[:, 0]))


        if i == num_frames - 1:
            event_timestamps.append(np.max(events[:, 0]))



        if events_list is None:
            events_list= events
        else:
            events_list= np.concatenate((events_list, events), axis=0)



    # sort events based on timestamp
    events_list= events_list[np.argsort(events_list[:, 0])]



    return events_list, event_timestamps



def get_event_idx_5frame(events, timestamps, frame_idx, skip= 3):
    
    if skip != -1:


        time_arr= timestamps


        event_idx_arr= []

        ts= events[:, 0]


        prev_idx= 0

        for i in range(len(time_arr)):
            cur_idx= binary_search_time(ts, time_arr[i], l=prev_idx)
            event_idx_arr.append(cur_idx)
            prev_idx= cur_idx
            
        
        return event_idx_arr
        
def get_event_idx_nframe(events, timestamps, frame_idx, skip= 3):


    time_arr= timestamps[frame_idx: frame_idx + skip + 2]


    event_idx_arr= []

    ts= events[:, 0]

    prev_idx= 0

    for i in range(len(time_arr)):
        cur_idx= binary_search_time(ts, time_arr[i], l=prev_idx)
        event_idx_arr.append(cur_idx)
        prev_idx= cur_idx
    
    return event_idx_arr
    




def get_event_idx_skip1(events, timestamps, frame_idx):

    num_frames= 14



    ts= events[:, 0]

    ts_min= np.min(ts)
    ts_max= np.max(ts)


    mid_ts= timestamps[frame_idx + 1]


    num_frames= num_frames - 1


    left_interpolated_num= num_frames // 2

    right_interpolated_num= num_frames // 2

    left_ts_arr= np.linspace(ts_min, mid_ts, left_interpolated_num + 2)

    right_ts_arr= np.linspace(mid_ts, ts_max, right_interpolated_num + 2)

    left_ts_arr= left_ts_arr[:-1]

    right_ts_arr= right_ts_arr

    interpolated_frame_ts= np.concatenate((left_ts_arr, right_ts_arr))

    event_idx_arr= []

    prev_idx= 0

    for i in range(len(interpolated_frame_ts)):
        cur_idx= binary_search_time(ts, interpolated_frame_ts[i], l=prev_idx)
        event_idx_arr.append(cur_idx)
        prev_idx= cur_idx


    return event_idx_arr




def get_event_idx_longGen(events, timestamps, frame_idx, gen_frames, delta_t):
    cur_timestamp= timestamps[frame_idx]

    ts= events[:, 0]

    # delta_t in ms

    gen_time_arr= [cur_timestamp + i * delta_t * 1000 for i in range(gen_frames)]


    event_idx_arr= []

    prev_idx= 0

    for i in range(len(gen_time_arr)):
        cur_idx= binary_search_time(ts, gen_time_arr[i], l=prev_idx)
        event_idx_arr.append(cur_idx)
        prev_idx= cur_idx

    
    return event_idx_arr
    



def load_event_npz(img, event_file, scale=1, event_scale=32):
    events_npz = read_events(event_file)




    ts= events_npz['timestamp'].astype(np.float64)

    # print('ts min, max: ', np.min(ts), np.max(ts))

    if len(ts) == 0:
        return None


    img= np.array(img)

   


    h, w= img.shape[:2]


    x= events_npz['x']

    x= x.astype(np.float32)

    # print('x min, max: ', np.min(x), np.max(x))

    x= x * scale

    x= x / event_scale
    # x= x.astype(np.int32)

    x= np.round(x).astype(np.float64)

    y= events_npz['y']
    y= y.astype(np.float64)

    # print('y min, max: ', np.min(y), np.max(y))

    y = y * scale

    y= y / event_scale
    # y= y.astype(np.int32)

    y= np.round(y).astype(np.float64)


    p= events_npz['polarity'].astype(np.float32)

    

    # make p -1 or 1
    if np.min(p) == 0:
        p= np.where(p == 0, -1, p)
    else:
        print('min p: ', np.min(p))


    num_pos= np.sum(p == 1)

    num_neg= np.sum(p == -1)

    

    if num_pos == 0 or num_neg == 0:
        return None
    
    num_ratio= num_pos / num_neg
    
    if num_ratio > 10 or num_ratio < 0.2:
        return None


    upsample_h= int(h * scale)
    upsample_w= int(w * scale)

    # print('upsample_h: ', upsample_h)
    # print('upsample_w: ', upsample_w)


    # print('x min, max: before ', np.min(x), np.max(x))
    # print('y min, max: before ', np.min(y), np.max(y))

    x= np.where(x < 0, 0, x)
    x= np.where(x >= upsample_w, upsample_w-1, x)

    y= np.where(y < 0, 0, y)
    y= np.where(y >= upsample_h, upsample_h-1, y)


    # print('x min, max: ', np.min(x), np.max(x))

    # print('y min, max: ', np.min(y), np.max(y))








    # x= np.where(x < 0, 0, x)
    # x= np.where(x >= w, w-1, x)

    # y= np.where(y < 0, 0, y)
    # y= np.where(y >= h, h-1, y)




    # upsample_h= int(h * scale)
    # upsample_w= int(w * scale)

    # x= np.where(x < 0, 0, x)
    # x= np.where(x >= upsample_w, upsample_w-1, x)

    # y= np.where(y < 0, 0, y)
    # y= np.where(y >= upsample_h, upsample_h-1, y)



    '''
    Put events in (t, x, y, p) format
    '''
    events= np.stack((ts, x, y, p), axis=1)


    return events


def binary_search_time(dset, x, l=None, r=None, side='left'):
    """
    Binary search for a timestamp in an HDF5 event file, without
    loading the entire file into RAM
    @param dset The data
    @param x The timestamp being searched for
    @param l Starting guess for the left side (0 if None is chosen)
    @param r Starting guess for the right side (-1 if None is chosen)
    @param side Which side to take final result for if exact match is not found
    @returns Index of nearest event to 'x'
    """
    l = 0 if l is None else l
    r = len(dset)-1 if r is None else r
    while l <= r:
        mid = l + (r - l)//2
        midval = dset[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r




def even_split_events(events, split=7):
    ts= events[:, 0]

    ts_min= np.min(ts)

    ts_max= np.max(ts)

    ts_range= ts_max - ts_min

    split_ts=(ts_range / split)

    split_indices= []

    prev_idx= 0

    for i in range(split):
        # split_indices.append(binary_search_time(ts,  ts_min + split_ts * i))
        cur_idx= binary_search_time(ts,  ts_min + split_ts * i, l=prev_idx)
        split_indices.append(cur_idx)
        prev_idx= cur_idx

    return np.split(events, split_indices, axis=0)
    


    



def even_split_events_skip1(left_events, right_events, split=13):
    
    left_split = split // 2
    right_split = split - left_split

    left_event_chunks = even_split_events(left_events, left_split)

    right_event_chunks = even_split_events(right_events, right_split)


    all_event_chunks = []

    for i in range(left_split):
        all_event_chunks.append(left_event_chunks[i])

    for i in range(right_split):
        all_event_chunks.append(right_event_chunks[i])

    all_event_chunks.append(right_event_chunks[-1])
    

    return all_event_chunks
    





def read_events(filename):
    """
    Read events from a file
    """
    events = np.load(filename)
    return events


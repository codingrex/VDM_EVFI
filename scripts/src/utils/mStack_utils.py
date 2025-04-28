import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from .event_utils.lib.representations.voxel_grid import plot_voxel_grid, events_to_voxel, events_to_neg_pos_voxel
from .event_utils.lib.representations.image import events_to_image, events_to_timestamp_image
import argparse
import torch
from PIL import Image



Num_BINS= False




def load_event_sequence_npz(img, event_folder, idx, num_frames):
    events_list= None

    event_npz_files= sorted(os.listdir(event_folder))

    for i in range(num_frames):
        cur_event_file= os.path.join(event_folder, event_npz_files[idx + i])

        # print('cur event file: ', cur_event_file)

        events= load_event_npz(img, cur_event_file)

        if events_list is None:
            events_list= events
        else:
            events_list= np.concatenate((events_list, events), axis=0)


    events_list= np.array(events_list)

    # sort events based on timestamp
    events_list= events_list[np.argsort(events_list[:, 0])]


    return events_list


def get_event_idx_5frame(events, timestamps, frame_idx, skip= 3):
    if skip == 1:
        mid_tx= timestamps[frame_idx + 1]
        left_tx= timestamps[frame_idx]
        left_mid_tx= left_tx + (mid_tx - left_tx) / 2
        right_tx= timestamps[frame_idx + 2]
        mid_right_tx= mid_tx + (right_tx - mid_tx) / 2

        time_arr= [left_tx, left_mid_tx, mid_tx, mid_right_tx, right_tx]

        event_idx_arr= []

        ts= events[:, 0]

        prev_idx= 0

        for i in range(len(time_arr)):
            cur_idx= binary_search_time(ts, time_arr[i], l=prev_idx)
            event_idx_arr.append(cur_idx)
            prev_idx= cur_idx
        
        return event_idx_arr
    
    if skip == 3:
        mid_tx= timestamps[frame_idx + 2]
        left_tx= timestamps[frame_idx]
        left_mid_tx= timestamps[frame_idx + 1]
        right_tx= timestamps[frame_idx + 4]
        mid_right_tx= timestamps[frame_idx + 3]

    

        time_arr= [left_tx, left_mid_tx, mid_tx, mid_right_tx, right_tx]

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
    









def load_event_npz(img, event_file, scale=1):
    events_npz = read_events(event_file)



    ts= events_npz['timestamp']


    img= np.array(img)

   


    h, w= img.shape[:2]


    x= events_npz['x']

    x= x.astype(np.float32)

    x= x / 32
    # x= x.astype(np.int32)

    x= np.round(x).astype(np.int32)

    y= events_npz['y']
    y= y.astype(np.float32)

    y= y / 32
    # y= y.astype(np.int32)

    y= np.round(y).astype(np.int32)


    p= events_npz['polarity']

    

    # make p -1 or 1
    if np.min(p) == 0:
        p= np.where(p == 0, -1, p)
    else:
        print('min p: ', np.min(p))
    



    x= np.where(x < 0, 0, x)
    x= np.where(x >= w, w-1, x)

    y= np.where(y < 0, 0, y)
    y= np.where(y >= h, h-1, y)

    x= x * scale
    y= y * scale



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



def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    if len(events) < 5:
        voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
        return voxel_grid
    

    events= events[np.argsort(events[:, 0])]
    

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp




    if deltaT == 0:
        deltaT = 1.0



    new_ts=  events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT


    


    # ts = events[:, 0]
    ts= new_ts
    xs = events[:, 1].astype(np.int64)
    ys = events[:, 2].astype(np.int64)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1



    tis = ts.astype(np.int64)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts




    valid_indices = tis < num_bins



    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))



    return voxel_grid


def voxel_norm(voxel):
    '''
    Normalize voxel grid to [-1, 1] based on 2% and 98% percentile for positive and negative values
    '''
    voxel_pos= voxel[voxel > 0]

    voxel_neg= voxel[voxel < 0]

    # print('number of positive pixels: ', len(voxel_pos))
    # print('number of negative pixels: ', len(voxel_neg))



    if len(voxel_pos) == 0:
        return voxel

    if len(voxel_neg) == 0:
        return voxel



    pos_2= np.percentile(voxel_pos, 2)

    pos_98= np.percentile(voxel_pos, 98)

    neg_2= np.percentile(voxel_neg, 2)

    neg_98= np.percentile(voxel_neg, 98)

    voxel[voxel > 0]= np.clip(voxel[voxel > 0], pos_2, pos_98)

    voxel[voxel < 0]= np.clip(voxel[voxel < 0], neg_2, neg_98)

    if pos_98 == pos_2:
        pos_98 = np.max(voxel_pos)
        pos_2 = np.min(voxel_pos)

    
    if neg_98 == neg_2:
        neg_98 = np.max(voxel_neg)
        neg_2 = np.min(voxel_neg)


    if pos_98 == pos_2:
        voxel[voxel > 0]= np.where(voxel[voxel > 0] > 0, 1, 0)
    else:
        # voxel[voxel > 0]= (voxel[voxel > 0] - pos_2) / (pos_98 - pos_2)
        voxel[voxel > 0]= voxel[voxel > 0] / pos_98

    if neg_98 == neg_2:
        voxel[voxel < 0]= np.where(voxel[voxel < 0] < 0, -1, 0)
    else:
        voxel[voxel < 0]=  -1 * (voxel[voxel < 0]) / (neg_98)


    return voxel



def visualize_voxel_grid(voxel_grid):
    """
    Visualize a voxel grid as RGB image.
    """
    
    # assert(voxel_grid.ndim == 3)
    # assert(voxel_grid.shape[0] == 3)

    voxel_grid = np.clip(voxel_grid, -1.0, 1.0)

    voxel_grid = (voxel_grid + 1.0) / 2

    voxel_grid= voxel_grid * 255
    voxel_grid= voxel_grid.astype(np.uint8)

    image_list= []

    for i in range(voxel_grid.shape[0]):
        img_i= voxel_grid[i, ...]

        img_save= Image.fromarray(img_i)

        image_list.append(img_save)


    return image_list



def get_event_stacks(events, num_stacks, reverse= False):
    
    # events_reversed= events[::-1]

    max_ts= np.max(events[:, 0])


    events[:, 0]= max_ts - events[:, 0]

    events_reversed= events[np.argsort(events[:, 0])]

    ts= events_reversed[:, 0]


    if reverse:
        # reverse polarity
        events_reversed[:, 3]= -1 * events_reversed[:, 3]

    event_stacks= []

    total_events= len(events_reversed)

    stack_idx_arr= np.linspace(0, num_stacks-1, num_stacks).astype(np.int32)

    for i in range(num_stacks):
        stack_idx= 2 ** stack_idx_arr[i]



        stack_idx= total_events // stack_idx - 1    

    
        event_stack= events_reversed[:stack_idx, ...]
   
    
        event_stacks.append(event_stack)

    
    
    return event_stacks



def center_crop(img, new_width, new_height):
    width, height = img.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return img.crop((left, top, right, bottom))
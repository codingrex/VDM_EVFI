import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.event_utils.lib.representations.voxel_grid import plot_voxel_grid, events_to_voxel, events_to_neg_pos_voxel
from utils.event_utils.lib.representations.image import events_to_image, events_to_timestamp_image
import h5py
import argparse
import torch
from PIL import Image


from utils.event_utils.lib.visualization.draw_event_stream import plot_between_frames


parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('--data_path', type=str, help='path to image file')
parser.add_argument('--save_path', type=str, help='path to save the output image')
args = parser.parse_args()





ROOT_DIR= args.data_path

ACC= 2 # in ms

Num_BINS= 1

NUM_STACKS= 10

SAVE_ROOT= args.save_path


TYPE= 'count'

TESTING = False

NPY= False





def get_event_stacks(events, num_stacks):
    
    events_reversed= events[::-1]

    event_stacks= []

    total_events= len(events_reversed)

    stack_idx_arr= np.linspace(0, num_stacks-1, num_stacks).astype(np.int32)

    for i in range(num_stacks):
        stack_idx= 2 ** stack_idx_arr[i]

        # print('stack_idx: ', stack_idx)

        stack_idx= total_events // stack_idx
    
        event_stack= events_reversed[:stack_idx, ...]
    
        event_stacks.append(event_stack)
    
    return event_stacks





def get_events_for_frame(events, frame_idx, timestamps):
    """
    Get events for a frame
    """

    cur_ts= timestamps[frame_idx]
    next_ts= timestamps[frame_idx + 1]

    ts= events[:, 0]

    idx_cur= binary_search_time(ts, cur_ts)
    idx_next= binary_search_time(ts, next_ts, l= idx_cur)

    event_slice= events[idx_cur: idx_next, ...]

    event_slice= event_slice[np.argsort(event_slice[:, 0])]

    return event_slice


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


    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0


    # events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT

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
        voxel[voxel > 0]= (voxel[voxel > 0] - pos_2) / (pos_98 - pos_2)

    if neg_98 == neg_2:
        voxel[voxel < 0]= np.where(voxel[voxel < 0] < 0, -1, 0)
    else:

        voxel[voxel < 0]=  -1 * (voxel[voxel < 0] - neg_2) / (neg_98 - neg_2)


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



def generate_event_image(x, y, ts, p, h, w, acc_time, type):
    if len(ts) == 0:
        event_image= np.zeros((h, w, 3))
        return event_image

    if type == 'count':
        events= np.stack((x, y, ts, p), axis=1)
        events= accumulate_events(events, ts, acc_time= acc_time)


        event_image= np.zeros((h, w, 3))
        event_image= render(events, event_image)
    elif type == 'timestamp':
        event_cnt_image= events_to_image(x, y, p, sensor_size=(h, w), meanval= True)

       
        event_timestep_img= events_to_timestamp_image(x, y, ts, p , sensor_size=(h, w),  padding = False)

        # stack channel wise
    
        event_image= np.stack((event_cnt_image, event_timestep_img[0], event_timestep_img[1]), axis=2)

    
    return event_image


def read_events(filename):
    """
    Read events from a file
    """
    events = np.load(filename)
    return events




def accumulate_events(events, ts, acc_time= None):
    """
    Show events
    """

    if acc_time is None:
        return events

    idx= binary_search_time(ts, ts[0] + acc_time * 1000)

    return events[:idx, ...]
    


def render(events, image):
    x,y,p = events[:,0].astype(np.int32), events[:,1].astype(np.int32), events[:,3]>0
    height, width = image.shape[:2]
    x = np.clip(x, 0, width-1)
    y = np.clip(y, 0, height-1)
    image[y[p],x[p],2] = 255
    image[y[p==0],x[p==0],0] = 255
    return image



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


def process_folder_EventImage(root_dir, idx, COUNT, dir_name= None, total_events= None):
    """
    Main function
    """

  
    event_folder = os.path.join(root_dir, 'events')
    img_folder = os.path.join(root_dir, 'images')

    folder_name= root_dir.split('/')[-1]

    if idx == 0:
        event_file = os.path.join(event_folder, '%06d.npz' % (idx))
    else:
        event_file = os.path.join(event_folder, '%06d.npz' % (idx - 1))


    events_npz = read_events(event_file)



    ts= events_npz['timestamp']


    if len(ts) == 0:
        return COUNT + 1


    



    img_file = os.path.join(img_folder, '%06d.png' % (idx))

    if idx == total_events - 1:
        last_img_file= os.path.join(img_folder, '%06d.png' % (idx + 1))
        last_img= cv2.imread(last_img_file)

    # img_file_t1= os.path.join(img_folder, '%06d.png' % (idx + 1))

    img= cv2.imread(img_file)

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


    p= events_npz['polarity'].astype(np.int32)
    



    # make p -1 or 1
    if np.min(p) == 0:
        p= np.where(p == 0, -1, p)
    else:
        print('min p: ', np.min(p))
    

    num_pos= np.sum(p == 1)

    num_neg= np.sum(p == -1)

    num_ratio= num_pos / num_neg

    if num_pos == 0 or num_neg == 0:
        print('num_pos or num_neg is 0')
        return COUNT + 1
    
    if num_ratio > 10 or num_ratio < 0.2:
        print('num_ratio: ', num_ratio)
        return COUNT + 1

    

    x= np.where(x < 0, 0, x)
    x= np.where(x >= w, w-1, x)

    y= np.where(y < 0, 0, y)
    y= np.where(y >= h, h-1, y)



    '''
    Put events in (t, x, y, p) format
    '''
    events= np.stack((ts, x, y, p), axis=1)


    events_stacks= get_event_stacks(events, NUM_STACKS)



    events_stack_voxels= []

    for i in range(NUM_STACKS):
        event_voxels_i= events_to_voxel_grid(events_stacks[i], Num_BINS, w, h)


        '''
        Normalize per channel!!!!
        '''

        event_voxels_i= voxel_norm(event_voxels_i)
        
        events_stack_voxels.append(event_voxels_i)

    event_voxels= np.stack(events_stack_voxels, axis=0)

    event_voxels= event_voxels.squeeze(1)



    if event_voxels is None:
        return COUNT + 1

    save_img_list= visualize_voxel_grid(event_voxels)


    if TESTING:
        if len(ts) == 0:
            event_image= np.zeros((h, w, 3))
        else:


            event_image= generate_event_image(x, y, ts, p, h, w, acc_time= ACC, type= TYPE)


    save_dir= os.path.join(SAVE_ROOT, dir_name)

    save_folder= os.path.join(save_dir, folder_name)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    save_folder_img_t0= os.path.join(save_folder, 'image')
    # save_folder_img_t1= os.path.join(save_folder, 't1')
    save_folder_ev= os.path.join(save_folder, 'event')



    if TESTING:
        save_folder_test= os.path.join(save_folder, 'test')
        if not os.path.exists(save_folder_test):
            os.makedirs(save_folder_test)


    if not os.path.exists(save_folder_img_t0):
        os.makedirs(save_folder_img_t0)

    # if not os.path.exists(save_folder_img_t1):
    #     os.makedirs(save_folder_img_t1)

    if not os.path.exists(save_folder_ev):
        os.makedirs(save_folder_ev)
    


    
    save_img_t0= os.path.join(save_folder_img_t0, '%06d.png' % (COUNT))
    # save_img_t1= os.path.join(save_folder_img_t1, '%06d.png' % (COUNT))

    if TESTING:
        save_event_img_file= os.path.join(save_folder_test, '%06d.png' % (COUNT))

    if idx == total_events - 1:
        save_img_t1= os.path.join(save_folder_img_t0, '%06d.png' % (COUNT + 1))


    if NPY == False:
        save_event_voxel_file= os.path.join(save_folder_ev, '%06d.png' % (COUNT))
    else:
        save_event_voxel_file= os.path.join(save_folder_ev, '%06d.npy' % (COUNT))

    cv2.imwrite(save_img_t0, img)

    if idx == total_events - 1:
        cv2.imwrite(save_img_t1, last_img)

    if NPY == False:
        # cv2.imwrite(save_event_voxel_file, voxel_img)
        for i in range(len(save_img_list)):
            save_img_list[i].save(os.path.join(save_folder_ev, '%06d_%02d.png' % (COUNT, i)))
    else:
        np.save(save_event_voxel_file, event_voxels)


    if TESTING:
        cv2.imwrite(save_event_img_file, event_image * 255)
    

    

    # cv2.imwrite(save_img_t1, img_t1)


    COUNT = COUNT + 1

    return COUNT







def main():

    save_dict= {}
    save_dict['t0']=[]
    save_dict['t1']= []
    save_dict['event']= [] 

    all_dir= sorted(os.listdir(ROOT_DIR))

    NUM_IMGS_IN_DIR= []


    for k in range(len(all_dir)):
        all_folders= os.listdir(os.path.join(ROOT_DIR, all_dir[k]))
        dir_name= os.path.join(ROOT_DIR, all_dir[k])
        for i in tqdm(range(len(all_folders))):
            folder= all_folders[i]
            folder_contents= os.listdir(os.path.join(dir_name, folder))
            if len(folder_contents) == 0:
                continue
            folder= os.path.join(dir_name, folder)

            event_folder = os.path.join(folder, 'events')
            if not os.path.exists(event_folder):
                continue
            COUNT= 0

            print('Processing folder: ', folder)

            
            for j in range(len(os.listdir(event_folder)) + 1):

                if '3_TRAINING' not in folder:
                    continue



                total_events= len(os.listdir(event_folder))

                COUNT= process_folder_EventImage(folder, j, COUNT, dir_name= all_dir[k], total_events=total_events)
            

            num_imgs= len(os.listdir(os.path.join(folder, 'events')))
            NUM_IMGS_IN_DIR.append(num_imgs)

        
        # save total number of images in each directory
        sum_imgs= np.sum(NUM_IMGS_IN_DIR)

        if 'TRAINING' in all_dir[k]:
            save_training_txt= os.path.join(SAVE_ROOT, 'TRAINING.txt')

            with open(save_training_txt, 'w') as f:
                f.write(str(sum_imgs))


    


if __name__ == '__main__':
    main()



import numpy as np
import os

import matplotlib.pyplot as plt

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to calculate the metrics"
    )
 
    parser.add_argument(
        "--test_metric_folder",
        type=str,
        default=None,
        required=True,
        help="Path to the test metric folder",
    )

    args = parser.parse_args()
 

    return args


def summary_metirc(dir, tone):
    metrics_folder= sorted(os.listdir(dir))

    for metric in metrics_folder:
        metric_path= os.path.join(dir, metric)
        seqs= sorted(os.listdir(metric_path))

        folder_name_list= []

        all_data= []

        each_seq_data= []

        
        for seq in seqs:
    
            seq_path= os.path.join(metric_path, seq)
            files= sorted(os.listdir(seq_path))

            folder_name_list.append(seq)

            seq_data= []

            for file in files:
                file_path= os.path.join(seq_path, file)
                with open(file_path, 'r') as f:
                    data= f.readlines()
                    data= [float(i.strip()) for i in data]
                    if len(data) == 0:
                        continue
                    data= np.array(data)[tone]
                    seq_data.append(data)
            
            seq_mean= np.mean(seq_data)
            each_seq_data.append(seq_mean)


        each_seq_data= np.array(each_seq_data)


        each_folder_mean= np.mean(each_seq_data)

        print(f'{metric} mean: {each_folder_mean}')


        # print('====================END====================')
    print('folder_name_list:', folder_name_list)


            


if __name__ == '__main__':

    args = parse_args()

    TEST_METRIC_FOLDER= args.test_metric_folder

    t= 0
   
    summary_metirc(TEST_METRIC_FOLDER, t)


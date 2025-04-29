# Repurposing Pre-trained Video Diffusion Models for Event-based Video Interpolation (CVPR 2025)
**Official repository for the CVPR 2025 paper, "Repurposing Pre-trained Video Diffusion Models for Event-based Video Interpolation"**

\[[Website](https://vdm-evfi.github.io/)\] 
\[[Paper](https://arxiv.org/abs/2412.07761)\] 



![teaser.gif](./assets/HTML_Teaser.gif)


## Installation

Create an conda environment

```
 conda create --name VDM_EVFI python=3.9
 conda activate VDM_EVFI
 cd VDM_EVFI
```


1. Torch with CUDA 12.4 (Required for XFormer)
```
pip install torch torchvision torchaudio
```

2. Installing Diffusers (Custom Version)

    Please be sure to build from source our custom version of diffusers library.
```
cd diffusers/
pip install accelerate
pip install -e ".[torch]"
```
3. Installing the rest of packages
```
cd ..
pip install -r requirements.txt
```

## Model Checkpoints & Example Data
1. Model Checkpoints: 

    Trained on the BS-ERGB dataset only. The 5frames.zip files are used for metric calculation and can also be used to generate videos. The 13frames.zip files are fine-tuned for inserting 11 frames for video interpolation.

    * Google Drive: \[[Models (Google Drive)](https://drive.google.com/drive/folders/1qvIamYuOFxIVI6-i3N4umJ_MHHKnYxAJ?usp=drive_link)\] 
    * Hugging Face: \[[5 Frames Model](https://huggingface.co/jxAIbot/VDM_EVFI)\], \[[13 Frames Model](https://huggingface.co/jxAIbot/VDM_EVFI_VIDEO)\] 

2. Example Data:

    The example data and file structure are in **example.zip**
    * Google Drive: \[[Example Data (Google Drive)](https://drive.google.com/drive/folders/1YbtXSGH2-x_Kuce4EGhGyeByVzxOPAfb?usp=drive_link)\] 


## Running the Inference Code
We provide a script to generate interpolated videos with 11 inserted frames. You can choose to use 5frames.zip (for the same setup as videos generated in the website) or 13frames.zip checkpoints. 13frames.zip checkpoints are slightly better. 
```
cd scripts/
sh valid_video.sh
```
Inside the valid_video.sh, there are some important configurations worth attention. 
```
--pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid" \
--output_dir="PATH_TO_OUTPUT_DIR" \
--test_data_path="PATH_TO_TEST_DATA" \
--event_scale=32 for BS-ERGB, 1 for all others \
--controlnet_model_name_or_path="PATH_TO_CONTROLNET_CHECKPOINTS" \
--eval_folder_start=START_FOLDER_INDEX \
--eval_folder_end=END_FOLDER_INDEX, -1 to the end folder \
--num_frames=NUM_INSERTED_FRAMES+2 \
--width=IMAGE_WIDTH \
--height=IMAGE_HEIGHT \
 --rescale_factor=UPSAMPLING_FACTOR \
```


## Calculating Metrics
### Reproducing Metrics in the Paper:
1. Running the inference code with metric calculcation
* As noted in the paper, we apply VAE encoding/decoding to both the model outputs and ground truths to eliminate non-essential effects, such as output tonemapping and noise differences, introduced by the frozen VAE from the pre-trained Video Diffusion Models during metric calculation.
* Use checkpoint from 5frames.zip
```
cd scripts/
sh valid.sh
```
Inside the valid.sh, there are some important configurations worth attention. 
```
--pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid" \
--output_dir="PATH_TO_OUTPUT_DIR" \
--test_data_path="PATH_TO_TEST_DATA" \
--event_scale=32 for BS-ERGB, 1 for all others \
--controlnet_model_name_or_path="PATH_TO_CONTROLNET_CHECKPOINTS" \
--eval_folder_start=START_FOLDER_INDEX \
--eval_folder_end=END_FOLDER_INDEX, -1 to the end folder \
--num_frames=NUM_INSERTED_FRAMES+2 \
--width=IMAGE_WIDTH \
--height=IMAGE_HEIGHT \
 --rescale_factor=UPSAMPLING_FACTOR \
```

2. Caluculating Metrics

    First change path in **cal_metrics.sh**
    ```
    python cal_metric.py \
        --test_metric_folder PATH_TO_OUTPUT_DIR_rescale_factor_2.0_overlapping_ratio_0.1_t0_0_M_2_s_churn_0.5/metrics 
    ```
    Then
    ```
    sh cal_metrics.sh
    ```



## Dataset Format
We assume the same dataset format as BS-ERGB dataset as following.
```
├── Clear-Motion
    ├── sequence_0001
        ├── images
            ├── ...
        ├── events
            ├── ...
    ├── sequence_0002
    ....
```



## To-do Plans
We plan to do following soon:
* The release of Training Code 
* The release of Clear-Motion test sequences


## Contact
If you have any questions or are interested in our research, please feel free to contact Jingxi Chen: ianchen@umd.edu

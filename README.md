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

#### Inference on HQF dataset:
Here is an example of running the model on a dataset other than BS-ERGB, such as HQF (which contains grayscale frames).

```
cd scripts/
sh valid_HQF.sh
```
The key argument changes are:
```
--event_scale=1 \ Downscaling factor of x ,y for events
--width=240 \
--height=180 \
--rescale_factor=3 \
```

2. Caluculating Metrics
* In our paper, to handle input frames that are not divisible by 8, we use \[["pad_to_multiple_of_8_pil"](https://github.com/codingrex/VDM_EVFI/blob/main/scripts/src/pipelines/pipeline_stable_video_diffusion_FullControlnet_MStack_timereversal.py#L56)\] to pad them accordingly and then resize to upsampled size.  However, in the released code, for simplicity and ease of use in other projects, we resize the input to the nearest multiple of 8 and upsampled size. This may cause slight differences from the metric numbers reported in the paper.


    First change path in **cal_metrics.sh**
    ```
    python cal_metric.py \
        --test_metric_folder PATH_TO_OUTPUT_DIR_rescale_factor_2.0_overlapping_ratio_0.1_t0_0_M_2_s_churn_0.5/metrics 
    ```
    Then
    ```
    sh cal_metrics.sh
    ```


## Training
As explained in our paper, our model is trained solely on the BS-ERGB dataset and tested on all other datasets without any finetuning (i.e., zero-shot evaluation).

1. To speed up training, we pre-processed the BS-ERGB dataset using the following code.
```
sh process_bsergb.sh
```

Set '--data_path' to the BS-ERGB directory and '--save_path' to the location where you want to save the processed outputs.
```
--data_path='/datasets/bs_ergb/' \
--save_path='/99_BSERGB_MStack/' \
```

2. Run the training code. Our setup assumes an effective batch size of 64, achieved using 4 A6000 Ada GPUs (50GB) with a per-GPU batch size of 1 and gradient accumulation of 16. You can adjust these settings based on your hardware configuration. Set --train_data_path and --valid_path1 to the previously saved location of the processed BS-ERGB dataset, and --output_dir to the directory where you want to save the trained model.

```
accelerate launch --multi_gpu --num_processes 4 train.py \
    --pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid" \
    --output_dir="/99_Release/train_bsergb" \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=16 \
    --num_train_epochs=600 \
    --width=512 \
    --height=320 \
    --checkpointing_steps=500 --checkpoints_total_limit=1 \
    --learning_rate=5e-5 --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=200 \
    --num_frames=5 \
    --num_workers=4 \
    --enable_xformers_memory_efficient_attention \
    --resume_from_checkpoint="latest" \
    --train_data_path="99_BSERGB_MStack/" \
    --valid_path1='99_BSERGB_MStack/3_TRAINING/horse_04/image/' \
    --valid_path1_idx=23
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
* The release of Training Code (✅ Done)
* The release of Clear-Motion test sequences


## Contact
If you have any questions or are interested in our research, please feel free to contact Jingxi Chen: ianchen@umd.edu

## Citation
If you find our code or paper useful for your projects or research, please consider citing our paper:
```
@inproceedings{chen2025repurposing,
  title={Repurposing pre-trained video diffusion models for event-based video interpolation},
  author={Chen, Jingxi and Feng, Brandon Y and Cai, Haoming and Wang, Tianfu and Burner, Levi and Yuan, Dehao and Fermuller, Cornelia and Metzler, Christopher A and Aloimonos, Yiannis},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={12456--12466},
  year={2025}
}
```

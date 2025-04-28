# CUDA_VISIBLE_DEVICES=0

accelerate launch valid.py \
    --pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid" \
    --output_dir="real_test_13/13frames" \
    --test_data_path="example_data/clear_motion_sample" \
    --event_scale=1 \
    --num_frames=13 \
    --video_mode \
    --rescale_factor=2 \
    --controlnet_model_name_or_path="13frames/checkpoint-7500/controlnet" \
    --eval_folder_start=0 \
    --eval_folder_end=-1 \
    --width=940 \
    --height=720 \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=4 \
    --num_train_epochs=50 \
    --checkpointing_steps=20 \
    --checkpoints_total_limit=1 \
    --learning_rate=3e-5 --lr_warmup_steps=0 \
    --seed=123 \
    --validation_steps=20 \
    --num_workers=8 \
    --num_inference_steps=25 \
    --decode_chunk_size=2 \
    --enable_xformers_memory_efficient_attention \
    --mixed_precision="fp16" \
    --overlapping_ratio=0.1 \
    --t0=0 \
    --M=2 \
    --s_churn=0.5 \
    
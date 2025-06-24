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


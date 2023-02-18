set -e 
mkdir -p $3
python finetune/merge_dd_tags_to_metadata.py $1 $3/meta_cap_dd.json
python finetune/clean_captions_and_tags.py $1 $3/meta_cap_dd.json $3/meta_clean.json
python finetune/prepare_buckets_latents.py \
   $1 $3/meta_clean.json $3/meta_lat.json \
   runwayml/stable-diffusion-v1-5 \
   --batch_size=14 \
   --max_resolution=768,768 \
   --mixed_precision=fp16 \
   --min_bucket_reso=64 \
   --max_bucket_reso=1024 \
   --num_gpus 8 \
   --max_data_loader_n_workers=4 \
nohup accelerate launch --num_cpu_threads_per_process 8 fine_tune.py     \
   --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5  \
   --train_data_dir=$1  \
   --output_dir=$2   \
   --shuffle_caption\
   --train_batch_size=40 \
   --in_json=$3/meta_lat.json \
   --learning_rate=5e-6 \
   --max_train_epochs=8 \
   --mixed_precision=fp16    \
   --gradient_checkpointing \
   --use_8bit_adam \
   --save_every_n_epochs=1 \
   --xformers \
   --train_text_encoder \
   --lr_scheduler=cosine_with_restarts > nohup3.out 2>&1 &
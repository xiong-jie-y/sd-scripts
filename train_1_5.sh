nohup accelerate launch --num_cpu_threads_per_process 8 fine_tune.py     \
   --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5  \
   --train_data_dir=$1  \
   --output_dir=$2   \
   --shuffle_caption\
   --train_batch_size=28 \
   --in_json=$3/meta_lat.json \
   --learning_rate=5e-6 \
   --max_train_epochs=2 \
   --mixed_precision=fp16    \
   --gradient_checkpointing \
   --use_8bit_adam \
   --save_every_n_epochs=1 \
   --xformers \
   --lr_scheduler=cosine_with_restarts &
set -e 
mkdir -p $3
python finetune/merge_dd_tags_to_metadata.py $1 $3/meta_cap_dd.json
python finetune/clean_captions_and_tags.py $1 $3/meta_cap_dd.json $3/meta_clean.json
nohup python finetune/prepare_buckets_latents.py \
   $1 $3/meta_clean.json $3/meta_lat.json \
   stabilityai/stable-diffusion-2-1 \
   --batch_size=8 \
   --max_resolution=896,896 \
   --mixed_precision=fp16 \
   --min_bucket_reso=64 \
   --max_bucket_reso=1536 \
   --v2 > nohup_896.out 2>&1 &
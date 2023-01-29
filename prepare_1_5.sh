set -e 
mkdir -p $3
python finetune/merge_dd_tags_to_metadata.py $1 $3/meta_cap_dd.json
python finetune/clean_captions_and_tags.py $1 $3/meta_cap_dd.json $3/meta_clean.json
nohup python finetune/prepare_buckets_latents.py \
   $1 $3/meta_clean.json $3/meta_lat.json \
   runwayml/stable-diffusion-v1-5 \
   --batch_size=12 \
   --max_resolution=768,768 \
   --mixed_precision=fp16 \
   --min_bucket_reso=64 \
   --max_bucket_reso=1536 &
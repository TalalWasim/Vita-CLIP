exp_dir=train_output/k400_supervised

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 2 \
  ./training/train.py \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --num_steps 30000 \
    --save_freq 5000 \
    --eval_freq 5000 \
    --batch_size 256 \
    --batch_split 8 \
    --backbone_path "./pretrained/clip_pretrained.pth" \
    --train_list_path ../../repos/datasets/kinetics-dataset/k400_resized/annotations/train.csv \
    --val_list_path ../../repos/datasets/kinetics-dataset/k400_resized/annotations/val.csv \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 6 \
    --num_frames 8 \
    --use_text_prompt_learning \
    --text_num_prompts 8 \
    --use_text_prompt_CSC \
    --use_summary_token \
    --use_local_prompts \
    --use_global_prompts \
    --num_global_prompts 8 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"
lr=3e-4
SAVE_DIR=/home1/sxy/transfer-learning-conv-ai/runs/personachat/$lr
CUDA_VISIBLE_DEVICES=2 python train.py \
--gradient_accumulation_steps=16 \
--lm_coef=2.0 \
--max_history=2 --n_epochs=100 --num_candidates=4 --personality_permutations=2 --train_batch_size=4 --valid_batch_size=4 \
--lr=$lr \
--save_dir $SAVE_DIR \

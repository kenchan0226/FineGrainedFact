#! /bin/bash
# Fine-tune FactCC model

# UPDATE PATHS BEFORE RUNNING SCRIPT
export CUDA_VISIBLE_DEVICES=0
export CODE_PATH=/shared/nas/data/users/hpchan/projects/fineGrainedFact/modeling # absolute path to modeling directory
export DATA_PATH=/shared/nas/data/users/hpchan/projects/fineGrainedFact/data/aggrefact-deduplicated-final # absolute path to data directory
export OUTPUT_PATH=/shared/nas/data/users/hpchan/projects/fineGrainedFact/model_ckpts/aggrefact-multi-label-claim-attn-mean-word-layer-attn-finetune-adapter-four-label-dedup # absolute path to store model checkpoint

export TASK_NAME=aggrefact_multi_label_claim_attn_annotated_with_entire_sent_four_label
export MODEL_NAME=bert-base-uncased
export N_CLAIM_ATTN_HEADS=16

export SEED=$RANDOM
python3 $CODE_PATH/run_new.py \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --do_lower_case \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 12 \
  --learning_rate 1e-5 \
  --num_train_epochs 40 \
  --data_dir $DATA_PATH \
  --model_type bertmultilabelclaimattnmeanwordlayerattnadapter \
  --model_name_or_path $MODEL_NAME \
  --output_dir $OUTPUT_PATH/$MODEL_NAME-$TASK_NAME-finetune-$RANDOM-$N_CLAIM_ATTN_HEADS-$SEED/ \
  --gradient_accumulation_steps 2 \
  --seed $SEED \
  --n_claim_attn_heads $N_CLAIM_ATTN_HEADS
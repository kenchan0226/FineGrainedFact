#! /bin/bash
# Evaluate FactCC model

# UPDATE PATHS BEFORE RUNNING SCRIPT
export CUDA_VISIBLE_DEVICES=0
export CODE_PATH=/shared/nas/data/users/hpchan/projects/fineGrainedFact/modeling # absolute path to modeling directory
export DATA_PATH=/shared/nas/data/users/hpchan/projects/fineGrainedFact/data/fever2 # absolute path to the directory that contains the data-dev.jsonl of fever2
export CKPT_PATH=/shared/nas/data/users/hpchan/projects/fineGrainedFact/model_ckpts/aggrefact-multi-label-claim-attn-mean-word-layer-attn-finetune-adapter-four-label-dedup/bert-base-uncased-aggrefact_multi_label_claim_attn_annotated_with_entire_sent_four_label-finetune-11803-16-2000/checkpoint-37
export TASK_NAME=aggrefact_multi_label_claim_attn_annotated_with_entire_sent_four_label
export N_CLAIM_ATTN_HEADS=16

#mkdir $CKPT_PATH/fever2
EVAL_OUTPUT_DIR=$CKPT_PATH/fever2
python3 $CODE_PATH/run_new.py \
  --task_name $TASK_NAME \
  --do_eval \
  --do_lower_case \
  --overwrite_cache \
  --max_seq_length 512 \
  --per_gpu_train_batch_size 12 \
  --model_type bertmultilabelclaimattnmeanwordlayerattnadapter \
  --model_name_or_path $CKPT_PATH \
  --data_dir $DATA_PATH \
  --output_dir $CKPT_PATH \
  --eval_output_dir $EVAL_OUTPUT_DIR \
  --n_claim_attn_heads $N_CLAIM_ATTN_HEADS \
  --export_output
python3 compute_interpretation_scores.py \
--model_output_file $EVAL_OUTPUT_DIR/model_outputs.jsonl \
--src_jsonl_file $DATA_PATH/data-dev.jsonl

#! /bin/bash
# Evaluate FactCC model

# UPDATE PATHS BEFORE RUNNING SCRIPT
export CUDA_VISIBLE_DEVICES=0
export CODE_PATH=/shared/nas/data/users/hpchan/projects/fineGrainedFact/modeling # absolute path to modeling directory
export DATA_PATH=/shared/nas/data/users/hpchan/projects/fineGrainedFact/data/aggrefact-deduplicated-final-test # absolute path to data directory
export TASK_NAME=aggrefact_multi_label_claim_attn_annotated_with_entire_sent_four_label
export N_CLAIM_ATTN_HEADS=16
export CKPT_PATH=/shared/nas/data/users/hpchan/projects/fineGrainedFact/model_ckpts/aggrefact-multi-label-claim-attn-mil-cos-sim-finetune-adapter-four-label/bert-base-uncased-aggrefact_multi_label_claim_attn_annotated_with_entire_sent_four_label-finetune-7643/checkpoint-29

# run inference
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
  --n_claim_attn_heads $N_CLAIM_ATTN_HEADS \
  --export_output
# compute evaluation scores
python3 evaluate_multi_label_classification_scores.py \
--model_output_file $CKPT_PATH/model_outputs.jsonl \
--src_jsonl_file $DATA_PATH/data-dev.jsonl \
--is_entire
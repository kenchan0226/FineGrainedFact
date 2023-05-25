# Copyright (c) 2020, Salesforce.com, Inc.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import shutil

import wandb
import numpy as np
import torch

from model import BertPointer, BertMultiLabelClassifier, BertMultiLabelAdapterClassifier, LongformerMultiLabelClassifier, LongformerMultiLabelSeqLabelClassifier, BertMultiLabelSeqLabelClassifier, BertSRLAttnMultiLabelClassifier, BertSRLAttnFCMultiLabelClassifier, BertClaimAttnMultiLabelAdapterClassifier, BertClaimAttnMilMultiLabelAdapterClassifier, BertClaimAttnMultiLabelClassifier, BertClaimAttnMilCosSimMultiLabelAdapterClassifier, BertClaimAttnMilMultiLabelClassifier, BertClaimAttnMilCosSimMultiLabelClassifier, BertClaimAttnSepMilCosSimMultiLabelAdapterClassifier, BertClaimAttnMeanMilCosSimMultiLabelAdapterClassifier, BertClaimAttnMaxMilCosSimMultiLabelAdapterClassifier, BertClaimAttnMeanMilMultiLabelAdapterClassifier, BertBinaryLabelSeqLabelClassifier, BertClaimAttnMilCosSimMultiLabelAdapterWordAttnClassifier, BertClaimAttnMilCosSimMultiLabelAdapterWordAttnSimpleClassifier, BertClaimAttnMeanCosSimMultiLabelAdapterClassifier, BertClaimAttnMeanCosSimWordAttnMultiLabelAdapterClassifier, BertClaimAttnMeanWordAttnMultiLabelAdapterClassifier, BertClaimAttnMeanWordLayerAttnMultiLabelAdapterClassifier, BertClaimAttnMeanWordLayerAttnMultiLabelClassifier, BertClaimAttnMeanWordLayerAttnNoSelectMultiLabelClassifier, BertClaimAttnMeanWordLayerAttnNoSelectMultiLabelAdapterClassifier, BertClaimAttnMeanWordAttnMultiLabelClassifier, BertClaimAttnWeightCosSimWordAttnMultiLabelAdapterClassifier, BertClaimAttnMeanCosSimWordLayerAttnMultiLabelAdapterClassifier, BertClaimAttnWeightCosSimWordLayerAttnMultiLabelAdapterClassifier, BertClaimAttnMeanMultiLabelAdapterClassifier
from utils import (compute_metrics, compute_seq_label_metrics, convert_examples_to_features, convert_binary_seq_label_examples_to_features, convert_seq_label_examples_to_features, convert_attn_examples_to_features, convert_claim_attn_examples_to_features, output_modes, processors, FactCCGeneratedAttnDataset, classification_metric_all, multi_label_acc_auc_f1, multi_label_bacc_auc_f1, multi_label_bacc_auc_f1_macro)

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer, BertTokenizerFast)
from transformers import LongformerConfig, LongformerTokenizer, LongformerTokenizerFast
#from transformers import AdamW, WarmupLinearSchedule
from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import classification_report

import json

logger = logging.getLogger(__name__)
wandb.init(project="entailment-metric")

#ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

MODEL_CLASSES = {
    'pbert': (BertConfig, BertPointer, BertTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'bertmultilabel': (BertConfig, BertMultiLabelClassifier, BertTokenizer),
    'longformermultilabel': (LongformerConfig, LongformerMultiLabelClassifier, LongformerTokenizer),
    'longformermultilabelseqlabel': (LongformerConfig, LongformerMultiLabelSeqLabelClassifier, LongformerTokenizerFast),
    'bertmultilabelseqlabel': (BertConfig, BertMultiLabelSeqLabelClassifier, BertTokenizerFast),
    'bertmultilabelsrlattn': (BertConfig, BertSRLAttnMultiLabelClassifier, BertTokenizerFast),
    'bertmultilabelsrlattnfc': (BertConfig, BertSRLAttnFCMultiLabelClassifier, BertTokenizerFast),
    'bertmultilabeladapter': (BertConfig, BertMultiLabelAdapterClassifier, BertTokenizer),
    'bertmultilabelclaimattnadapter': (BertConfig, BertClaimAttnMultiLabelAdapterClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnmiladapter': (BertConfig, BertClaimAttnMilMultiLabelAdapterClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnmil': (BertConfig, BertClaimAttnMilMultiLabelClassifier, BertTokenizerFast),
    'bertmultilabelclaimattn': (BertConfig, BertClaimAttnMultiLabelClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnmilcossimadapter': (BertConfig, BertClaimAttnMilCosSimMultiLabelAdapterClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnmeancossimadapter': (BertConfig, BertClaimAttnMeanCosSimMultiLabelAdapterClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnmeancossimwordattnadapter': (BertConfig, BertClaimAttnMeanCosSimWordAttnMultiLabelAdapterClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnmeanwordattnadapter': (BertConfig, BertClaimAttnMeanWordAttnMultiLabelAdapterClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnmeanwordlayerattnadapter': (BertConfig, BertClaimAttnMeanWordLayerAttnMultiLabelAdapterClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnmeanwordlayerattn': (BertConfig, BertClaimAttnMeanWordLayerAttnMultiLabelClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnmeanwordlayerattnnoselect': (BertConfig, BertClaimAttnMeanWordLayerAttnNoSelectMultiLabelClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnmeanwordlayerattnnoselectadapter': (BertConfig, BertClaimAttnMeanWordLayerAttnNoSelectMultiLabelAdapterClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnmeanwordattn': (BertConfig, BertClaimAttnMeanWordAttnMultiLabelClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnweightcossimwordattnadapter': (BertConfig, BertClaimAttnWeightCosSimWordAttnMultiLabelAdapterClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnmeancossimwordlayerattnadapter': (BertConfig, BertClaimAttnMeanCosSimWordLayerAttnMultiLabelAdapterClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnweightcossimwordlayerattnadapter': (BertConfig, BertClaimAttnWeightCosSimWordLayerAttnMultiLabelAdapterClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnmeanadapter': (BertConfig, BertClaimAttnMeanMultiLabelAdapterClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnmilcossimwordattnadapter': (BertConfig, BertClaimAttnMilCosSimMultiLabelAdapterWordAttnClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnmilcossimwordattnsimpleadapter': (BertConfig, BertClaimAttnMilCosSimMultiLabelAdapterWordAttnSimpleClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnmeanmilcossimadapter': (BertConfig, BertClaimAttnMeanMilCosSimMultiLabelAdapterClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnmeanmiladapter': (BertConfig, BertClaimAttnMeanMilMultiLabelAdapterClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnmaxmilcossimadapter': (BertConfig, BertClaimAttnMaxMilCosSimMultiLabelAdapterClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnsepmilcossimadapter': (BertConfig, BertClaimAttnSepMilCosSimMultiLabelAdapterClassifier, BertTokenizerFast),
    'bertmultilabelclaimattnmilcossim': (BertConfig, BertClaimAttnMilCosSimMultiLabelClassifier, BertTokenizerFast),
    'bertbinarylabelseqlabel': (BertConfig, BertBinaryLabelSeqLabelClassifier, BertTokenizerFast),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def make_model_input(args, batch):
    inputs = {'input_ids':        batch[0],
              'attention_mask':   batch[1],
              'token_type_ids':   batch[2],
              'labels':           batch[3]}

    # add extraction and augmentation spans for PointerBert model
    if args.model_type == "pbert":
        inputs["ext_mask"] = batch[4]
        inputs["ext_start_labels"] = batch[5]
        inputs["ext_end_labels"] = batch[6]
        inputs["aug_mask"] = batch[7]
        inputs["aug_start_labels"] = batch[8]
        inputs["aug_end_labels"] = batch[9]
        inputs["loss_lambda"] = args.loss_lambda
    elif args.model_type == "longformermultilabelseqlabel" or args.model_type == "bertmultilabelseqlabel" or args.model_type == "bertbinarylabelseqlabel":
        inputs["aug_seq_labels_mask"] = batch[7]
        inputs["aug_seq_labels"] = batch[8]
        inputs["loss_lambda"] = args.loss_lambda
    elif args.model_type == "bertmultilabelsrlattn" or args.model_type == "bertmultilabelsrlattnfc":
        inputs["verb_attn_mask"] = batch[4]
        inputs["claim_attn_mask"] = batch[5]
    elif args.model_type in ["bertmultilabelclaimattn", "bertmultilabelclaimattnmiladapter", "bertmultilabelclaimattnadapter", "bertmultilabelclaimattnmil"]:
        inputs["doc_frames_word_mask"] = batch[4]
        inputs["claim_frames_word_mask"] = batch[5]
        inputs["claim_attn_mask"] = batch[6]
        inputs["claim_frames_padding_mask"] = batch[7]
    elif args.model_type in ["bertmultilabelclaimattnmilcossimadapter", "bertmultilabelclaimattnmilcossim", "bertmultilabelclaimattnmeanmilcossimadapter", "bertmultilabelclaimattnmaxmilcossimadapter", "bertmultilabelclaimattnmeanmiladapter", "bertmultilabelclaimattnmilcossimwordattnadapter", "bertmultilabelclaimattnmilcossimwordattnsimpleadapter", "bertmultilabelclaimattnmeancossimadapter", "bertmultilabelclaimattnmeancossimwordattnadapter", "bertmultilabelclaimattnmeanwordattnadapter", "bertmultilabelclaimattnmeanwordlayerattnadapter", "bertmultilabelclaimattnmeanwordlayerattnnoselect", "bertmultilabelclaimattnmeanwordlayerattnnoselectadapter", "bertmultilabelclaimattnmeanwordlayerattn", "bertmultilabelclaimattnmeanwordattn", "bertmultilabelclaimattnweightcossimwordattnadapter", "bertmultilabelclaimattnmeancossimwordlayerattnadapter", "bertmultilabelclaimattnweightcossimwordlayerattnadapter", "bertmultilabelclaimattnmeanadapter"]:
        inputs["doc_frames_word_mask"] = batch[4]
        inputs["claim_frames_word_mask"] = batch[5]
        inputs["claim_attn_mask"] = batch[6]
        inputs["claim_frames_padding_mask"] = batch[7]
        inputs["doc_words_mask"] = batch[8]
        # doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask, claim_frames_padding_mask
    elif args.model_type == "bertmultilabelclaimattnsepmilcossimadapter":
        inputs["doc_frames_word_mask"] = batch[4]
        inputs["claim_frames_word_mask"] = batch[5]
        inputs["claim_attn_mask"] = batch[6]
        inputs["claim_frames_padding_mask"] = batch[7]
        inputs["doc_words_mask"] = batch[8]
        #doc_frames_np_word_mask, doc_frames_predicate_word_mask, claim_frames_np_word_mask, claim_frames_predicate_word_mask
        inputs["doc_frames_np_word_mask"] = batch[9]
        inputs["doc_frames_predicate_word_mask"] = batch[10]
        inputs["claim_frames_np_word_mask"] = batch[11]
        inputs["claim_frames_predicate_word_mask"] = batch[12]
        # doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask, claim_frames_padding_mask
    return inputs


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if "adapter" in args.model_type:
        params = []
        params_name = []
        params_name_frozen = []

        num_params = 0
        num_params_frozen = 0
        #trained_params = ["adapter", "classifier", "pooler_g", "embeddings"]
        if args.model_type == "bertmultilabeladapter":
            trained_params = ["adapter", "classifier", "embeddings"]
        elif args.model_type == "bertmultilabelclaimattnadapter":
            trained_params = ["adapter", "classifier", "embeddings", "claim_frame_attn", "frame_feature_project", "claim_frame_attn_layer_norm"]
        elif args.model_type == "bertmultilabelclaimattnmiladapter":
            trained_params = ["adapter", "classifier", "embeddings", "claim_frame_attn", "frame_feature_project",
                              "claim_frame_attn_layer_norm", "classifier_attn"]
        elif args.model_type == "bertmultilabelclaimattnmilcossimadapter":
            trained_params = ["adapter", "classifier", "embeddings", "claim_frame_attn", "frame_feature_project",
                              "claim_frame_attn_layer_norm", "classifier_attn"]
        elif args.model_type in ["bertmultilabelclaimattnmeancossimadapter", "bertmultilabelclaimattnmeancossimwordattnadapter", "bertmultilabelclaimattnmeanwordattnadapter", "bertmultilabelclaimattnmeanwordlayerattnadapter", "bertmultilabelclaimattnmeanwordlayerattnnoselectadapter", "bertmultilabelclaimattnweightcossimwordattnadapter", "bertmultilabelclaimattnmeancossimwordlayerattnadapter", "bertmultilabelclaimattnweightcossimwordlayerattnadapter", "bertmultilabelclaimattnmeanadapter"]:
            trained_params = ["adapter", "classifier", "embeddings", "claim_frame_attn", "frame_feature_project",
                              "claim_frame_attn_layer_norm", "layer_attn", "token_self_attn", "frame_classification_features_attn"]
        elif args.model_type == "bertmultilabelclaimattnmilcossimwordattnadapter":
            trained_params = ["adapter", "classifier", "embeddings", "claim_frame_attn", "frame_feature_project",
                              "claim_frame_attn_layer_norm", "classifier_attn", "token_self_attn"]
        elif args.model_type == "bertmultilabelclaimattnmilcossimwordattnsimpleadapter":
            trained_params = ["adapter", "classifier", "embeddings", "claim_frame_attn",
                              "claim_frame_attn_layer_norm", "classifier_attn", "token_self_attn"]
        elif args.model_type == "bertmultilabelclaimattnmeanmilcossimadapter" or args.model_type == "bertmultilabelclaimattnmaxmilcossimadapter" or args.model_type == "bertmultilabelclaimattnmeanmiladapter":
            trained_params = ["adapter", "classifier", "embeddings", "claim_frame_attn", "frame_feature_project",
                              "claim_frame_attn_layer_norm"]
        elif args.model_type == "bertmultilabelclaimattnsepmilcossimadapter":
            trained_params = ["adapter", "classifier", "embeddings", "claim_frame_attn", "np_feature_project", "predicate_feature_project",
                              "claim_frame_attn_layer_norm", "classifier_attn"]
        else:
            if "adapter" in args.model_type:
                print("did not specify trained_params !")
                raise ValueError

        for n, p in model.named_parameters():
            trained = False
            for trained_param in trained_params:
                if trained_param in n:
                    if trained_param != 'embeddings':
                        num_params += p.numel()
                        trained = True
                    params.append(p)
                    params_name.append(n)
            if not trained:
                num_params_frozen += p.numel()
                params_name_frozen.append(n)
        named_parameters = zip(params_name, params)

        logging.info("Frozen parameters: %s", params_name_frozen)
        logging.info("Learned parameters: %s", params_name)
        logging.info("# parameters: %f", num_params)
        logging.info("# frozen parameters: %f", num_params_frozen)
        logging.info("percentage learned parameters: %.4f", num_params / num_params_frozen)
    else:
        named_parameters = model.named_parameters()

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in named_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", 
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    if args.model_type == "longformermultilabelseqlabel" or args.model_type == "bertmultilabelseqlabel" or args.model_type == "bertbinarylabelseqlabel":
        tr_error_class_loss, logging_error_class_loss = 0.0, 0.0
        tr_seq_label_loss, logging_seq_label_loss = 0.0, 0.0

    model.zero_grad()
    if args.resume_from_checkpoint:
        logger.info("loading optimizer state from {}".format(os.path.join(args.model_name_or_path, 'optimizer_state.pt')))
        previous_optimizer_state_dict = torch.load(os.path.join(args.model_name_or_path, 'optimizer_state.pt'), map_location=torch.device('cpu'))
        epochs_trained = previous_optimizer_state_dict["epochs_trained"]
        optimizer.load_state_dict(previous_optimizer_state_dict["optimizer"])
        scheduler.load_state_dict(previous_optimizer_state_dict["scheduler"])
    else:
        epochs_trained = 0
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    best_eval_f1 = -1
    #best_eval_auc = -1
    best_eval_bacc = -1
    best_f1_epoch_ix = -1
    #best_auc_epoch_ix = -1
    best_bacc_epoch_ix = -1

    for epoch_ix in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = make_model_input(args, batch)
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if args.model_type == "longformermultilabelseqlabel" or args.model_type == "bertmultilabelseqlabel" or args.model_type == "bertbinarylabelseqlabel":
                error_class_loss = outputs[1]
                seq_label_loss = outputs[2]

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if args.model_type == "longformermultilabelseqlabel" or args.model_type == "bertmultilabelseqlabel" or args.model_type == "bertbinarylabelseqlabel":
                tr_error_class_loss += error_class_loss.item()
                tr_seq_label_loss += seq_label_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    results = {}
                    # Log metrics
                    #logits_ix = 1 if args.model_type == "bert" or args.model_type == "bertmultilabel" or args.model_type == "longformermultilabel" else 7
                    if args.model_type == "bert" or args.model_type == "bertmultilabel" or args.model_type == "bertmultilabeladapter" or args.model_type.startswith("bertmultilabelclaimattn"):
                        logits_ix = 1
                    elif args.model_type == "longformermultilabelseqlabel" or args.model_type == "bertmultilabelseqlabel" or args.model_type == "bertbinarylabelseqlabel":
                        logits_ix = 3
                    else:
                        raise ValueError
                        logits_ix = 7
                    logits = outputs[logits_ix]
                    #if args.task_name == "factcc_multi_label_generated":
                    if "multi_label" in args.task_name or args.task_name == "factcc_binary_label_seq_label_generated":
                        preds = torch.sigmoid(logits).detach().cpu().numpy().round().astype(int)
                    else:
                        preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
                    gold_label_ids = inputs['labels'].detach().cpu().numpy()

                    result = compute_metrics(args.task_name, preds, gold_label_ids)
                    results.update(result)

                    for key, value in results.items():
                        wandb.log({'train_{}'.format(key): value})

                    wandb.log({"train_loss": (tr_loss - logging_loss) / args.logging_steps})
                    wandb.log({"train_lr": scheduler.get_last_lr()[0]})
                    if "seq_label" in args.task_name:
                        wandb.log({"error_class_loss": (tr_error_class_loss - logging_error_class_loss) / args.logging_steps})
                        wandb.log({"seq_label_loss": (tr_seq_label_loss - logging_seq_label_loss) / args.logging_steps})
                        logging_error_class_loss = tr_error_class_loss
                        logging_seq_label_loss = tr_seq_label_loss
                    logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.evaluate_during_training:
            # Only evaluate when single GPU otherwise metrics may not average well
            results = evaluate(args, model, tokenizer)
            for key, value in results.items():
                wandb.log({'eval_{}'.format(key): value})
                logger.info("log eval_{} to wandb".format(key))

            if results["f1"] > best_eval_f1:
                best_eval_f1 = results["f1"]
                best_f1_epoch_ix = epoch_ix
            #if results["auc"] > best_eval_auc:
            #    best_eval_auc = results["auc"]
            #    best_auc_epoch_ix = epoch_ix
            if results["bacc"] > best_eval_bacc:
                best_eval_bacc = results["bacc"]
                best_bacc_epoch_ix = epoch_ix

        if args.local_rank in [-1, 0]:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch_ix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            tokenizer.save_pretrained(output_dir)
            optimizer_state = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epochs_trained": epoch_ix + 1
            }
            torch.save(optimizer_state, os.path.join(output_dir, 'optimizer_state.pt'))
            logger.info("Saving model checkpoint to %s", output_dir)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    logger.info("*********************".format(best_eval_f1))
    logger.info("Best eval f1: {:.4f}".format(best_eval_f1))
    logger.info("Best f1 epoch ix: {}".format(best_f1_epoch_ix))
    logger.info("Best f1 chkpt path: {}".format(os.path.join(args.output_dir, 'checkpoint-{}'.format(best_f1_epoch_ix))))

    #logger.info("Best eval auc: {:.4f}".format(best_eval_auc))
    #logger.info("Best auc epoch ix: {}".format(best_auc_epoch_ix))
    #logger.info(
    #    "Best auc chkpt path: {}".format(os.path.join(args.output_dir, 'checkpoint-{}'.format(best_auc_epoch_ix))))

    logger.info("Best eval bacc: {:.4f}".format(best_eval_bacc))
    logger.info("Best bacc epoch ix: {}".format(best_bacc_epoch_ix))
    logger.info(
        "Best bacc chkpt path: {}".format(os.path.join(args.output_dir, 'checkpoint-{}'.format(best_bacc_epoch_ix))))

    # remove useless checkpts
    if args.keep_all_ckpts:
        pass
    else:
        for i in range(int(args.num_train_epochs)):
            if i not in [best_f1_epoch_ix, best_bacc_epoch_ix]:
                remove_dir_path = os.path.join(args.output_dir, 'checkpoint-{}'.format(i))
                if os.path.exists(remove_dir_path):
                    shutil.rmtree(remove_dir_path)
                    logger.info("Remove dir: {}".format(remove_dir_path))

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    if args.eval_output_dir:
        eval_outputs_dirs = (args.eval_output_dir,)
    else:
        eval_outputs_dirs = (args.output_dir,)

    results = {}
    cnt = 0
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, _ = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        gold_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = make_model_input(args, batch)
                outputs = model(**inputs)

                # monitoring
                tmp_eval_loss = outputs[0]
                #logits_ix = 1 if args.model_type == "bert" or args.model_type == "bertmultilabel" or args.model_type == "longformermultilabel" else 7
                if args.model_type == "bert" or args.model_type == "bertmultilabel" or args.model_type == "bertmultilabeladapter" or args.model_type.startswith("bertmultilabelclaimattn"):
                    logits_ix = 1
                elif args.model_type == "longformermultilabelseqlabel" or args.model_type == "bertmultilabelseqlabel" or args.model_type == "bertbinarylabelseqlabel":
                    logits_ix = 3
                else:
                    raise ValueError
                    logits_ix = 7
                logits = outputs[logits_ix]
                eval_loss += tmp_eval_loss.mean().item()
                if "seq_label" in args.task_name:
                    seq_label_logits_ix = 4
                    seq_label_logits = outputs[seq_label_logits_ix]
                nb_eval_steps += 1

            if preds is None:
                #if args.task_name == "factcc_multi_label_generated":
                if "multi_label" in args.task_name or args.task_name == "factcc_binary_label_seq_label_generated":
                    # logits: [batch, num_classes]
                    preds = torch.sigmoid(logits).detach().cpu().numpy()

                    if args.model_type.startswith("bertmultilabelclaimattnmil"):
                        claim_frame_logits = outputs[logits_ix + 1]
                        claim_frame_preds = torch.sigmoid(
                            claim_frame_logits).detach().cpu().numpy()  # [batch, n_claim_fram, n_class]
                        claim_attn_output_weights = outputs[
                            logits_ix + 3].detach().cpu().numpy()  # [batch, tgt_seq_len, src_seq_len]
                        classifier_attn_weights = outputs[logits_ix + 2].detach().cpu().numpy()
                    elif args.model_type.startswith("bertmultilabelclaimattnmeanmil"):
                        claim_frame_logits = outputs[logits_ix + 1]
                        claim_frame_preds = torch.sigmoid(
                            claim_frame_logits).detach().cpu().numpy()  # [batch, n_claim_fram, n_class]
                        claim_attn_output_weights = outputs[
                            logits_ix + 2].detach().cpu().numpy()  # [batch, tgt_seq_len, src_seq_len]
                    elif args.model_type in ["bertmultilabelclaimattnmeanadapter", "bertmultilabelclaimattnmeancossimadapter", "bertmultilabelclaimattnmeancossimwordattnadapter", "bertmultilabelclaimattnmeanwordattnadapter", "bertmultilabelclaimattnmeanwordlayerattnadapter", "bertmultilabelclaimattnmeanwordlayerattn", "bertmultilabelclaimattnmeanwordattn", "bertmultilabelclaimattnweightcossimwordattnadapter", "bertmultilabelclaimattnmeancossimwordlayerattnadapter", "bertmultilabelclaimattnweightcossimwordlayerattnadapter"]:
                        claim_attn_output_weights = outputs[
                            logits_ix + 1].detach().cpu().numpy()  # [batch, tgt_seq_len, src_seq_len]
                    elif args.model_type == "bertmultilabel" or args.model_type == "bertmultilabeladapter":
                        total_attn_from_cls = outputs[
                            logits_ix + 1].detach().cpu().numpy()  # [batch, seq_len]
                    if "frank" in args.task_name or args.export_output or args.export_val_output:
                        guids = batch[-1].detach().cpu().numpy()
                        input_ids = inputs["input_ids"].detach().cpu().numpy()
                        #original_text_b_ids = batch[-2].detach().cpu().numpy()
                else:
                    preds = logits.detach().cpu().numpy()
                gold_label_ids = inputs['labels'].detach().cpu().numpy()
                if "seq_label" in args.task_name:
                    seq_label_preds = torch.sigmoid(seq_label_logits).squeeze(-1).detach().cpu().numpy()
                    seq_label_ids = inputs['aug_seq_labels'].squeeze(-1).detach().cpu().numpy()
            else:
                #if args.task_name == "factcc_multi_label_generated":
                if "multi_label" in args.task_name or args.task_name == "factcc_binary_label_seq_label_generated":
                    preds = np.append(preds, torch.sigmoid(logits).detach().cpu().numpy(), axis=0)
                    if args.model_type.startswith("bertmultilabelclaimattnmil"):
                        claim_frame_logits = outputs[logits_ix + 1]
                        claim_frame_preds = np.append(claim_frame_preds,
                                                      torch.sigmoid(claim_frame_logits).detach().cpu().numpy(), axis=0)
                        claim_attn_output_weights = np.append(claim_attn_output_weights,
                                                              outputs[logits_ix + 3].detach().cpu().numpy(), axis=0)
                        classifier_attn_weights = np.append(classifier_attn_weights, outputs[logits_ix + 2].detach().cpu().numpy(), axis=0)
                    elif args.model_type.startswith("bertmultilabelclaimattnmeanmil"):
                        claim_frame_logits = outputs[logits_ix + 1]
                        claim_frame_preds = np.append(claim_frame_preds,
                                                      torch.sigmoid(claim_frame_logits).detach().cpu().numpy(), axis=0)
                        claim_attn_output_weights = np.append(claim_attn_output_weights,
                                                              outputs[logits_ix + 2].detach().cpu().numpy(), axis=0)
                    elif args.model_type in ["bertmultilabelclaimattnmeanadapter",
                                             "bertmultilabelclaimattnmeancossimadapter", "bertmultilabelclaimattnmeancossimwordattnadapter", "bertmultilabelclaimattnmeanwordattnadapter", "bertmultilabelclaimattnmeanwordlayerattnadapter", "bertmultilabelclaimattnmeanwordlayerattn", "bertmultilabelclaimattnmeanwordattn", "bertmultilabelclaimattnweightcossimwordattnadapter", "bertmultilabelclaimattnmeancossimwordlayerattnadapter", "bertmultilabelclaimattnweightcossimwordlayerattnadapter"]:
                        claim_attn_output_weights = np.append(claim_attn_output_weights,
                                                              outputs[logits_ix + 1].detach().cpu().numpy(), axis=0)
                    elif args.model_type == "bertmultilabel" or args.model_type == "bertmultilabeladapter":
                        total_attn_from_cls = np.append(total_attn_from_cls, outputs[logits_ix + 1].detach().cpu().numpy(), axis=0)
                    if "frank" in args.task_name or args.export_output or args.export_val_output:
                        guids = np.append(guids, batch[-1].detach().cpu().numpy(), axis=0)
                        input_ids = np.append(input_ids, inputs["input_ids"].detach().cpu().numpy(), axis=0)
                        #original_text_b_ids = np.append(original_text_b_ids, batch[-2].detach().cpu().numpy(), axis=0)
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                gold_label_ids = np.append(gold_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                if "seq_label" in args.task_name:
                    seq_label_preds = np.append(seq_label_preds, torch.sigmoid(seq_label_logits).squeeze(-1).detach().cpu().numpy(), axis=0)
                    seq_label_ids = np.append(seq_label_ids, inputs['aug_seq_labels'].squeeze(-1).detach().cpu().numpy(), axis=0)
        #if args.task_name == "factcc_multi_label_generated":
        if "multi_label" in args.task_name or args.task_name == "factcc_binary_label_seq_label_generated":
            # preds: [batch, num_classes]
            #print("preds")
            #print(preds)
            preds_raw = preds
            preds = preds_raw.round().astype(int)
            gold_label_ids = gold_label_ids.astype(int)
            #print("preds")
            #print(preds)
            #print("gold_label_ids")
            #print(gold_label_ids)
        else:
            preds = np.argmax(preds, axis=1)
        if "seq_label" in args.task_name:
            seq_label_preds = seq_label_preds.round().astype(int)
            seq_label_ids = seq_label_ids.astype(int)
            #print("seq_label_preds")
            #print(seq_label_preds)
            #print("seq_label_ids")
            #print(seq_label_ids)
        if "frank" in args.task_name:
            summary_level_preds_dict = dict()  # input_id: prediction array
            summary_level_labels_dict = dict()  # input_id: label array
            sep_id = tokenizer.convert_tokens_to_ids([tokenizer.sep_token])[0]

            # export output
            output_js = []
            #for sample_i, (input_id, pred, label, original_text_b_id) in enumerate(zip(input_ids, preds, gold_label_ids, original_text_b_ids)):
            for sample_i, (input_id, pred, label) in enumerate(zip(input_ids, preds, gold_label_ids)):
                input_id = input_id.tolist()
                #original_text_b_id = original_text_b_id.tolist()
                input_tokens = tokenizer.convert_ids_to_tokens(input_id)
                input_text = tokenizer.decode(input_id, skip_special_tokens=False)
                #original_text_b = tokenizer.decode(original_text_b_id, skip_special_tokens=True)
                output_dict = {"text": input_text, "ground-truth": [args.label_list[idx] for idx, l in enumerate(label) if l == 1],
                               "pred": [args.label_list[idx] for idx, p in enumerate(pred) if p == 1],
                               "pred_id": pred.tolist(), "ground-truth id": label.tolist()}
                               #"pred_id": pred.tolist(), "ground-truth id": label.tolist(), "original_text_b": original_text_b}
                if "seq_label" in args.task_name:
                    seq_label_pred = seq_label_preds[sample_i].tolist()
                    assert len(input_id) == len(seq_label_pred)
                    sep_locations = [i for i, idx in enumerate(input_id) if idx == sep_id]
                    #sep_location = sep_locations[0]
                    seq_label_pred_sequence_b = seq_label_pred[sep_locations[0]+1: sep_locations[1]]
                    input_tokens_sequence_b = input_tokens[sep_locations[0]+1: sep_locations[1]]
                    sequence_label_prediction = []
                    assert len(seq_label_pred) == len(input_tokens)
                    for _label, _token in zip(seq_label_pred, input_tokens):
                        if _label == 1:
                            sequence_label_prediction.append(_token)
                    output_dict["seq_label_preds_tokens"] = sequence_label_prediction
                    output_dict["input_tokens_sequence_b"] = input_tokens_sequence_b
                    output_dict["seq_label_preds"] = seq_label_pred_sequence_b
                #output_dict = {"text": input_text, "pred": pred.tolist(), "label": label.tolist(), "label_names": [args.label_list[idx] for idx, l in enumerate(label) if l == 1], "label_names_all": args.label_list}
                output_js.append(output_dict)

            with open(os.path.join(eval_output_dir, "model_outputs.json"), "w") as outfile:
                json.dump(output_js, outfile, indent=4)

            torch.save(preds_raw, os.path.join(eval_output_dir, "preds_raw.pt"))
            print("preds_raw size")
            print(preds_raw.size)

            for guid, pred, label in zip(guids, preds, gold_label_ids):
                if guid in summary_level_preds_dict.keys():
                    summary_level_preds_dict[guid] = summary_level_preds_dict[guid] | pred
                    summary_level_labels_dict[guid] = summary_level_labels_dict[guid] | label
                else:
                    summary_level_preds_dict[guid] = pred
                    summary_level_labels_dict[guid] = label
            summary_level_preds = []
            summary_level_labels = []
            for guid in summary_level_preds_dict.keys():
                summary_level_preds.append(summary_level_preds_dict[guid])
                summary_level_labels.append(summary_level_labels_dict[guid])
            print("summary_level_preds")
            print(len(summary_level_preds))
            preds = np.asarray(summary_level_preds)
            gold_label_ids = np.asarray(summary_level_labels)
        elif args.export_output or args.export_val_output:
            if args.export_output:
                model_output_filename = "model_outputs.jsonl"
            elif args.export_val_output:
                model_output_filename = "model_outputs_val.jsonl"
            # export output
            with open(os.path.join(eval_output_dir, model_output_filename), "w") as f_out:
                for sample_i, (input_id, pred, pred_raw, label) in enumerate(zip(input_ids, preds, preds_raw, gold_label_ids)):
                    input_id = input_id.tolist()
                    # original_text_b_id = original_text_b_id.tolist()
                    #input_tokens = tokenizer.convert_ids_to_tokens(input_id)
                    input_text = tokenizer.decode(input_id, skip_special_tokens=False)
                    # original_text_b = tokenizer.decode(original_text_b_id, skip_special_tokens=True)
                    output_dict = {"text": input_text,
                                   "ground-truth": [args.label_list[idx] for idx, l in enumerate(label) if l == 1],
                                   "pred": [args.label_list[idx] for idx, p in enumerate(pred) if p == 1],
                                   "pred_id": pred.tolist(), "pred_scores": pred_raw.tolist(), "ground-truth id": label.tolist()}
                    #output_js.append(output_dict)
                    if args.model_type.startswith("bertmultilabelclaimattnmil"):
                        output_dict["claim_frame_preds"] = claim_frame_preds[sample_i, :, :].tolist()
                        output_dict["claim_attn_output_weights"] = claim_attn_output_weights[sample_i, :, :].tolist()
                        output_dict["classifier_attn_weights"] = classifier_attn_weights[sample_i, :, :].tolist()
                    elif args.model_type.startswith("bertmultilabelclaimattnmeanmil"):
                        output_dict["claim_frame_preds"] = claim_frame_preds[sample_i, :, :].tolist()
                        output_dict["claim_attn_output_weights"] = claim_attn_output_weights[sample_i, :, :].tolist()
                    elif args.model_type in ["bertmultilabelclaimattnmeanadapter",
                                                 "bertmultilabelclaimattnmeancossimadapter", "bertmultilabelclaimattnmeancossimwordattnadapter", "bertmultilabelclaimattnmeanwordattnadapter", "bertmultilabelclaimattnmeanwordlayerattnadapter", "bertmultilabelclaimattnmeanwordlayerattn", "bertmultilabelclaimattnmeanwordattn", "bertmultilabelclaimattnweightcossimwordattnadapter", "bertmultilabelclaimattnmeancossimwordlayerattnadapter", "bertmultilabelclaimattnweightcossimwordlayerattnadapter"]:
                        output_dict["claim_attn_output_weights"] = claim_attn_output_weights[sample_i, :, :].tolist()
                    elif args.model_type == "bertmultilabel" or args.model_type == "bertmultilabeladapter":
                        #print("attn_out_shape")
                        #print(total_attn_from_cls.shape)
                        output_dict["total_attn_from_cls"] = total_attn_from_cls[sample_i, :].tolist()
                    f_out.write(json.dumps(output_dict) + '\n')
            print("output exported to {}".format(os.path.join(eval_output_dir, "model_outputs.jsonl")))

        if "multi_label" in args.task_name:
            """
            gold_consistency_label = gold_label_ids[:, 0]
            pred_consistency_label = preds[:, 0]  # [batch]
            gold_error_class_label = gold_label_ids[:, 1:]
            pred_error_class_label = preds[:, 1:]
            pred_consistency_label_expanded = np.expand_dims(pred_consistency_label, axis=1)  # [batch, 1]
            pred_error_class_label_filtered = pred_error_class_label * pred_consistency_label_expanded
            result = compute_metrics(args.task_name, pred_error_class_label_filtered, gold_error_class_label)
            """
            result = multi_label_bacc_auc_f1_macro(preds=preds, preds_scores=preds_raw, labels=gold_label_ids)
        else:
            result = compute_metrics(args.task_name, preds, gold_label_ids)
        #print("result")
        #print(result)
        eval_loss = eval_loss / nb_eval_steps
        result["loss"] = eval_loss
        results.update(result)

        if args.task_name == "factcc_amr_aug_multi_label_seq_label_generated" or args.task_name == "factcc_multi_label_seq_label_generated" or args.task_name == "factcc_binary_label_seq_label_generated":
            seq_label_results = compute_seq_label_metrics(seq_label_preds, seq_label_ids, prefix="seq_label_")
            logger.info("*** seq_label_results ***")
            logger.info(seq_label_results)
            results.update(seq_label_results)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))

        if "multi_label" in args.task_name:
            """
            logger.info("***** Consistency Label *****")
            logger.info(classification_metric_all(preds=pred_consistency_label, labels=gold_consistency_label))
            logger.info("***** Error type Label *****")
            for i, error_type_name in enumerate(args.label_list[1:]):
                logger.info(error_type_name)
                gold_error_class_label_i = gold_error_class_label[:, i]
                pred_error_class_label_i = pred_error_class_label_filtered[:, i]
                logger.info(classification_metric_all(preds=pred_error_class_label_i, labels=gold_error_class_label_i))
                #error_types_scores[error_type_name] = classification_metric_all(preds=pred_error_class_label_i,
                #                                                                labels=gold_error_class_label_i)
            logger.info(classification_report(
                y_true=gold_error_class_label,
                y_pred=pred_error_class_label_filtered,
                target_names=args.label_list[1:],
                digits=4
            ))
            """
            for i, error_type_name in enumerate(args.label_list):
                logger.info(error_type_name)
                gold_error_class_label_i = gold_label_ids[:, i]
                pred_error_class_label_i = preds[:, i]
                pred_error_scores_label_i = preds_raw[:, i]
                #logger.info(classification_metric_all(preds=pred_error_class_label_i, preds_scores=pred_error_scores_label_i, labels=gold_error_class_label_i))
            """
            logger.info("***** classification_report *****")
            logger.info(classification_report(
                y_true=gold_label_ids,
                y_pred=preds,
                target_names=args.label_list,
                digits=4
            ))
            """

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    if args.model_type == "bertmultilabelsrlattnfc" or args.model_type == "bertmultilabelsrlattn":
        model_type = "bertmultilabelsrlattn"
    else:
        model_type = args.model_type
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        #list(filter(None, args.model_name_or_path.split('/'))).pop(),
        model_type,
        str(args.max_seq_length),
        str(task)))

    if "attn" in task:
        if not os.path.exists(cached_features_file):
            os.makedirs(cached_features_file)
            logger.info("Creating features from dataset file at %s", args.data_dir)
            label_list = processor.get_labels()
            examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(
                args.data_dir)
            print("len(examples)")
            print(len(examples))
            for (ex_index, example) in enumerate(examples):
                if ex_index % 10000 == 0:
                    logger.info("Writing example %d of %d" % (ex_index, len(examples)))
                feature = convert_claim_attn_examples_to_features(example, label_list, args.max_seq_length, tokenizer, output_mode,
                                                  pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0])
                # log
                if ex_index < 3:
                    logger.info("*** Example ***")
                    logger.info("guid: %s" % (example.guid))
                    # logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                    logger.info("input_ids: %s" % " ".join([str(x) for x in feature.input_ids]))
                    logger.info("input_mask: %s" % " ".join([str(x) for x in feature.input_mask]))
                    logger.info("segment ids: %s" % " ".join([str(x) for x in feature.segment_ids]))
                    logger.info("claim_attn_mask: %s" % " ".join([str(x) for x in feature.claim_attn_mask]))
                    logger.info("claim_frames_padding_mask: %s" % " ".join([str(x) for x in feature.claim_frames_padding_mask]))
                    logger.info("doc_frames_word_mask: %s" % " ".join([str(x) for x in feature.doc_frames_word_mask[0]]))
                    #logger.info("doc_frames_word_mask: %s" % " ".join([str(x) for x in feature.doc_frames_word_mask[1]]))
                    logger.info("doc_frames_np_word_mask: %s" % " ".join([str(x) for x in feature.doc_frames_np_word_mask[0]]))
                    logger.info("doc_frames_predicate_word_mask: %s" % " ".join([str(x) for x in feature.doc_frames_predicate_word_mask[0]]))
                    logger.info("claim_frames_word_mask: %s" % " ".join([str(x) for x in feature.claim_frames_word_mask[0]]))
                    logger.info("claim_frames_np_word_mask: %s" % " ".join([str(x) for x in feature.claim_frames_np_word_mask[0]]))
                    logger.info("claim_frames_predicate_word_mask: %s" % " ".join([str(x) for x in feature.claim_frames_predicate_word_mask[0]]))
                    #logger.info("claim_frames_word_mask: %s" % " ".join([str(x) for x in feature.claim_frames_word_mask[1]]))
                    logger.info("doc_words_mask: %s" % " ".join([str(x) for x in feature.doc_words_mask]))
                    if output_mode == "multi_label_classification":
                        logger.info("label: %s" % " ".join([str(x) for x in feature.label_id]))
                    else:
                        logger.info("label: %d" % feature.label_id)

                # Convert to Tensors and build dataset
                input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
                input_mask = torch.tensor(feature.input_mask, dtype=torch.long)
                segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long)
                doc_frames_word_mask = torch.tensor(feature.doc_frames_word_mask, dtype=torch.float)
                doc_frames_np_word_mask = torch.tensor(feature.doc_frames_np_word_mask, dtype=torch.float)
                doc_frames_predicate_word_mask = torch.tensor(feature.doc_frames_predicate_word_mask, dtype=torch.float)
                claim_frames_word_mask = torch.tensor(feature.claim_frames_word_mask, dtype=torch.float)
                claim_frames_np_word_mask = torch.tensor(feature.claim_frames_np_word_mask, dtype=torch.float)
                claim_frames_predicate_word_mask = torch.tensor(feature.claim_frames_predicate_word_mask, dtype=torch.float)
                claim_attn_mask = torch.tensor(feature.claim_attn_mask, dtype=torch.bool)
                claim_frames_padding_mask = torch.tensor(feature.claim_frames_padding_mask, dtype=torch.float)
                doc_words_mask = torch.tensor(feature.doc_words_mask, dtype=torch.float)
                guid = torch.tensor(feature.guid, dtype=torch.long)

                if output_mode == "classification":
                    if model_type == "bertbinarylabelseqlabel":
                        label_ids = torch.tensor(feature.label_id, dtype=torch.float)
                    else:
                        label_ids = torch.tensor(feature.label_id, dtype=torch.long)
                elif output_mode == "multi_label_classification":
                    label_ids = torch.tensor(feature.label_id, dtype=torch.float)
                else:
                    raise ValueError

                torch.save((input_ids, input_mask, segment_ids, label_ids, doc_frames_word_mask, claim_frames_word_mask,
                            claim_attn_mask, claim_frames_padding_mask, doc_words_mask, doc_frames_np_word_mask, doc_frames_predicate_word_mask, claim_frames_np_word_mask, claim_frames_predicate_word_mask, guid),
                           os.path.join(cached_features_file, "{}.pt".format(ex_index)))

        dataset = FactCCGeneratedAttnDataset(cached_features_file)
        # compute positive weights for multi label classification
        if output_mode == "multi_label_classification":
            with torch.no_grad():
                all_label_ids = []
                num_samples = len(dataset)
                for i in range(num_samples):
                    input_ids, input_mask, segment_ids, label_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask, claim_frames_padding_mask, doc_words_mask, doc_frames_np_word_mask, doc_frames_predicate_word_mask, claim_frames_np_word_mask, claim_frames_predicate_word_mask, guid = dataset[i]
                    all_label_ids.append(label_ids)
                all_label_ids = torch.stack(all_label_ids, dim=0)
                print("all_label_ids")
                print(all_label_ids.size())
                num_samples = all_label_ids.size(0)
                num_positive_samples_in_each_dim = all_label_ids.sum(0)  # [label_dim]
                num_negative_samples_in_each_dim = num_samples - num_positive_samples_in_each_dim
                positive_weights = num_negative_samples_in_each_dim / num_positive_samples_in_each_dim
                print("num_samples")
                print(num_samples)
                print("num_positive_samples_in_each_dim")
                print(num_positive_samples_in_each_dim)
                print("num_negative_samples_in_each_dim")
                print(num_negative_samples_in_each_dim)
                print("positive_weights")
                print(positive_weights)
        else:
            positive_weights = None

    else:
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            label_list = processor.get_labels()
            examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
            if "multi_label_seq_label" in task:
                convert_func = convert_seq_label_examples_to_features
            elif "binary_label_seq_label" in task:
                convert_func = convert_binary_seq_label_examples_to_features
            else:
                convert_func = convert_examples_to_features
            features = convert_func(examples, label_list, args.max_seq_length, tokenizer, output_mode,
                cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=bool("roberta" in args.model_type or "longformer" in args.model_type),
                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                sequence_b_segment_id=0 if "roberta" in args.model_type or "longformer" in args.model_type else 1)
            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        #if "attn" not in task:
        all_ext_mask = torch.tensor([f.extraction_mask for f in features], dtype=torch.float)
        all_ext_start_ids = torch.tensor([f.extraction_start_ids for f in features], dtype=torch.long)
        all_ext_end_ids = torch.tensor([f.extraction_end_ids for f in features], dtype=torch.long)

        if "seq_label" in task:
            all_aug_seq_labels_mask = torch.tensor([f.augmentation_seq_labels_mask for f in features], dtype=torch.float)
            all_aug_seq_labels_ids = torch.tensor([f.augmentation_seq_label for f in features], dtype=torch.float).unsqueeze(-1)
        #elif "attn" in task:
        #    all_verb_attn_mask = torch.tensor([f.verb_attn_mask for f in features], dtype=torch.bool)
        #    all_cls_attn_mask = torch.tensor([f.cls_attn_mask for f in features], dtype=torch.bool)
        else:
            all_aug_mask = torch.tensor([f.augmentation_mask for f in features], dtype=torch.float)
            all_aug_start_ids = torch.tensor([f.augmentation_start_ids for f in features], dtype=torch.long)
            all_aug_end_ids = torch.tensor([f.augmentation_end_ids for f in features], dtype=torch.long)

        if "frank" in task:
            all_guids = torch.tensor([f.guid for f in features], dtype=torch.long)
            all_original_text_b_ids = torch.tensor([f.original_text_b_ids for f in features], dtype=torch.long)

        if output_mode == "classification":
            if model_type == "bertbinarylabelseqlabel":
                all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
            else:
                all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
            positive_weights = None
        elif output_mode == "multi_label_classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
            # compute positive weights for multi label classification
            with torch.no_grad():
                num_samples = all_label_ids.size(0)
                num_positive_samples_in_each_dim = all_label_ids.sum(0)  # [label_dim]
                num_negative_samples_in_each_dim = num_samples - num_positive_samples_in_each_dim
                positive_weights = num_negative_samples_in_each_dim/num_positive_samples_in_each_dim
                print("num_samples")
                print(num_samples)
                print("num_positive_samples_in_each_dim")
                print(num_positive_samples_in_each_dim)
                print("num_negative_samples_in_each_dim")
                print(num_negative_samples_in_each_dim)
                print("positive_weights")
                print(positive_weights)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
            positive_weights = None
        else:
            raise ValueError

        if task == "factcc_amr_aug_multi_label_seq_label_generated" or task == "factcc_multi_label_seq_label_generated" or task == "factcc_binary_label_seq_label_generated":
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                all_ext_mask, all_ext_start_ids, all_ext_end_ids,
                                all_aug_seq_labels_mask, all_aug_seq_labels_ids)
        #elif task == "factcc_multi_label_attn_generated":
        #    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
        #                            all_verb_attn_mask, all_cls_attn_mask)
        elif task == "frank_amr_multi_label_seq_label_annotated" or task == "frank_multi_label_srl_nested_seq_label_annotated":
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                    all_ext_mask, all_ext_start_ids, all_ext_end_ids,
                                    all_aug_seq_labels_mask, all_aug_seq_labels_ids, all_original_text_b_ids, all_guids)
        elif task == "frank_multi_label_annotated" or task == "frank_amr_multi_label_annotated" or task == "frank_multi_label_srl_annotated" or task == "frank_multi_label_srl_nested_annotated":
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                    all_ext_mask, all_ext_start_ids, all_ext_end_ids,
                                    all_aug_mask, all_aug_start_ids, all_aug_end_ids, all_original_text_b_ids, all_guids)
        else:
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                    all_ext_mask, all_ext_start_ids, all_ext_end_ids,
                                    all_aug_mask, all_aug_start_ids, all_aug_end_ids)
    return dataset, positive_weights


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_from_scratch", action='store_true',
			help="Whether to run training without loading pretrained weights.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--window_size", default=512, type=int,
                        help="Window size of longformer")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--resume_from_checkpoint", action='store_true',
                        help=".")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--loss_lambda", default=0.1, type=float,
                        help="The lambda parameter for loss mixing.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    parser.add_argument('--use_srl', action='store_true',
                        help="Store SRL special token")
    parser.add_argument('--keep_all_ckpts', action='store_true',
                        help="Store SRL special token")
    parser.add_argument('--use_srl_nested', action='store_true',
                        help="Store SRL special token")
    parser.add_argument('--export_output', action='store_true',
                        help="export the model output")
    parser.add_argument('--export_val_output', action='store_true',
                        help="export the model output")
    #parser.add_argument('--use_adapter', action='store_true',
    #                    help="Store SRL special token")
    parser.add_argument('--adapter_size', type=int, default=32,
                        help="")
    parser.add_argument('--n_claim_attn_heads', type=int, default=6,
                        help="")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    args.label_list = label_list
    num_labels = len(label_list)
    logger.info("label list: {}".format(label_list))

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    if args.model_type == "longformermultilabel" or args.model_type == "longformermultilabelseqlabel":
        config.attention_window = [args.window_size] * 12
        print(config.attention_window)
    if "adapter" in args.model_type:
        config.adapter_size = args.adapter_size
        logger.info("adapter size: {}".format(config.adapter_size))

    if "claimattn" in args.model_type:
        config.n_claim_attn_heads = args.n_claim_attn_heads
        logger.info("n_claim_attn_heads: {}".format(config.n_claim_attn_heads))

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    if "seq_label" in args.task_name:
        tokenizer.add_prefix_space = True
    if args.train_from_scratch:
        logger.info("Training model from scratch.")
        model = model_class(config=config)
    else:
        logger.info("Loading model from checkpoint.")
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # add special tokens
    if not args.resume_from_checkpoint:
        if args.use_srl:
            #ARG, VERB, ARGM-TMP, ARGM-LOC, ARGM, R-ARG, R-ARGM, R-ARGM-TMP, R-ARGM-LOC, C-ARG, C-ARGM
            special_tokens_dict = {'additional_special_tokens': ['[V]','[ARG]', '[VERB]', '[ARGM-TMP]', '[ARGM-LOC]', '[ARGM-NEG]', '[ARGM]', '[R-ARG]', '[R-ARGM]', '[C-ARG]', '[C-ARGM]', '[SRL-SEP]']}
        elif args.use_srl_nested:
            special_tokens_dict = {
                'additional_special_tokens': ['<V>', '<ARG>', '<ARGM-TMP>', '<ARGM-LOC>', '<ARGM-NEG>',
                                              '<ARGM>', '<R-ARG>', '<R-ARGM>', '<C-ARG>', '<C-ARGM>', '<P>']}
        else:
            special_tokens_dict = {'additional_special_tokens': ['ARG']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)

    wandb.watch(model)
    logger.info("Training/evaluation parameters %s", args)


    # Training
    if args.do_train:
        train_dataset, positive_weights = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        if positive_weights is not None:
            positive_weights = positive_weights.to(args.device)
            model.set_loss_pos_weight(positive_weights)
        print("data done")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)


    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()

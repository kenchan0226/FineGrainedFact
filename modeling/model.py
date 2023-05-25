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

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import math


#from pytorch_transformers.modeling_utils import (WEIGHTS_NAME, CONFIG_NAME, PretrainedConfig,
#                                                 PreTrainedModel, prune_linear_layer, add_start_docstrings)

#from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel, BertLayer, BertPooler
from transformers import BertPreTrainedModel, BertModel, BertAdapterModel
from transformers import LongformerPreTrainedModel, LongformerModel


class LongformerMultiLabelSeqLabelClassifier(LongformerPreTrainedModel):
    def __init__(self, config):
        super(LongformerMultiLabelSeqLabelClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.longformer = LongformerModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.node_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1, bias=False)
        )
        #self.apply(self.init_weights)
        #self.init_weights()
        self.post_init()
        print("init LongformerMultiLabelClassifier")

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, global_attention_mask=None,
                aug_seq_labels=None, aug_seq_labels_mask=None, loss_lambda=1.0):

        #print("aug_seq_labels")
        #print(aug_seq_labels.detach().cpu().numpy()[0])
        #print(aug_seq_labels_mask.detach().cpu().numpy()[0])
        #print(loss_lambda)
        #print(labels.size())

        # run through bert
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1
        longformer_outputs = self.longformer(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, head_mask=head_mask, global_attention_mask=global_attention_mask)
        pooled_output = longformer_outputs[1]

        # error type classifier
        pooled_output = self.dropout(pooled_output)
        label_logits = self.classifier(pooled_output)
        #print("pooled_output")
        #print(pooled_output.size())
        #print("label_logits")
        #print(label_logits.size())

        # sequence label classifier
        output = longformer_outputs[0]
        #print("output")
        #print(output.size())
        aug_seq_labels_mask = aug_seq_labels_mask.unsqueeze(-1)
        seq_labels_logits = self.node_classifier(output) * aug_seq_labels_mask

        outputs = (label_logits,) + (seq_labels_logits,) + longformer_outputs[2:]  # add hidden states and attention if they are here

        if labels is not None and aug_seq_labels is not None:
            # label loss
            labels_loss = self.criterion(label_logits, labels)

            # sequence labeling loss
            sequence_labels_loss = self.criterion(seq_labels_logits, aug_seq_labels)

            # combined loss
            loss = labels_loss + loss_lambda * sequence_labels_loss

            outputs = (loss, labels_loss, sequence_labels_loss) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertMultiLabelSeqLabelClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertMultiLabelSeqLabelClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.node_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1, bias=False)
        )
        #self.apply(self.init_weights)
        #self.init_weights()
        self.post_init()
        print("init BERTMultiLabelClassifier")

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, aug_seq_labels=None, aug_seq_labels_mask=None, loss_lambda=1.0):

        #print("aug_seq_labels")
        #print(aug_seq_labels.detach().cpu().numpy()[0])
        #print(aug_seq_labels_mask.detach().cpu().numpy()[0])
        #print(loss_lambda)
        #print(labels.size())

        # run through bert
        bert_outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = bert_outputs[1]

        # error type classifier
        pooled_output = self.dropout(pooled_output)
        label_logits = self.classifier(pooled_output)
        #print("pooled_output")
        #print(pooled_output.size())
        #print("label_logits")
        #print(label_logits.size())

        # sequence label classifier
        output = bert_outputs[0]
        #print("output")
        #print(output.size())
        aug_seq_labels_mask = aug_seq_labels_mask.unsqueeze(-1)
        seq_labels_logits = self.node_classifier(output) * aug_seq_labels_mask

        outputs = (label_logits,) + (seq_labels_logits,) + bert_outputs[2:]  # add hidden states and attention if they are here

        if labels is not None and aug_seq_labels is not None:
            # label loss
            labels_loss = self.criterion(label_logits, labels)

            # sequence labeling loss
            sequence_labels_loss = self.criterion(seq_labels_logits, aug_seq_labels)

            # combined loss
            loss = labels_loss + loss_lambda * sequence_labels_loss

            outputs = (loss, labels_loss, sequence_labels_loss) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertBinaryLabelSeqLabelClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertBinaryLabelSeqLabelClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.binary_classifier = nn.Linear(config.hidden_size, 1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.node_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 1)
        )
        #self.apply(self.init_weights)
        #self.init_weights()
        self.post_init()
        print("init BERTBinaryLabelClassifier")

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, aug_seq_labels=None, aug_seq_labels_mask=None, loss_lambda=1.0):
        """
        print("aug_seq_labels")
        print(aug_seq_labels.size())
        print("labels")
        print(labels.size())
        print(labels.type())
        """

        # run through bert
        bert_outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = bert_outputs[1]

        # error type classifier
        pooled_output = self.dropout(pooled_output)
        label_logits = self.binary_classifier(pooled_output)
        """
        print("pooled_output")
        print(pooled_output.size())
        print("label_logits")
        print(label_logits.size())
        """

        # sequence label classifier
        output = bert_outputs[0]
        #print("output")
        #print(output.size())
        aug_seq_labels_mask = aug_seq_labels_mask.unsqueeze(-1)
        seq_labels_logits = self.node_classifier(self.dropout(output)) * aug_seq_labels_mask
        """
        print("seq_labels_logits")
        print(seq_labels_logits.size())
        """
        outputs = (label_logits,) + (seq_labels_logits,) + bert_outputs[2:]  # add hidden states and attention if they are here

        if labels is not None and aug_seq_labels is not None:
            # label loss
            labels_loss = self.criterion(label_logits.squeeze(), labels.squeeze())

            # sequence labeling loss
            sequence_labels_loss = self.criterion(seq_labels_logits, aug_seq_labels)

            # combined loss
            loss = (1- loss_lambda) * labels_loss + loss_lambda * sequence_labels_loss

            outputs = (loss, labels_loss, sequence_labels_loss) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertIntermediate(BertPreTrainedModel):
    def __init__(self, config):
        super(BertIntermediate, self).__init__(config)
        self.bert = BertModel(config)
        #self.apply(self.init_weights)
        #self.init_weights()
        self.post_init()
        print("init BertIntermediate")


class LongformerMultiLabelClassifier(LongformerPreTrainedModel):
    def __init__(self, config):
        super(LongformerMultiLabelClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.longformer = LongformerModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()

        #self.apply(self.init_weights)
        #self.init_weights()
        self.post_init()
        print("init LongformerMultiLabelClassifier")

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, global_attention_mask=None):
        # run through bert
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1
        outputs = self.longformer(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, head_mask=head_mask, global_attention_mask=global_attention_mask)
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.criterion(logits, labels)
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertMultiLabelClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertMultiLabelClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        config.output_attentions = True
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()

        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        # run through bert
        #print("input_ids")
        #print(input_ids.size())
        #print("token_type_ids")
        #print(token_type_ids.size())
        #print(token_type_ids[0].detach().cpu().numpy())
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]
        attentions = outputs[2]  # [batch, n_head, seq_len, seq_len]
        total_attn_from_cls = torch.sum(attentions[-1][:,:,0,:], dim=1)  # [batch, seq_len]
        assert total_attn_from_cls.size() == torch.Size([attentions[-1].size(0), attentions[-1].size(-1)])
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits, total_attn_from_cls) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            """
            print("logits")
            print(logits.detach().cpu().numpy())
            print("labels")
            print(labels.detach().cpu().numpy())
            """
            loss = self.criterion(logits, labels)
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertMultiLabelAdapterClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertMultiLabelAdapterClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        config.output_attentions = True
        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()

        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        # run through bert
        #print("input_ids")
        #print(input_ids.size())
        #print("token_type_ids")
        #print(token_type_ids.size())
        #print(token_type_ids[0].detach().cpu().numpy())
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]
        attentions = outputs[2]  # tuple of [batch, n_head, seq_len, seq_len]
        total_attn_from_cls = torch.sum(attentions[-1][:, :, 0, :], dim=1)  # [batch, seq_len]
        assert total_attn_from_cls.size() == torch.Size([attentions[-1].size(0), attentions[-1].size(-1)])
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits, total_attn_from_cls) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            """
            print("logits")
            print(logits.detach().cpu().numpy())
            print("labels")
            print(labels.detach().cpu().numpy())
            """
            #print("positive_weights")
            #print(positive_weights.detach().cpu().numpy())
            #exit()
            loss = self.criterion(logits, labels)
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertSRLAttnMultiLabelClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSRLAttnMultiLabelClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        #nn.init.xavier_uniform_(self.classifier.weight)
        self.criterion = nn.BCEWithLogitsLoss()

        self.num_verb_attn_heads = 6
        self.verb_attention = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_verb_attn_heads)
        self.cls_attention = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_verb_attn_heads)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        #self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertSRLAttnMultiLabelClassifier")


    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())


    def forward(self, input_ids, verb_attn_mask, cls_attn_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        # run through bert
        #print("input_ids")
        #print(input_ids.size())
        #print("token_type_ids")
        #print(token_type_ids.size())
        #print(token_type_ids[0].detach().cpu().numpy())
        # verb_attn_mask [batch, seq_len, seq_len]
        # cls_attn_mask [batch, seq_len]
        #print("verb_attn_mask")
        #print(verb_attn_mask.size())
        #print(verb_attn_mask[0,0].detach().cpu().numpy())
        #print("cls_attn_mask")
        #print(cls_attn_mask.size())
        #print(cls_attn_mask[0].detach().cpu().numpy())
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, head_mask=head_mask)
        cls_output = outputs[1]  # [batch, hidden_size]
        #print("cls_output")
        #print(cls_output.size())

        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        last_hidden_state = torch.transpose(last_hidden_state, 0, 1)
        #batch_size, seq_len, _ = verb_attn_mask.size()
        #verb_attn_mask = verb_attn_mask.expand(batch_size * self.num_verb_attn_heads, -1, -1)
        verb_attn_mask = verb_attn_mask.repeat(self.num_verb_attn_heads, 1, 1)
        #verb_attn_mask = torch.transpose(verb_attn_mask, 0, 1)
        # batch first?
        #print("last_hidden_state")
        #print(last_hidden_state.size())
        #print("verb_attn_mask")
        #print(verb_attn_mask.size())
        verb_attn_output, verb_attn_output_weights = self.verb_attention(last_hidden_state, last_hidden_state,
                                                                         last_hidden_state, attn_mask=verb_attn_mask)
        verb_attn_output = self.layer_norm(self.dropout(verb_attn_output) + last_hidden_state)
        # verb_attn_output: [seq_len, batch_size, hidden_size]

        cls_output_ = cls_output.unsqueeze(0)  # [1, batch, hidden_size]

        # concat cls representation with verb attn output
        verb_attn_output_ = torch.cat([cls_output_, verb_attn_output[1:, :, :]], dim=0)

        cls_attn_output, cls_attn_output_weights = self.cls_attention(cls_output_, verb_attn_output_, verb_attn_output_, key_padding_mask=cls_attn_mask)
        # cls_attn_output  [1, batch, hidden_size]
        #print("cls_attn_output")
        #print(cls_attn_output.size())
        #print(cls_attn_output[0,0].detach().cpu().numpy())
        #exit()
        cls_attn_output = self.layer_norm(self.dropout(cls_attn_output.squeeze(0))) + self.dropout(cls_output)
        #print("cls_attn_output")
        #print(cls_attn_output.size())
        #print(cls_attn_output[0].detach().cpu().numpy())
        logits = self.classifier(cls_attn_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            #print("logits")
            #print(logits.detach().cpu().numpy())
            #print("labels")
            #print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            #print("loss")
            #print(loss.detach().cpu().numpy())
            #exit()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertSRLAttnMultiLabelAdapterClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSRLAttnMultiLabelAdapterClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        #nn.init.xavier_uniform_(self.classifier.weight)
        self.criterion = nn.BCEWithLogitsLoss()

        self.num_verb_attn_heads = 6
        self.verb_attention = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_verb_attn_heads)
        self.cls_attention = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_verb_attn_heads)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        #self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertSRLAttnMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, verb_attn_mask, cls_attn_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, head_mask=head_mask)
        cls_output = outputs[1]  # [batch, hidden_size]
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        last_hidden_state = torch.transpose(last_hidden_state, 0, 1)
        #batch_size, seq_len, _ = verb_attn_mask.size()
        #verb_attn_mask = verb_attn_mask.expand(batch_size * self.num_verb_attn_heads, -1, -1)
        verb_attn_mask = verb_attn_mask.repeat(self.num_verb_attn_heads, 1, 1)
        #verb_attn_mask = torch.transpose(verb_attn_mask, 0, 1)
        # batch first?
        #print("last_hidden_state")
        #print(last_hidden_state.size())
        #print("verb_attn_mask")
        #print(verb_attn_mask.size())
        verb_attn_output, verb_attn_output_weights = self.verb_attention(last_hidden_state, last_hidden_state,
                                                                         last_hidden_state, attn_mask=verb_attn_mask)
        verb_attn_output = self.layer_norm(self.dropout(verb_attn_output) + last_hidden_state)
        # verb_attn_output: [seq_len, batch_size, hidden_size]

        cls_output_ = cls_output.unsqueeze(0)  # [1, batch, hidden_size]

        # concat cls representation with verb attn output
        verb_attn_output_ = torch.cat([cls_output_, verb_attn_output[1:, :, :]], dim=0)

        cls_attn_output, cls_attn_output_weights = self.cls_attention(cls_output_, verb_attn_output_, verb_attn_output_, key_padding_mask=cls_attn_mask)
        # cls_attn_output  [1, batch, hidden_size]
        #print("cls_attn_output")
        #print(cls_attn_output.size())
        #print(cls_attn_output[0,0].detach().cpu().numpy())
        #exit()
        cls_attn_output = self.layer_norm(self.dropout(cls_attn_output.squeeze(0))) + self.dropout(cls_output)
        #print("cls_attn_output")
        #print(cls_attn_output.size())
        #print(cls_attn_output[0].detach().cpu().numpy())
        logits = self.classifier(cls_attn_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            #print("logits")
            #print(logits.detach().cpu().numpy())
            #print("labels")
            #print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            #print("loss")
            #print(loss.detach().cpu().numpy())
            #exit()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMilMultiLabelClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMilMultiLabelClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.classifier_attn = nn.Linear(config.hidden_size * 3, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax = MaskedSoftmax(dim=1)

        self.num_attn_heads = 6
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilMultiLabelClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]

        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        #last_hidden_state_projected = self.frame_feature_project(
        #    last_hidden_state)  # [batch_size, sequence_length, hidden_size]

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        doc_frame_features_sum = torch.bmm(doc_frames_word_mask,
                                           last_hidden_state)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
            mean_pool_mask]  # [batch_size, n_frames, hid]

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)
        claim_frame_features_sum = torch.bmm(claim_frames_word_mask,
                                             last_hidden_state)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words = num_claim_frame_words.expand(-1, -1,
                                                             claim_frame_features_sum.size(
                                                                 2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_claim_frame_words != 0)
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding

        claim_attn_output = self.claim_frame_layer_norm(claim_attn_output)

        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features], dim=2)  #  [batch, n_claim_frames, 2*hid_size]
        claim_frame_logits = self.classifier(self.dropout(frame_classification_features))  # [batch, n_claim_frames, num_classes]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]
        # expand cls representation
        cls_representation_expanded = cls_representation.unsqueeze(1).expand(-1, claim_frame_features.size(1), -1)  # [batch, n_claim_frames, hid_size]

        classifier_attn_weights = self.classifier_attn(self.dropout(torch.cat([cls_representation_expanded, frame_classification_features], dim=2)))  # [batch, n_claim_frames, num_classes]
        # claim_frames_padding_mask [batch, n_claim_frames]
        claim_frames_padding_mask_expanded = claim_frames_padding_mask.unsqueeze(2).expand(-1, -1, classifier_attn_weights.size(2))
        # claim_frames_padding_mask [batch, n_claim_frames, num_classes]
        classifier_attn_weights = self.masked_softmax(classifier_attn_weights, mask=claim_frames_padding_mask_expanded)  # [batch, n_claim_frames, num_classes]
        logits = (claim_frame_logits * classifier_attn_weights).sum(dim=1)  # [batch, num_classes]

        #logits = self.classifier(claim_attn_output_mean_pooled)
        # [batch, n_claim_frames, hid] ->  # [batch, hid]

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMilMultiLabelAdapterClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMilMultiLabelAdapterClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.classifier_attn = nn.Linear(config.hidden_size * 3, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax = MaskedSoftmax(dim=1)

        self.num_attn_heads = 6
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]

        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        #last_hidden_state_projected = self.frame_feature_project(
        #    last_hidden_state)  # [batch_size, sequence_length, hidden_size]

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        doc_frame_features_sum = torch.bmm(doc_frames_word_mask,
                                           last_hidden_state)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
            mean_pool_mask]  # [batch_size, n_frames, hid]

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)
        claim_frame_features_sum = torch.bmm(claim_frames_word_mask,
                                             last_hidden_state)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words = num_claim_frame_words.expand(-1, -1,
                                                             claim_frame_features_sum.size(
                                                                 2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_claim_frame_words != 0)
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        """
        print("claim_attn_mask")  # [batch_size, n_doc_frames]
        """

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding
        """
        claim_attn_output_nan_mask = claim_frames_padding_mask.unsqueeze(2).expand(-1, -1, claim_attn_output.size(2))  # [batch_size, n_claim_frames, hid]
        claim_attn_output_nan_mask = (claim_attn_output_nan_mask != 0)
        claim_attn_output_no_nan = torch.zeros_like(claim_attn_output)
        claim_attn_output_no_nan[claim_attn_output_nan_mask] = claim_attn_output[claim_attn_output_nan_mask]
        claim_attn_output_no_nan = self.layer_norm(self.dropout(claim_attn_output_no_nan) + claim_frame_features)
        """

        """
        print("claim_attn_output_no_nan")
        print(claim_attn_output_no_nan[0, 0, :].detach().cpu().numpy())
        print(claim_attn_output_no_nan[0, 1, :].detach().cpu().numpy())
        print(claim_attn_output_no_nan[0, 2, :].detach().cpu().numpy())
        print(claim_attn_output_no_nan[0, 3, :].detach().cpu().numpy())
        print()
        print()
        """
        claim_attn_output = self.claim_frame_layer_norm(claim_attn_output)

        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features], dim=2)  #  [batch, n_claim_frames, 2*hid_size]
        claim_frame_logits = self.classifier(self.dropout(frame_classification_features))  # [batch, n_claim_frames, num_classes]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]
        # expand cls representation
        cls_representation_expanded = cls_representation.unsqueeze(1).expand(-1, claim_frame_features.size(1), -1)  # [batch, n_claim_frames, hid_size]

        classifier_attn_weights = self.classifier_attn(self.dropout(torch.cat([cls_representation_expanded, frame_classification_features], dim=2)))  # [batch, n_claim_frames, num_classes]
        # claim_frames_padding_mask [batch, n_claim_frames]
        claim_frames_padding_mask_expanded = claim_frames_padding_mask.unsqueeze(2).expand(-1, -1, classifier_attn_weights.size(2))
        # claim_frames_padding_mask [batch, n_claim_frames, num_classes]
        classifier_attn_weights = self.masked_softmax(classifier_attn_weights, mask=claim_frames_padding_mask_expanded)  # [batch, n_claim_frames, num_classes]
        logits = (claim_frame_logits * classifier_attn_weights).sum(dim=1)  # [batch, num_classes]

        #logits = self.classifier(claim_attn_output_mean_pooled)
        # [batch, n_claim_frames, hid] ->  # [batch, hid]

        outputs = (logits, claim_frame_logits, classifier_attn_weights, claim_attn_output_weights) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimMultiHeadAttnMilMultiLabelAdapterClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimMultiHeadAttnMilMultiLabelAdapterClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.classifier_attn = nn.Linear(config.hidden_size * 3, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax = MaskedSoftmax(dim=1)

        self.num_attn_heads = 6
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        """
        print("last_hidden_state")
        print(last_hidden_state[0, 0, :])
        print(last_hidden_state[0, 1, :])
        print(last_hidden_state[0, -1, :])
        """

        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        last_hidden_state_projected = self.frame_feature_project(
            last_hidden_state)  # [batch_size, sequence_length, hidden_size]

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        """
        print("doc_frames_word_mask")
        print(doc_frames_word_mask[0,0,:])
        print(doc_frames_word_mask[0,1,:])
        print("non empty doc frames")
        print((num_doc_frame_words[0, :, 0] > 0).sum())
        print(num_doc_frame_words[0, : 0])
        """

        doc_frame_features_sum = torch.bmm(doc_frames_word_mask,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
            mean_pool_mask]  # [batch_size, n_frames, hid]

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)
        claim_frame_features_sum = torch.bmm(claim_frames_word_mask,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words = num_claim_frame_words.expand(-1, -1,
                                                             claim_frame_features_sum.size(
                                                                 2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_claim_frame_words != 0)
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        """
        print("claim_attn_mask")  # [batch_size, n_doc_frames]
        print(claim_attn_mask.size())
        print(claim_attn_mask[0, :])
        print("num_claim_frame_words")
        print(num_claim_frame_words[0, 0, 0])
        print(num_claim_frame_words[0, 1, 0])
        print(num_claim_frame_words[0, 2, 0])
        print("claim_frame_features")
        print(claim_frame_features[0, 0, :])
        print(claim_frame_features[0, 1, :])
        print(claim_frame_features[0, 2, :])
        print(claim_frame_features[0, -1, :])
        """

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding
        """
        claim_attn_output_nan_mask = claim_frames_padding_mask.unsqueeze(2).expand(-1, -1, claim_attn_output.size(2))  # [batch_size, n_claim_frames, hid]
        claim_attn_output_nan_mask = (claim_attn_output_nan_mask != 0)
        claim_attn_output_no_nan = torch.zeros_like(claim_attn_output)
        claim_attn_output_no_nan[claim_attn_output_nan_mask] = claim_attn_output[claim_attn_output_nan_mask]
        claim_attn_output_no_nan = self.layer_norm(self.dropout(claim_attn_output_no_nan) + claim_frame_features)
        """

        """
        print("claim_attn_output_no_nan")
        print(claim_attn_output_no_nan[0, 0, :].detach().cpu().numpy())
        print(claim_attn_output_no_nan[0, 1, :].detach().cpu().numpy())
        print(claim_attn_output_no_nan[0, 2, :].detach().cpu().numpy())
        print(claim_attn_output_no_nan[0, 3, :].detach().cpu().numpy())
        print()
        print()
        """

        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features], dim=2)  #  [batch, n_claim_frames, 2*hid_size]
        claim_frame_logits = self.classifier(self.dropout(frame_classification_features))  # [batch, n_claim_frames, num_classes]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]
        # expand cls representation
        cls_representation_expanded = cls_representation.unsqueeze(1).expand(-1, claim_frame_features.size(1), -1)  # [batch, n_claim_frames, hid_size]

        classifier_attn_weights = self.classifier_attn(self.dropout(torch.cat([cls_representation_expanded, frame_classification_features], dim=2)))  # [batch, n_claim_frames, num_classes]
        # claim_frames_padding_mask [batch, n_claim_frames]
        claim_frames_padding_mask_expanded = claim_frames_padding_mask.unsqueeze(2).expand(-1, -1, classifier_attn_weights.size(2))
        # claim_frames_padding_mask [batch, n_claim_frames, num_classes]
        classifier_attn_weights = self.masked_softmax(classifier_attn_weights, mask=claim_frames_padding_mask_expanded)  # [batch, n_claim_frames, num_classes]
        logits = (claim_frame_logits * classifier_attn_weights).sum(dim=1)  # [batch, num_classes]

        #logits = self.classifier(claim_attn_output_mean_pooled)
        # [batch, n_claim_frames, hid] ->  # [batch, hid]

        outputs = (logits, claim_frame_logits) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMilCosSimMultiLabelClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMilCosSimMultiLabelClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2 + 1, self.num_labels)
        self.classifier_attn = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax = MaskedSoftmax(dim=1)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilCosSimMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]

        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        last_hidden_state_projected = self.frame_feature_project(
            last_hidden_state)  # [batch_size, sequence_length, hidden_size]

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        doc_frame_features_sum = torch.bmm(doc_frames_word_mask,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
            mean_pool_mask]  # [batch_size, n_frames, hid]

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]
        claim_frame_features_sum = torch.bmm(claim_frames_word_mask,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                             claim_frame_features_sum.size(
                                                                 2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding

        # features of cosine similarity
        last_hidden_state_normalized = nn.functional.normalize(last_hidden_state, p=2, dim=2)
        similarity_matrix = torch.bmm(last_hidden_state_normalized,  torch.transpose(last_hidden_state_normalized, 1, 2))  # [batch, seq_len, seq_len]

        #similarity_matrix = self.cos_sim(last_hidden_state, last_hidden_state)  # [batch, seq_len, seq_len]
        # doc_words_mask [batch, seq_len]
        doc_words_mask_expanded = doc_words_mask.unsqueeze(1).expand(-1, similarity_matrix.size(1), -1)  # [batch, seq_len,seq_len]

        similarity_matrix_masked = similarity_matrix * doc_words_mask_expanded  # [batch, seq_len, seq_len]
        similarity_max_pooled, _ = similarity_matrix_masked.max(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        # claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        claim_words_similarities_sum = torch.bmm(claim_frames_word_mask, similarity_max_pooled).squeeze(2)  # [batch, n_claim_frames]
        # mean_pool_mask [batch, n_claim_frames, hid]
        mean_pool_mask_shrinked = mean_pool_mask[:, :, 0]  # [batch, n_claim_frames]
        claim_words_precision = torch.zeros_like(claim_words_similarities_sum)  # [batch, n_claim_frames]
        # num_claim_frame_words [batch, n_claim_frames, 1]

        claim_words_precision[mean_pool_mask_shrinked] = claim_words_similarities_sum[mean_pool_mask_shrinked] / num_claim_frame_words.squeeze(-1)[
            mean_pool_mask_shrinked]  # [batch, n_claim_frames]

        claim_words_precision = claim_words_precision.unsqueeze(2)  # [batch, n_claim_frames, 1]

        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)

        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features, claim_words_precision], dim=2)  #  [batch, n_claim_frames, 2*hid_size+1]
        #claim_frame_logits = self.classifier(self.claim_frame_layer_norm( self.dropout(frame_classification_features) ))  # [batch, n_claim_frames, num_classes]
        claim_frame_logits = self.classifier( self.dropout(frame_classification_features) )  # [batch, n_claim_frames, num_classes]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]
        # expand cls representation
        cls_representation_expanded = cls_representation.unsqueeze(1).expand(-1, claim_frame_features.size(1), -1)  # [batch, n_claim_frames, hid_size]

        #classifier_attn_weights = self.classifier_attn(self.claim_frame_layer_norm( self.dropout(torch.cat([cls_representation_expanded, frame_classification_features], dim=2)) ))  # [batch, n_claim_frames, num_classes]
        classifier_attn_weights = self.classifier_attn( self.dropout(torch.cat([cls_representation_expanded, frame_classification_features], dim=2)) )  # [batch, n_claim_frames, num_classes]
        # claim_frames_padding_mask [batch, n_claim_frames]
        claim_frames_padding_mask_expanded = claim_frames_padding_mask.unsqueeze(2).expand(-1, -1, classifier_attn_weights.size(2))
        # claim_frames_padding_mask [batch, n_claim_frames, num_classes]
        classifier_attn_weights = self.masked_softmax(classifier_attn_weights, mask=claim_frames_padding_mask_expanded)  # [batch, n_claim_frames, num_classes]
        logits = (claim_frame_logits * classifier_attn_weights).sum(dim=1)  # [batch, num_classes]

        #logits = self.classifier(claim_attn_output_mean_pooled)
        # [batch, n_claim_frames, hid] ->  # [batch, hid]

        outputs = (logits, claim_frame_logits, classifier_attn_weights, claim_attn_output_weights) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMilCosSimMultiLabelAdapterClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMilCosSimMultiLabelAdapterClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2 + 1, self.num_labels)
        self.classifier_attn = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax = MaskedSoftmax(dim=1)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilCosSimMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        """
        print("last_hidden_state")
        print(last_hidden_state[0, 0, :])
        print(last_hidden_state[0, 1, :])
        print(last_hidden_state[0, -1, :])
        """

        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        last_hidden_state_projected = self.frame_feature_project(
            last_hidden_state)  # [batch_size, sequence_length, hidden_size]

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        """
        print("doc_frames_word_mask")
        print(doc_frames_word_mask[0,0,:])
        print(doc_frames_word_mask[0,1,:])
        print("non empty doc frames")
        print((num_doc_frame_words[0, :, 0] > 0).sum())
        print(num_doc_frame_words[0, : 0])
        """

        doc_frame_features_sum = torch.bmm(doc_frames_word_mask,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
            mean_pool_mask]  # [batch_size, n_frames, hid]

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]
        claim_frame_features_sum = torch.bmm(claim_frames_word_mask,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                             claim_frame_features_sum.size(
                                                                 2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        """
        print("claim_attn_mask")  # [batch_size, n_doc_frames]
        print(claim_attn_mask.size())
        print(claim_attn_mask[0, :])
        print("num_claim_frame_words")
        print(num_claim_frame_words[0, 0, 0])
        print(num_claim_frame_words[0, 1, 0])
        print(num_claim_frame_words[0, 2, 0])
        print("claim_frame_features")
        print(claim_frame_features[0, 0, :])
        print(claim_frame_features[0, 1, :])
        print(claim_frame_features[0, 2, :])
        print(claim_frame_features[0, -1, :])
        """

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding

        # features of cosine similarity
        last_hidden_state_normalized = nn.functional.normalize(last_hidden_state, p=2, dim=2)
        similarity_matrix = torch.bmm(last_hidden_state_normalized,  torch.transpose(last_hidden_state_normalized, 1, 2))  # [batch, seq_len, seq_len]

        #similarity_matrix = self.cos_sim(last_hidden_state, last_hidden_state)  # [batch, seq_len, seq_len]
        # doc_words_mask [batch, seq_len]
        doc_words_mask_expanded = doc_words_mask.unsqueeze(1).expand(-1, similarity_matrix.size(1), -1)  # [batch, seq_len,seq_len]
        # inverted and mask to negative number
        doc_words_mask_expanded_inverted = doc_words_mask_expanded < 0.5 * -10000.0

        # similarity_matrix_masked = similarity_matrix * doc_words_mask_expanded  # [batch, seq_len, seq_len]
        similarity_matrix_masked = similarity_matrix + doc_words_mask_expanded_inverted  # [batch, seq_len, seq_len]
        similarity_max_pooled, _ = similarity_matrix_masked.max(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        # claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        claim_words_similarities_sum = torch.bmm(claim_frames_word_mask, similarity_max_pooled).squeeze(2)  # [batch, n_claim_frames]
        # mean_pool_mask [batch, n_claim_frames, hid]
        mean_pool_mask_shrinked = mean_pool_mask[:, :, 0]  # [batch, n_claim_frames]
        claim_words_precision = torch.zeros_like(claim_words_similarities_sum)  # [batch, n_claim_frames]
        # num_claim_frame_words [batch, n_claim_frames, 1]
        """
        print("similarity_max_pooled")
        print(similarity_max_pooled.size())
        print(similarity_max_pooled[0])
        print("claim_words_similarities_sum")
        print(claim_words_similarities_sum.size())
        print(claim_words_similarities_sum[0])
        print("claim_words_precision")
        print(claim_words_precision.size())
        print("mean_pool_mask_shrinked")
        print(mean_pool_mask_shrinked.size())
        print(mean_pool_mask_shrinked[0])
        """
        claim_words_precision[mean_pool_mask_shrinked] = claim_words_similarities_sum[mean_pool_mask_shrinked] / num_claim_frame_words.squeeze(-1)[
            mean_pool_mask_shrinked]  # [batch, n_claim_frames]
        """
        print("claim_words_precision")
        print(claim_words_precision.size())
        print(claim_words_precision[0])
        """
        claim_words_precision = claim_words_precision.unsqueeze(2)  # [batch, n_claim_frames, 1]

        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features, claim_words_precision], dim=2)  #  [batch, n_claim_frames, 2*hid_size+1]
        claim_frame_logits = self.classifier(self.dropout(frame_classification_features))  # [batch, n_claim_frames, num_classes]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]
        # expand cls representation
        cls_representation_expanded = cls_representation.unsqueeze(1).expand(-1, claim_frame_features.size(1), -1)  # [batch, n_claim_frames, hid_size]

        classifier_attn_weights = self.classifier_attn(self.dropout(torch.cat([cls_representation_expanded, frame_classification_features], dim=2)))  # [batch, n_claim_frames, num_classes]
        # claim_frames_padding_mask [batch, n_claim_frames]
        claim_frames_padding_mask_expanded = claim_frames_padding_mask.unsqueeze(2).expand(-1, -1, classifier_attn_weights.size(2))
        # claim_frames_padding_mask [batch, n_claim_frames, num_classes]
        classifier_attn_weights = self.masked_softmax(classifier_attn_weights, mask=claim_frames_padding_mask_expanded)  # [batch, n_claim_frames, num_classes]
        logits = (claim_frame_logits * classifier_attn_weights).sum(dim=1)  # [batch, num_classes]

        #logits = self.classifier(claim_attn_output_mean_pooled)
        # [batch, n_claim_frames, hid] ->  # [batch, hid]

        outputs = (logits, claim_frame_logits, classifier_attn_weights, claim_attn_output_weights) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMeanCosSimMultiLabelAdapterClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMeanCosSimMultiLabelAdapterClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        #self.classifier_attn = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax = MaskedSoftmax(dim=1)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        #self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilCosSimMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        """
        print("last_hidden_state")
        print(last_hidden_state[0, 0, :])
        print(last_hidden_state[0, 1, :])
        print(last_hidden_state[0, -1, :])
        """

        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        last_hidden_state_projected = self.frame_feature_project(
            last_hidden_state)  # [batch_size, sequence_length, hidden_size]

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        """
        print("doc_frames_word_mask")
        print(doc_frames_word_mask[0,0,:])
        print(doc_frames_word_mask[0,1,:])
        print("non empty doc frames")
        print((num_doc_frame_words[0, :, 0] > 0).sum())
        print(num_doc_frame_words[0, : 0])
        """

        doc_frame_features_sum = torch.bmm(doc_frames_word_mask,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
            mean_pool_mask]  # [batch_size, n_frames, hid]

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]
        claim_frame_features_sum = torch.bmm(claim_frames_word_mask,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                             claim_frame_features_sum.size(
                                                                 2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        """
        print("claim_attn_mask")  # [batch_size, n_doc_frames]
        print(claim_attn_mask.size())
        print(claim_attn_mask[0, :])
        print("num_claim_frame_words")
        print(num_claim_frame_words[0, 0, 0])
        print(num_claim_frame_words[0, 1, 0])
        print(num_claim_frame_words[0, 2, 0])
        print("claim_frame_features")
        print(claim_frame_features[0, 0, :])
        print(claim_frame_features[0, 1, :])
        print(claim_frame_features[0, 2, :])
        print(claim_frame_features[0, -1, :])
        """

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding

        # features of cosine similarity
        last_hidden_state_normalized = nn.functional.normalize(last_hidden_state, p=2, dim=2)
        similarity_matrix = torch.bmm(last_hidden_state_normalized,  torch.transpose(last_hidden_state_normalized, 1, 2))  # [batch, seq_len, seq_len]

        #similarity_matrix = self.cos_sim(last_hidden_state, last_hidden_state)  # [batch, seq_len, seq_len]
        # doc_words_mask [batch, seq_len]
        doc_words_mask_expanded = doc_words_mask.unsqueeze(1).expand(-1, similarity_matrix.size(1), -1)  # [batch, seq_len,seq_len]
        # inverted and mask to negative number
        doc_words_mask_expanded_inverted = doc_words_mask_expanded < 0.5 * -10000.0

        # similarity_matrix_masked = similarity_matrix * doc_words_mask_expanded  # [batch, seq_len, seq_len]
        similarity_matrix_masked = similarity_matrix + doc_words_mask_expanded_inverted  # [batch, seq_len, seq_len]
        similarity_max_pooled, _ = similarity_matrix_masked.max(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        # claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        claim_words_similarities_sum = torch.bmm(claim_frames_word_mask, similarity_max_pooled).squeeze(2)  # [batch, n_claim_frames]
        # mean_pool_mask [batch, n_claim_frames, hid]
        mean_pool_mask_shrinked = mean_pool_mask[:, :, 0]  # [batch, n_claim_frames]
        claim_words_precision = torch.zeros_like(claim_words_similarities_sum)  # [batch, n_claim_frames]
        # num_claim_frame_words [batch, n_claim_frames, 1]
        """
        print("similarity_max_pooled")
        print(similarity_max_pooled.size())
        print(similarity_max_pooled[0])
        print("claim_words_similarities_sum")
        print(claim_words_similarities_sum.size())
        print(claim_words_similarities_sum[0])
        print("claim_words_precision")
        print(claim_words_precision.size())
        print("mean_pool_mask_shrinked")
        print(mean_pool_mask_shrinked.size())
        print(mean_pool_mask_shrinked[0])
        """
        claim_words_precision[mean_pool_mask_shrinked] = claim_words_similarities_sum[mean_pool_mask_shrinked] / num_claim_frame_words.squeeze(-1)[
            mean_pool_mask_shrinked]  # [batch, n_claim_frames]
        """
        print("claim_words_precision")
        print(claim_words_precision.size())
        print(claim_words_precision[0])
        """
        claim_words_precision = claim_words_precision.unsqueeze(2)  # [batch, n_claim_frames, 1]

        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features, claim_words_precision], dim=2)  #  [batch, n_claim_frames, 2*hid_size+1]

        frame_classification_features_mean_pooled = frame_classification_features.mean(dim=1)  #  [batch, 2*hid_size+1]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]

        all_classification_feature = torch.cat([cls_representation, frame_classification_features_mean_pooled], dim=1)  #  [batch, 3*hid_size+1]

        logits = self.classifier(self.dropout(all_classification_feature))  # [batch, num_classes]

        outputs = (logits, claim_attn_output_weights) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMeanCosSimWordAttnMultiLabelAdapterClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMeanCosSimWordAttnMultiLabelAdapterClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        config.output_hidden_states = True
        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        #self.classifier_attn = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax_dim_1 = MaskedSoftmax(dim=1)
        self.masked_softmax_dim_2 = MaskedSoftmax(dim=2)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        #self.layer_attn = nn.Linear(config.hidden_size, 1)
        self.token_self_attn = nn.Linear(config.hidden_size, 1)

        #self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilCosSimMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]


        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        last_hidden_state_projected = self.frame_feature_project(
            last_hidden_state)  # [batch_size, sequence_length, hidden_size]

        token_attn_score = self.token_self_attn(last_hidden_state_projected).squeeze()  # [batch_size, sequence_length]

        token_attn_score_resized_for_doc_frames = token_attn_score.unsqueeze(1).expand(-1, doc_frames_word_mask.size(1),
                                                                                       -1)  # [batch, n_doc_frames, seq_len]
        doc_frames_word_mask_multiplied = doc_frames_word_mask * token_attn_score_resized_for_doc_frames  # [batch, n_doc_frames, seq_len]
        # doc_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(doc_frames_word_mask_multiplied, -1)  # [batch, n_doc_frames, seq_len]

        doc_frames_word_mask_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)
        doc_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        num_doc_frame_words_expanded_seq_len = num_doc_frame_words.expand(-1, -1,
                                                                          doc_frames_word_mask.size(
                                                                              2))  # [batch, n_claim_frames, seq_len]

        doc_frames_word_mask_all_zero_mask = (
                num_doc_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        doc_frames_word_mask_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = doc_frames_word_mask[
            doc_frames_word_mask_all_zero_mask]
        doc_frames_word_mask_multiplied_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = \
            doc_frames_word_mask_multiplied[doc_frames_word_mask_all_zero_mask]

        doc_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            doc_frames_word_mask_multiplied_fixed_all_zeros,
            doc_frames_word_mask_fixed_all_zeros)  # [batch, n_doc_frames, seq_len]


        doc_frame_features_sum = torch.bmm(doc_frames_word_mask_multiplied_softmaxed,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask]
        # doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
        #    mean_pool_mask]  # [batch_size, n_frames, hid]

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        token_attn_score_resized_for_claim_frames = token_attn_score.unsqueeze(1).expand(-1,
                                                                                         claim_frames_word_mask.size(1),
                                                                                         -1)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_multiplied = claim_frames_word_mask * token_attn_score_resized_for_claim_frames  # [batch, n_claim_frames, seq_len]
        # claim_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(claim_frames_word_mask_multiplied, -1)  # [batch, n_claim_frames, seq_len]

        claim_frames_word_mask_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)
        claim_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]

        num_claim_frame_words_expanded_seq_len = num_claim_frame_words.expand(-1, -1,
                                                                              claim_frames_word_mask.size(
                                                                                  2))  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_all_zero_mask = (
                    num_claim_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = claim_frames_word_mask[
            claim_frames_word_mask_all_zero_mask]
        claim_frames_word_mask_multiplied_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = \
        claim_frames_word_mask_multiplied[claim_frames_word_mask_all_zero_mask]

        claim_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            claim_frames_word_mask_multiplied_fixed_all_zeros,
            claim_frames_word_mask_fixed_all_zeros)  # [batch, n_claim_frames, seq_len]

        claim_frame_features_sum = torch.bmm(claim_frames_word_mask_multiplied_softmaxed,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                                      claim_frame_features_sum.size(
                                                                          2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        # claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
        #    mean_pool_mask]  # [batch_size, n_claim_frames, hid]
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding

        # features of cosine similarity
        last_hidden_state_normalized = nn.functional.normalize(last_hidden_state, p=2, dim=2)
        similarity_matrix = torch.bmm(last_hidden_state_normalized,  torch.transpose(last_hidden_state_normalized, 1, 2))  # [batch, seq_len, seq_len]

        #similarity_matrix = self.cos_sim(last_hidden_state, last_hidden_state)  # [batch, seq_len, seq_len]
        # doc_words_mask [batch, seq_len]
        doc_words_mask_expanded = doc_words_mask.unsqueeze(1).expand(-1, similarity_matrix.size(1), -1)  # [batch, seq_len,seq_len]
        # inverted and mask to negative number
        doc_words_mask_expanded_inverted = doc_words_mask_expanded < 0.5 * -10000.0

        # similarity_matrix_masked = similarity_matrix * doc_words_mask_expanded  # [batch, seq_len, seq_len]
        similarity_matrix_masked = similarity_matrix + doc_words_mask_expanded_inverted  # [batch, seq_len, seq_len]
        similarity_max_pooled, _ = similarity_matrix_masked.max(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        # claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        claim_words_similarities_sum = torch.bmm(claim_frames_word_mask, similarity_max_pooled).squeeze(2)  # [batch, n_claim_frames]
        # mean_pool_mask [batch, n_claim_frames, hid]
        mean_pool_mask_shrinked = mean_pool_mask[:, :, 0]  # [batch, n_claim_frames]
        claim_words_precision = torch.zeros_like(claim_words_similarities_sum)  # [batch, n_claim_frames]
        # num_claim_frame_words [batch, n_claim_frames, 1]
        claim_words_precision[mean_pool_mask_shrinked] = claim_words_similarities_sum[mean_pool_mask_shrinked] / num_claim_frame_words.squeeze(-1)[
            mean_pool_mask_shrinked]  # [batch, n_claim_frames]
        """
        print("claim_words_precision")
        print(claim_words_precision.size())
        print(claim_words_precision[0])
        """
        claim_words_precision = claim_words_precision.unsqueeze(2)  # [batch, n_claim_frames, 1]

        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features, claim_words_precision], dim=2)  #  [batch, n_claim_frames, 2*hid_size+1]

        frame_classification_features_mean_pooled = frame_classification_features.mean(dim=1)  #  [batch, 2*hid_size+1]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]

        all_classification_feature = torch.cat([cls_representation, frame_classification_features_mean_pooled], dim=1)  #  [batch, 3*hid_size+1]

        logits = self.classifier(self.dropout(all_classification_feature))  # [batch, num_classes]

        outputs = (logits, claim_attn_output_weights) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMeanWordAttnMultiLabelAdapterClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMeanWordAttnMultiLabelAdapterClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        config.output_hidden_states = True
        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, self.num_labels)
        #self.classifier_attn = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax_dim_1 = MaskedSoftmax(dim=1)
        self.masked_softmax_dim_2 = MaskedSoftmax(dim=2)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        #self.layer_attn = nn.Linear(config.hidden_size, 1)
        self.token_self_attn = nn.Linear(config.hidden_size, 1)

        #self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilCosSimMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]


        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        last_hidden_state_projected = self.frame_feature_project(
            last_hidden_state)  # [batch_size, sequence_length, hidden_size]

        token_attn_score = self.token_self_attn(last_hidden_state_projected).squeeze()  # [batch_size, sequence_length]

        token_attn_score_resized_for_doc_frames = token_attn_score.unsqueeze(1).expand(-1, doc_frames_word_mask.size(1),
                                                                                       -1)  # [batch, n_doc_frames, seq_len]
        doc_frames_word_mask_multiplied = doc_frames_word_mask * token_attn_score_resized_for_doc_frames  # [batch, n_doc_frames, seq_len]
        # doc_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(doc_frames_word_mask_multiplied, -1)  # [batch, n_doc_frames, seq_len]

        doc_frames_word_mask_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)
        doc_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        num_doc_frame_words_expanded_seq_len = num_doc_frame_words.expand(-1, -1,
                                                                          doc_frames_word_mask.size(
                                                                              2))  # [batch, n_claim_frames, seq_len]

        doc_frames_word_mask_all_zero_mask = (
                num_doc_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        doc_frames_word_mask_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = doc_frames_word_mask[
            doc_frames_word_mask_all_zero_mask]
        doc_frames_word_mask_multiplied_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = \
            doc_frames_word_mask_multiplied[doc_frames_word_mask_all_zero_mask]

        doc_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            doc_frames_word_mask_multiplied_fixed_all_zeros,
            doc_frames_word_mask_fixed_all_zeros)  # [batch, n_doc_frames, seq_len]


        doc_frame_features_sum = torch.bmm(doc_frames_word_mask_multiplied_softmaxed,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask]
        # doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
        #    mean_pool_mask]  # [batch_size, n_frames, hid]

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        token_attn_score_resized_for_claim_frames = token_attn_score.unsqueeze(1).expand(-1,
                                                                                         claim_frames_word_mask.size(1),
                                                                                         -1)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_multiplied = claim_frames_word_mask * token_attn_score_resized_for_claim_frames  # [batch, n_claim_frames, seq_len]
        # claim_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(claim_frames_word_mask_multiplied, -1)  # [batch, n_claim_frames, seq_len]

        claim_frames_word_mask_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)
        claim_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]

        num_claim_frame_words_expanded_seq_len = num_claim_frame_words.expand(-1, -1,
                                                                              claim_frames_word_mask.size(
                                                                                  2))  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_all_zero_mask = (
                    num_claim_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = claim_frames_word_mask[
            claim_frames_word_mask_all_zero_mask]
        claim_frames_word_mask_multiplied_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = \
        claim_frames_word_mask_multiplied[claim_frames_word_mask_all_zero_mask]

        claim_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            claim_frames_word_mask_multiplied_fixed_all_zeros,
            claim_frames_word_mask_fixed_all_zeros)  # [batch, n_claim_frames, seq_len]

        claim_frame_features_sum = torch.bmm(claim_frames_word_mask_multiplied_softmaxed,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                                      claim_frame_features_sum.size(
                                                                          2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        # claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
        #    mean_pool_mask]  # [batch_size, n_claim_frames, hid]
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding


        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features], dim=2)  #  [batch, n_claim_frames, 2*hid_size]

        frame_classification_features_mean_pooled = frame_classification_features.mean(dim=1)  #  [batch, 2*hid_size+1]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]

        all_classification_feature = torch.cat([cls_representation, frame_classification_features_mean_pooled], dim=1)  #  [batch, 3*hid_size]

        logits = self.classifier(self.dropout(all_classification_feature))  # [batch, num_classes]

        outputs = (logits, claim_attn_output_weights) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMeanWordLayerAttnMultiLabelAdapterClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMeanWordLayerAttnMultiLabelAdapterClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        config.output_hidden_states = True
        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.classifier = nn.Linear(config.hidden_size * 3, self.num_labels)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax_dim_1 = MaskedSoftmax(dim=1)
        self.masked_softmax_dim_2 = MaskedSoftmax(dim=2)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        #self.layer_attn = nn.Linear(config.hidden_size, 1)
        self.token_self_attn = nn.Linear(config.hidden_size, 1)

        #self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilCosSimMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        all_layers_hidden_state = outputs[2]  # tuple of [batch_size, sequence_length, hidden_size]
        all_layers_hidden_state_tensor = torch.stack(all_layers_hidden_state,
                                                     dim=2)  # [batch_size, sequence_length, num_layers, hidden_size]
        hidden_state_layer_pooled, _ = torch.max(all_layers_hidden_state_tensor, dim=2)

        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        last_hidden_state_projected = self.frame_feature_project(
            hidden_state_layer_pooled)  # [batch_size, sequence_length, hidden_size]

        token_attn_score = self.token_self_attn(last_hidden_state_projected).squeeze()  # [batch_size, sequence_length]

        token_attn_score_resized_for_doc_frames = token_attn_score.unsqueeze(1).expand(-1, doc_frames_word_mask.size(1),
                                                                                       -1)  # [batch, n_doc_frames, seq_len]
        doc_frames_word_mask_multiplied = doc_frames_word_mask * token_attn_score_resized_for_doc_frames  # [batch, n_doc_frames, seq_len]
        # doc_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(doc_frames_word_mask_multiplied, -1)  # [batch, n_doc_frames, seq_len]

        doc_frames_word_mask_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)
        doc_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        num_doc_frame_words_expanded_seq_len = num_doc_frame_words.expand(-1, -1,
                                                                          doc_frames_word_mask.size(
                                                                              2))  # [batch, n_claim_frames, seq_len]

        doc_frames_word_mask_all_zero_mask = (
                num_doc_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        doc_frames_word_mask_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = doc_frames_word_mask[
            doc_frames_word_mask_all_zero_mask]
        doc_frames_word_mask_multiplied_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = \
            doc_frames_word_mask_multiplied[doc_frames_word_mask_all_zero_mask]

        doc_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            doc_frames_word_mask_multiplied_fixed_all_zeros,
            doc_frames_word_mask_fixed_all_zeros)  # [batch, n_doc_frames, seq_len]


        doc_frame_features_sum = torch.bmm(doc_frames_word_mask_multiplied_softmaxed,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask]
        # doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
        #    mean_pool_mask]  # [batch_size, n_frames, hid]

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        token_attn_score_resized_for_claim_frames = token_attn_score.unsqueeze(1).expand(-1,
                                                                                         claim_frames_word_mask.size(1),
                                                                                         -1)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_multiplied = claim_frames_word_mask * token_attn_score_resized_for_claim_frames  # [batch, n_claim_frames, seq_len]
        # claim_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(claim_frames_word_mask_multiplied, -1)  # [batch, n_claim_frames, seq_len]

        claim_frames_word_mask_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)
        claim_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]

        num_claim_frame_words_expanded_seq_len = num_claim_frame_words.expand(-1, -1,
                                                                              claim_frames_word_mask.size(
                                                                                  2))  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_all_zero_mask = (
                    num_claim_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = claim_frames_word_mask[
            claim_frames_word_mask_all_zero_mask]
        claim_frames_word_mask_multiplied_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = \
        claim_frames_word_mask_multiplied[claim_frames_word_mask_all_zero_mask]

        claim_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            claim_frames_word_mask_multiplied_fixed_all_zeros,
            claim_frames_word_mask_fixed_all_zeros)  # [batch, n_claim_frames, seq_len]

        claim_frame_features_sum = torch.bmm(claim_frames_word_mask_multiplied_softmaxed,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                                      claim_frame_features_sum.size(
                                                                          2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        # claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
        #    mean_pool_mask]  # [batch_size, n_claim_frames, hid]
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding


        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features], dim=2)  #  [batch, n_claim_frames, 2*hid_size]

        # claim_frames_padding_mask  [batch, n_claim_frames]

        frame_classification_features_mean_pooled = torch.sum(frame_classification_features * claim_frames_padding_mask.unsqueeze(2), dim=1) / torch.sum(claim_frames_padding_mask, dim=1, keepdim=True)

        #frame_classification_features_mean_pooled = frame_classification_features.mean(dim=1)  #  [batch, 2*hid_size+1]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]

        #all_classification_feature = torch.cat([cls_representation, frame_classification_features_mean_pooled], dim=1)  #  [batch, 3*hid_size]
        all_classification_feature = frame_classification_features_mean_pooled

        logits = self.classifier(self.dropout(all_classification_feature))  # [batch, num_classes]

        outputs = (logits, claim_attn_output_weights) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMeanWordLayerAttnNoSelectMultiLabelAdapterClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMeanWordLayerAttnNoSelectMultiLabelAdapterClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        config.output_hidden_states = True
        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.classifier = nn.Linear(config.hidden_size * 3, self.num_labels)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax_dim_1 = MaskedSoftmax(dim=1)
        self.masked_softmax_dim_2 = MaskedSoftmax(dim=2)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        #self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        #self.layer_attn = nn.Linear(config.hidden_size, 1)
        self.token_self_attn = nn.Linear(config.hidden_size, 1)

        #self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilCosSimMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        all_layers_hidden_state = outputs[2]  # tuple of [batch_size, sequence_length, hidden_size]
        all_layers_hidden_state_tensor = torch.stack(all_layers_hidden_state,
                                                     dim=2)  # [batch_size, sequence_length, num_layers, hidden_size]
        hidden_state_layer_pooled, _ = torch.max(all_layers_hidden_state_tensor, dim=2)

        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        last_hidden_state_projected = self.frame_feature_project(
            hidden_state_layer_pooled)  # [batch_size, sequence_length, hidden_size]

        token_attn_score = self.token_self_attn(last_hidden_state_projected).squeeze()  # [batch_size, sequence_length]

        token_attn_score_resized_for_doc_frames = token_attn_score.unsqueeze(1).expand(-1, doc_frames_word_mask.size(1),
                                                                                       -1)  # [batch, n_doc_frames, seq_len]
        doc_frames_word_mask_multiplied = doc_frames_word_mask * token_attn_score_resized_for_doc_frames  # [batch, n_doc_frames, seq_len]
        # doc_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(doc_frames_word_mask_multiplied, -1)  # [batch, n_doc_frames, seq_len]

        doc_frames_word_mask_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)
        doc_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        num_doc_frame_words_expanded_seq_len = num_doc_frame_words.expand(-1, -1,
                                                                          doc_frames_word_mask.size(
                                                                              2))  # [batch, n_claim_frames, seq_len]

        doc_frames_word_mask_all_zero_mask = (
                num_doc_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        doc_frames_word_mask_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = doc_frames_word_mask[
            doc_frames_word_mask_all_zero_mask]
        doc_frames_word_mask_multiplied_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = \
            doc_frames_word_mask_multiplied[doc_frames_word_mask_all_zero_mask]

        doc_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            doc_frames_word_mask_multiplied_fixed_all_zeros,
            doc_frames_word_mask_fixed_all_zeros)  # [batch, n_doc_frames, seq_len]


        doc_frame_features_sum = torch.bmm(doc_frames_word_mask_multiplied_softmaxed,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask]
        # doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
        #    mean_pool_mask]  # [batch_size, n_frames, hid]

        doc_mean_pool_mask = mean_pool_mask[:, :, 0]  # [batch, n_doc_frames]
        doc_frame_features_mean_pooled = torch.sum(doc_frame_features * doc_mean_pool_mask.unsqueeze(2),
                                                     dim=1) / torch.sum(doc_mean_pool_mask, dim=1, keepdim=True)

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        token_attn_score_resized_for_claim_frames = token_attn_score.unsqueeze(1).expand(-1,
                                                                                         claim_frames_word_mask.size(1),
                                                                                         -1)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_multiplied = claim_frames_word_mask * token_attn_score_resized_for_claim_frames  # [batch, n_claim_frames, seq_len]
        # claim_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(claim_frames_word_mask_multiplied, -1)  # [batch, n_claim_frames, seq_len]

        claim_frames_word_mask_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)
        claim_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]

        num_claim_frame_words_expanded_seq_len = num_claim_frame_words.expand(-1, -1,
                                                                              claim_frames_word_mask.size(
                                                                                  2))  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_all_zero_mask = (
                    num_claim_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = claim_frames_word_mask[
            claim_frames_word_mask_all_zero_mask]
        claim_frames_word_mask_multiplied_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = \
        claim_frames_word_mask_multiplied[claim_frames_word_mask_all_zero_mask]

        claim_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            claim_frames_word_mask_multiplied_fixed_all_zeros,
            claim_frames_word_mask_fixed_all_zeros)  # [batch, n_claim_frames, seq_len]

        claim_frame_features_sum = torch.bmm(claim_frames_word_mask_multiplied_softmaxed,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                                      claim_frame_features_sum.size(
                                                                          2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        # claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
        #    mean_pool_mask]  # [batch_size, n_claim_frames, hid]
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding


        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        #frame_classification_features = torch.cat([claim_attn_output, claim_frame_features], dim=2)  #  [batch, n_claim_frames, 2*hid_size]

        # claim_frames_padding_mask  [batch, n_claim_frames]

        claim_frame_features_mean_pooled = torch.sum(claim_frame_features * claim_frames_padding_mask.unsqueeze(2), dim=1) / torch.sum(claim_frames_padding_mask, dim=1, keepdim=True)

        #frame_classification_features_mean_pooled = frame_classification_features.mean(dim=1)  #  [batch, 2*hid_size+1]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]

        #all_classification_feature = torch.cat([cls_representation, frame_classification_features_mean_pooled], dim=1)  #  [batch, 3*hid_size]
        #all_classification_feature = frame_classification_features_mean_pooled
        all_classification_feature = torch.cat([doc_frame_features_mean_pooled, claim_frame_features_mean_pooled], dim=1)

        logits = self.classifier(self.dropout(all_classification_feature))  # [batch, num_classes]

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMeanWordLayerAttnMultiLabelClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMeanWordLayerAttnMultiLabelClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        config.output_hidden_states = True
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.classifier = nn.Linear(config.hidden_size * 3, self.num_labels)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        #self.classifier_attn = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax_dim_1 = MaskedSoftmax(dim=1)
        self.masked_softmax_dim_2 = MaskedSoftmax(dim=2)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        #self.layer_attn = nn.Linear(config.hidden_size, 1)
        self.token_self_attn = nn.Linear(config.hidden_size, 1)

        #self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilCosSimMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        all_layers_hidden_state = outputs[2]  # tuple of [batch_size, sequence_length, hidden_size]
        all_layers_hidden_state_tensor = torch.stack(all_layers_hidden_state,
                                                     dim=2)  # [batch_size, sequence_length, num_layers, hidden_size]
        hidden_state_layer_pooled, _ = torch.max(all_layers_hidden_state_tensor, dim=2)

        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        #last_hidden_state_projected = hidden_state_layer_pooled
        last_hidden_state_projected = self.frame_feature_project(
            hidden_state_layer_pooled)  # [batch_size, sequence_length, hidden_size]

        token_attn_score = self.token_self_attn(last_hidden_state_projected).squeeze()  # [batch_size, sequence_length]

        token_attn_score_resized_for_doc_frames = token_attn_score.unsqueeze(1).expand(-1, doc_frames_word_mask.size(1),
                                                                                       -1)  # [batch, n_doc_frames, seq_len]
        doc_frames_word_mask_multiplied = doc_frames_word_mask * token_attn_score_resized_for_doc_frames  # [batch, n_doc_frames, seq_len]
        # doc_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(doc_frames_word_mask_multiplied, -1)  # [batch, n_doc_frames, seq_len]

        doc_frames_word_mask_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)
        doc_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        num_doc_frame_words_expanded_seq_len = num_doc_frame_words.expand(-1, -1,
                                                                          doc_frames_word_mask.size(
                                                                              2))  # [batch, n_claim_frames, seq_len]

        doc_frames_word_mask_all_zero_mask = (
                num_doc_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        doc_frames_word_mask_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = doc_frames_word_mask[
            doc_frames_word_mask_all_zero_mask]
        doc_frames_word_mask_multiplied_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = \
            doc_frames_word_mask_multiplied[doc_frames_word_mask_all_zero_mask]

        doc_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            doc_frames_word_mask_multiplied_fixed_all_zeros,
            doc_frames_word_mask_fixed_all_zeros)  # [batch, n_doc_frames, seq_len]


        doc_frame_features_sum = torch.bmm(doc_frames_word_mask_multiplied_softmaxed,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask]
        # doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
        #    mean_pool_mask]  # [batch_size, n_frames, hid]

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        token_attn_score_resized_for_claim_frames = token_attn_score.unsqueeze(1).expand(-1,
                                                                                         claim_frames_word_mask.size(1),
                                                                                         -1)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_multiplied = claim_frames_word_mask * token_attn_score_resized_for_claim_frames  # [batch, n_claim_frames, seq_len]
        # claim_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(claim_frames_word_mask_multiplied, -1)  # [batch, n_claim_frames, seq_len]

        claim_frames_word_mask_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)
        claim_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]

        num_claim_frame_words_expanded_seq_len = num_claim_frame_words.expand(-1, -1,
                                                                              claim_frames_word_mask.size(
                                                                                  2))  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_all_zero_mask = (
                    num_claim_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = claim_frames_word_mask[
            claim_frames_word_mask_all_zero_mask]
        claim_frames_word_mask_multiplied_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = \
        claim_frames_word_mask_multiplied[claim_frames_word_mask_all_zero_mask]

        claim_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            claim_frames_word_mask_multiplied_fixed_all_zeros,
            claim_frames_word_mask_fixed_all_zeros)  # [batch, n_claim_frames, seq_len]

        claim_frame_features_sum = torch.bmm(claim_frames_word_mask_multiplied_softmaxed,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                                      claim_frame_features_sum.size(
                                                                          2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        # claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
        #    mean_pool_mask]  # [batch_size, n_claim_frames, hid]
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding


        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features], dim=2)  #  [batch, n_claim_frames, 2*hid_size]

        frame_classification_features_mean_pooled = torch.sum(
            frame_classification_features * claim_frames_padding_mask.unsqueeze(2), dim=1) / torch.sum(
            claim_frames_padding_mask, dim=1, keepdim=True)

        # frame_classification_features_mean_pooled = frame_classification_features.mean(dim=1)  #  [batch, 2*hid_size+1]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]

        #all_classification_feature = torch.cat([cls_representation, frame_classification_features_mean_pooled], dim=1)  #  [batch, 3*hid_size]
        all_classification_feature = frame_classification_features_mean_pooled

        logits = self.classifier(self.dropout(all_classification_feature))  # [batch, num_classes]

        outputs = (logits, claim_attn_output_weights) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMeanWordLayerAttnNoSelectMultiLabelClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMeanWordLayerAttnNoSelectMultiLabelClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        config.output_hidden_states = True
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.classifier = nn.Linear(config.hidden_size * 3, self.num_labels)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax_dim_1 = MaskedSoftmax(dim=1)
        self.masked_softmax_dim_2 = MaskedSoftmax(dim=2)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        #self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        #self.layer_attn = nn.Linear(config.hidden_size, 1)
        self.token_self_attn = nn.Linear(config.hidden_size, 1)

        #self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilCosSimMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        all_layers_hidden_state = outputs[2]  # tuple of [batch_size, sequence_length, hidden_size]
        all_layers_hidden_state_tensor = torch.stack(all_layers_hidden_state,
                                                     dim=2)  # [batch_size, sequence_length, num_layers, hidden_size]
        hidden_state_layer_pooled, _ = torch.max(all_layers_hidden_state_tensor, dim=2)

        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        last_hidden_state_projected = self.frame_feature_project(
            hidden_state_layer_pooled)  # [batch_size, sequence_length, hidden_size]

        token_attn_score = self.token_self_attn(last_hidden_state_projected).squeeze()  # [batch_size, sequence_length]

        token_attn_score_resized_for_doc_frames = token_attn_score.unsqueeze(1).expand(-1, doc_frames_word_mask.size(1),
                                                                                       -1)  # [batch, n_doc_frames, seq_len]
        doc_frames_word_mask_multiplied = doc_frames_word_mask * token_attn_score_resized_for_doc_frames  # [batch, n_doc_frames, seq_len]
        # doc_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(doc_frames_word_mask_multiplied, -1)  # [batch, n_doc_frames, seq_len]

        doc_frames_word_mask_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)
        doc_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        num_doc_frame_words_expanded_seq_len = num_doc_frame_words.expand(-1, -1,
                                                                          doc_frames_word_mask.size(
                                                                              2))  # [batch, n_claim_frames, seq_len]

        doc_frames_word_mask_all_zero_mask = (
                num_doc_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        doc_frames_word_mask_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = doc_frames_word_mask[
            doc_frames_word_mask_all_zero_mask]
        doc_frames_word_mask_multiplied_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = \
            doc_frames_word_mask_multiplied[doc_frames_word_mask_all_zero_mask]

        doc_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            doc_frames_word_mask_multiplied_fixed_all_zeros,
            doc_frames_word_mask_fixed_all_zeros)  # [batch, n_doc_frames, seq_len]


        doc_frame_features_sum = torch.bmm(doc_frames_word_mask_multiplied_softmaxed,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask]
        # doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
        #    mean_pool_mask]  # [batch_size, n_frames, hid]

        doc_mean_pool_mask = mean_pool_mask[:, :, 0]  # [batch, n_doc_frames]
        doc_frame_features_mean_pooled = torch.sum(doc_frame_features * doc_mean_pool_mask.unsqueeze(2),
                                                     dim=1) / torch.sum(doc_mean_pool_mask, dim=1, keepdim=True)

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        token_attn_score_resized_for_claim_frames = token_attn_score.unsqueeze(1).expand(-1,
                                                                                         claim_frames_word_mask.size(1),
                                                                                         -1)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_multiplied = claim_frames_word_mask * token_attn_score_resized_for_claim_frames  # [batch, n_claim_frames, seq_len]
        # claim_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(claim_frames_word_mask_multiplied, -1)  # [batch, n_claim_frames, seq_len]

        claim_frames_word_mask_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)
        claim_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]

        num_claim_frame_words_expanded_seq_len = num_claim_frame_words.expand(-1, -1,
                                                                              claim_frames_word_mask.size(
                                                                                  2))  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_all_zero_mask = (
                    num_claim_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = claim_frames_word_mask[
            claim_frames_word_mask_all_zero_mask]
        claim_frames_word_mask_multiplied_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = \
        claim_frames_word_mask_multiplied[claim_frames_word_mask_all_zero_mask]

        claim_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            claim_frames_word_mask_multiplied_fixed_all_zeros,
            claim_frames_word_mask_fixed_all_zeros)  # [batch, n_claim_frames, seq_len]

        claim_frame_features_sum = torch.bmm(claim_frames_word_mask_multiplied_softmaxed,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                                      claim_frame_features_sum.size(
                                                                          2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        # claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
        #    mean_pool_mask]  # [batch_size, n_claim_frames, hid]
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding


        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        #frame_classification_features = torch.cat([claim_attn_output, claim_frame_features], dim=2)  #  [batch, n_claim_frames, 2*hid_size]

        # claim_frames_padding_mask  [batch, n_claim_frames]

        claim_frame_features_mean_pooled = torch.sum(claim_frame_features * claim_frames_padding_mask.unsqueeze(2), dim=1) / torch.sum(claim_frames_padding_mask, dim=1, keepdim=True)

        #frame_classification_features_mean_pooled = frame_classification_features.mean(dim=1)  #  [batch, 2*hid_size+1]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]

        #all_classification_feature = torch.cat([cls_representation, frame_classification_features_mean_pooled], dim=1)  #  [batch, 3*hid_size]
        #all_classification_feature = frame_classification_features_mean_pooled
        all_classification_feature = torch.cat([doc_frame_features_mean_pooled, claim_frame_features_mean_pooled], dim=1)

        logits = self.classifier(self.dropout(all_classification_feature))  # [batch, num_classes]

        outputs = (logits, ) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMeanWordAttnMultiLabelClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMeanWordAttnMultiLabelClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        config.output_hidden_states = True
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, self.num_labels)
        #self.classifier_attn = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax_dim_1 = MaskedSoftmax(dim=1)
        self.masked_softmax_dim_2 = MaskedSoftmax(dim=2)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        #self.layer_attn = nn.Linear(config.hidden_size, 1)
        self.token_self_attn = nn.Linear(config.hidden_size, 1)

        #self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilCosSimMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]


        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        last_hidden_state_projected = self.frame_feature_project(
            last_hidden_state)  # [batch_size, sequence_length, hidden_size]

        token_attn_score = self.token_self_attn(last_hidden_state_projected).squeeze()  # [batch_size, sequence_length]

        token_attn_score_resized_for_doc_frames = token_attn_score.unsqueeze(1).expand(-1, doc_frames_word_mask.size(1),
                                                                                       -1)  # [batch, n_doc_frames, seq_len]
        doc_frames_word_mask_multiplied = doc_frames_word_mask * token_attn_score_resized_for_doc_frames  # [batch, n_doc_frames, seq_len]
        # doc_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(doc_frames_word_mask_multiplied, -1)  # [batch, n_doc_frames, seq_len]

        doc_frames_word_mask_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)
        doc_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        num_doc_frame_words_expanded_seq_len = num_doc_frame_words.expand(-1, -1,
                                                                          doc_frames_word_mask.size(
                                                                              2))  # [batch, n_claim_frames, seq_len]

        doc_frames_word_mask_all_zero_mask = (
                num_doc_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        doc_frames_word_mask_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = doc_frames_word_mask[
            doc_frames_word_mask_all_zero_mask]
        doc_frames_word_mask_multiplied_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = \
            doc_frames_word_mask_multiplied[doc_frames_word_mask_all_zero_mask]

        doc_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            doc_frames_word_mask_multiplied_fixed_all_zeros,
            doc_frames_word_mask_fixed_all_zeros)  # [batch, n_doc_frames, seq_len]


        doc_frame_features_sum = torch.bmm(doc_frames_word_mask_multiplied_softmaxed,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask]
        # doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
        #    mean_pool_mask]  # [batch_size, n_frames, hid]

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        token_attn_score_resized_for_claim_frames = token_attn_score.unsqueeze(1).expand(-1,
                                                                                         claim_frames_word_mask.size(1),
                                                                                         -1)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_multiplied = claim_frames_word_mask * token_attn_score_resized_for_claim_frames  # [batch, n_claim_frames, seq_len]
        # claim_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(claim_frames_word_mask_multiplied, -1)  # [batch, n_claim_frames, seq_len]

        claim_frames_word_mask_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)
        claim_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]

        num_claim_frame_words_expanded_seq_len = num_claim_frame_words.expand(-1, -1,
                                                                              claim_frames_word_mask.size(
                                                                                  2))  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_all_zero_mask = (
                    num_claim_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = claim_frames_word_mask[
            claim_frames_word_mask_all_zero_mask]
        claim_frames_word_mask_multiplied_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = \
        claim_frames_word_mask_multiplied[claim_frames_word_mask_all_zero_mask]

        claim_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            claim_frames_word_mask_multiplied_fixed_all_zeros,
            claim_frames_word_mask_fixed_all_zeros)  # [batch, n_claim_frames, seq_len]

        claim_frame_features_sum = torch.bmm(claim_frames_word_mask_multiplied_softmaxed,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                                      claim_frame_features_sum.size(
                                                                          2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        # claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
        #    mean_pool_mask]  # [batch_size, n_claim_frames, hid]
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding


        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features], dim=2)  #  [batch, n_claim_frames, 2*hid_size]

        frame_classification_features_mean_pooled = frame_classification_features.mean(dim=1)  #  [batch, 2*hid_size+1]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]

        all_classification_feature = torch.cat([cls_representation, frame_classification_features_mean_pooled], dim=1)  #  [batch, 3*hid_size]

        logits = self.classifier(self.dropout(all_classification_feature))  # [batch, num_classes]

        outputs = (logits, claim_attn_output_weights) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnWeightCosSimWordAttnMultiLabelAdapterClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnWeightCosSimWordAttnMultiLabelAdapterClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        config.output_hidden_states = True
        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        #self.classifier_attn = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax_dim_1 = MaskedSoftmax(dim=1)
        self.masked_softmax_dim_2 = MaskedSoftmax(dim=2)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        #self.layer_attn = nn.Linear(config.hidden_size, 1)
        self.token_self_attn = nn.Linear(config.hidden_size, 1)
        self.frame_classification_features_attn = nn.Linear(config.hidden_size * 2 + 1, 1)

        #self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilCosSimMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]


        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        last_hidden_state_projected = self.frame_feature_project(
            last_hidden_state)  # [batch_size, sequence_length, hidden_size]

        token_attn_score = self.token_self_attn(last_hidden_state_projected).squeeze()  # [batch_size, sequence_length]

        token_attn_score_resized_for_doc_frames = token_attn_score.unsqueeze(1).expand(-1, doc_frames_word_mask.size(1),
                                                                                       -1)  # [batch, n_doc_frames, seq_len]
        doc_frames_word_mask_multiplied = doc_frames_word_mask * token_attn_score_resized_for_doc_frames  # [batch, n_doc_frames, seq_len]
        # doc_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(doc_frames_word_mask_multiplied, -1)  # [batch, n_doc_frames, seq_len]

        doc_frames_word_mask_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)
        doc_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        num_doc_frame_words_expanded_seq_len = num_doc_frame_words.expand(-1, -1,
                                                                          doc_frames_word_mask.size(
                                                                              2))  # [batch, n_claim_frames, seq_len]

        doc_frames_word_mask_all_zero_mask = (
                num_doc_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        doc_frames_word_mask_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = doc_frames_word_mask[
            doc_frames_word_mask_all_zero_mask]
        doc_frames_word_mask_multiplied_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = \
            doc_frames_word_mask_multiplied[doc_frames_word_mask_all_zero_mask]

        doc_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            doc_frames_word_mask_multiplied_fixed_all_zeros,
            doc_frames_word_mask_fixed_all_zeros)  # [batch, n_doc_frames, seq_len]


        doc_frame_features_sum = torch.bmm(doc_frames_word_mask_multiplied_softmaxed,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask]
        # doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
        #    mean_pool_mask]  # [batch_size, n_frames, hid]

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        token_attn_score_resized_for_claim_frames = token_attn_score.unsqueeze(1).expand(-1,
                                                                                         claim_frames_word_mask.size(1),
                                                                                         -1)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_multiplied = claim_frames_word_mask * token_attn_score_resized_for_claim_frames  # [batch, n_claim_frames, seq_len]
        # claim_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(claim_frames_word_mask_multiplied, -1)  # [batch, n_claim_frames, seq_len]

        claim_frames_word_mask_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)
        claim_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]

        num_claim_frame_words_expanded_seq_len = num_claim_frame_words.expand(-1, -1,
                                                                              claim_frames_word_mask.size(
                                                                                  2))  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_all_zero_mask = (
                    num_claim_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = claim_frames_word_mask[
            claim_frames_word_mask_all_zero_mask]
        claim_frames_word_mask_multiplied_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = \
        claim_frames_word_mask_multiplied[claim_frames_word_mask_all_zero_mask]

        claim_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            claim_frames_word_mask_multiplied_fixed_all_zeros,
            claim_frames_word_mask_fixed_all_zeros)  # [batch, n_claim_frames, seq_len]

        claim_frame_features_sum = torch.bmm(claim_frames_word_mask_multiplied_softmaxed,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                                      claim_frame_features_sum.size(
                                                                          2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        # claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
        #    mean_pool_mask]  # [batch_size, n_claim_frames, hid]
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding

        # features of cosine similarity
        last_hidden_state_normalized = nn.functional.normalize(last_hidden_state, p=2, dim=2)
        similarity_matrix = torch.bmm(last_hidden_state_normalized,  torch.transpose(last_hidden_state_normalized, 1, 2))  # [batch, seq_len, seq_len]

        #similarity_matrix = self.cos_sim(last_hidden_state, last_hidden_state)  # [batch, seq_len, seq_len]
        # doc_words_mask [batch, seq_len]
        doc_words_mask_expanded = doc_words_mask.unsqueeze(1).expand(-1, similarity_matrix.size(1), -1)  # [batch, seq_len,seq_len]
        # inverted and mask to negative number
        doc_words_mask_expanded_inverted = doc_words_mask_expanded < 0.5 * -10000.0

        # similarity_matrix_masked = similarity_matrix * doc_words_mask_expanded  # [batch, seq_len, seq_len]
        similarity_matrix_masked = similarity_matrix + doc_words_mask_expanded_inverted  # [batch, seq_len, seq_len]
        similarity_max_pooled, _ = similarity_matrix_masked.max(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        # claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        claim_words_similarities_sum = torch.bmm(claim_frames_word_mask, similarity_max_pooled).squeeze(2)  # [batch, n_claim_frames]
        # mean_pool_mask [batch, n_claim_frames, hid]
        mean_pool_mask_shrinked = mean_pool_mask[:, :, 0]  # [batch, n_claim_frames]
        claim_words_precision = torch.zeros_like(claim_words_similarities_sum)  # [batch, n_claim_frames]
        # num_claim_frame_words [batch, n_claim_frames, 1]
        claim_words_precision[mean_pool_mask_shrinked] = claim_words_similarities_sum[mean_pool_mask_shrinked] / num_claim_frame_words.squeeze(-1)[
            mean_pool_mask_shrinked]  # [batch, n_claim_frames]
        """
        print("claim_words_precision")
        print(claim_words_precision.size())
        print(claim_words_precision[0])
        """
        claim_words_precision = claim_words_precision.unsqueeze(2)  # [batch, n_claim_frames, 1]

        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features, claim_words_precision], dim=2)  #  [batch, n_claim_frames, 2*hid_size+1]

        claim_frame_weights = self.frame_classification_features_attn(
            frame_classification_features)  # [batch, n_claim_frames, 1]

        claim_frame_weights_normalized = F.softmax(claim_frame_weights, dim=1)  # [batch_size, n_claim_frames, 1]

        claim_frame_weights_normalized_expanded = claim_frame_weights_normalized.expand(-1, -1,
                                                                                        frame_classification_features.size(
                                                                                            2))  # [batch, n_claim_frames, 2*hid_size+1]

        frame_classification_features_pooled = torch.sum(
            frame_classification_features * claim_frame_weights_normalized_expanded, dim=1)  # [batch, 2*hid_size+1]

        #frame_classification_features_mean_pooled = frame_classification_features.mean(dim=1)  #  [batch, 2*hid_size+1]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]

        all_classification_feature = torch.cat([cls_representation, frame_classification_features_pooled], dim=1)  #  [batch, 3*hid_size+1]

        logits = self.classifier(self.dropout(all_classification_feature))  # [batch, num_classes]

        outputs = (logits, claim_attn_output_weights) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMeanCosSimWordLayerAttnMultiLabelAdapterClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMeanCosSimWordLayerAttnMultiLabelAdapterClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        config.output_hidden_states = True
        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        #self.classifier_attn = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax_dim_1 = MaskedSoftmax(dim=1)
        self.masked_softmax_dim_2 = MaskedSoftmax(dim=2)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_attn = nn.Linear(config.hidden_size, 1)
        self.token_self_attn = nn.Linear(config.hidden_size, 1)

        #self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMeanCosSimWordLayerAttnMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        all_layers_hidden_state = outputs[2]  # tuple of [batch_size, sequence_length, hidden_size]
        all_layers_hidden_state_tensor = torch.stack(all_layers_hidden_state,
                                                     dim=2)  # [batch_size, sequence_length, num_layers, hidden_size]
        all_layers_hidden_state_tensor = self.frame_feature_project(all_layers_hidden_state_tensor)

        # Layer attention
        layer_attn_weights_normalized = F.softmax(self.layer_attn(all_layers_hidden_state_tensor),
                                                  dim=2)  # [batch_size, sequence_length, num_layers, 1]
        layer_attn_weights_normalized_expanded = layer_attn_weights_normalized.expand(-1, -1, -1,
                                                                                      all_layers_hidden_state_tensor.size(
                                                                                          3))  # [batch_size, sequence_length, num_layers, hid_size]
        hidden_state_fused = torch.sum(all_layers_hidden_state_tensor * layer_attn_weights_normalized_expanded,
                                       dim=2)  # [batch_size, sequence_length, hid_size]
        # print("hidden_state_fused")
        # print(hidden_state_fused.size())
        # exit()
        last_hidden_state_projected = hidden_state_fused

        token_attn_score = self.token_self_attn(last_hidden_state_projected).squeeze()  # [batch_size, sequence_length]

        token_attn_score_resized_for_doc_frames = token_attn_score.unsqueeze(1).expand(-1, doc_frames_word_mask.size(1),
                                                                                       -1)  # [batch, n_doc_frames, seq_len]
        doc_frames_word_mask_multiplied = doc_frames_word_mask * token_attn_score_resized_for_doc_frames  # [batch, n_doc_frames, seq_len]
        # doc_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(doc_frames_word_mask_multiplied, -1)  # [batch, n_doc_frames, seq_len]

        doc_frames_word_mask_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)
        doc_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        num_doc_frame_words_expanded_seq_len = num_doc_frame_words.expand(-1, -1,
                                                                          doc_frames_word_mask.size(
                                                                              2))  # [batch, n_claim_frames, seq_len]

        doc_frames_word_mask_all_zero_mask = (
                num_doc_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        doc_frames_word_mask_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = doc_frames_word_mask[
            doc_frames_word_mask_all_zero_mask]
        doc_frames_word_mask_multiplied_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = \
            doc_frames_word_mask_multiplied[doc_frames_word_mask_all_zero_mask]

        doc_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            doc_frames_word_mask_multiplied_fixed_all_zeros,
            doc_frames_word_mask_fixed_all_zeros)  # [batch, n_doc_frames, seq_len]


        doc_frame_features_sum = torch.bmm(doc_frames_word_mask_multiplied_softmaxed,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask]
        # doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
        #    mean_pool_mask]  # [batch_size, n_frames, hid]

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        token_attn_score_resized_for_claim_frames = token_attn_score.unsqueeze(1).expand(-1,
                                                                                         claim_frames_word_mask.size(1),
                                                                                         -1)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_multiplied = claim_frames_word_mask * token_attn_score_resized_for_claim_frames  # [batch, n_claim_frames, seq_len]
        # claim_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(claim_frames_word_mask_multiplied, -1)  # [batch, n_claim_frames, seq_len]

        claim_frames_word_mask_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)
        claim_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]

        num_claim_frame_words_expanded_seq_len = num_claim_frame_words.expand(-1, -1,
                                                                              claim_frames_word_mask.size(
                                                                                  2))  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_all_zero_mask = (
                    num_claim_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = claim_frames_word_mask[
            claim_frames_word_mask_all_zero_mask]
        claim_frames_word_mask_multiplied_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = \
        claim_frames_word_mask_multiplied[claim_frames_word_mask_all_zero_mask]

        claim_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            claim_frames_word_mask_multiplied_fixed_all_zeros,
            claim_frames_word_mask_fixed_all_zeros)  # [batch, n_claim_frames, seq_len]

        claim_frame_features_sum = torch.bmm(claim_frames_word_mask_multiplied_softmaxed,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                                      claim_frame_features_sum.size(
                                                                          2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        # claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
        #    mean_pool_mask]  # [batch_size, n_claim_frames, hid]
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding

        # features of cosine similarity
        last_hidden_state_normalized = nn.functional.normalize(last_hidden_state, p=2, dim=2)
        similarity_matrix = torch.bmm(last_hidden_state_normalized,  torch.transpose(last_hidden_state_normalized, 1, 2))  # [batch, seq_len, seq_len]

        #similarity_matrix = self.cos_sim(last_hidden_state, last_hidden_state)  # [batch, seq_len, seq_len]
        # doc_words_mask [batch, seq_len]
        doc_words_mask_expanded = doc_words_mask.unsqueeze(1).expand(-1, similarity_matrix.size(1), -1)  # [batch, seq_len,seq_len]
        # inverted and mask to negative number
        doc_words_mask_expanded_inverted = doc_words_mask_expanded < 0.5 * -10000.0

        # similarity_matrix_masked = similarity_matrix * doc_words_mask_expanded  # [batch, seq_len, seq_len]
        similarity_matrix_masked = similarity_matrix + doc_words_mask_expanded_inverted  # [batch, seq_len, seq_len]
        similarity_max_pooled, _ = similarity_matrix_masked.max(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        # claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        claim_words_similarities_sum = torch.bmm(claim_frames_word_mask, similarity_max_pooled).squeeze(2)  # [batch, n_claim_frames]
        # mean_pool_mask [batch, n_claim_frames, hid]
        mean_pool_mask_shrinked = mean_pool_mask[:, :, 0]  # [batch, n_claim_frames]
        claim_words_precision = torch.zeros_like(claim_words_similarities_sum)  # [batch, n_claim_frames]
        # num_claim_frame_words [batch, n_claim_frames, 1]
        claim_words_precision[mean_pool_mask_shrinked] = claim_words_similarities_sum[mean_pool_mask_shrinked] / num_claim_frame_words.squeeze(-1)[
            mean_pool_mask_shrinked]  # [batch, n_claim_frames]
        """
        print("claim_words_precision")
        print(claim_words_precision.size())
        print(claim_words_precision[0])
        """
        claim_words_precision = claim_words_precision.unsqueeze(2)  # [batch, n_claim_frames, 1]

        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features, claim_words_precision], dim=2)  #  [batch, n_claim_frames, 2*hid_size+1]

        frame_classification_features_mean_pooled = frame_classification_features.mean(dim=1)  #  [batch, 2*hid_size+1]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]

        all_classification_feature = torch.cat([cls_representation, frame_classification_features_mean_pooled], dim=1)  #  [batch, 3*hid_size+1]

        logits = self.classifier(self.dropout(all_classification_feature))  # [batch, num_classes]

        outputs = (logits, claim_attn_output_weights) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnWeightCosSimWordLayerAttnMultiLabelAdapterClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnWeightCosSimWordLayerAttnMultiLabelAdapterClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        config.output_hidden_states = True
        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        #self.classifier_attn = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax_dim_1 = MaskedSoftmax(dim=1)
        self.masked_softmax_dim_2 = MaskedSoftmax(dim=2)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_attn = nn.Linear(config.hidden_size, 1)
        self.token_self_attn = nn.Linear(config.hidden_size, 1)
        self.frame_classification_features_attn = nn.Linear(config.hidden_size * 2 + 1, 1)

        #self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnWeightCosSimWordLayerAttnMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        all_layers_hidden_state = outputs[2]  # tuple of [batch_size, sequence_length, hidden_size]
        all_layers_hidden_state_tensor = torch.stack(all_layers_hidden_state,
                                                     dim=2)  # [batch_size, sequence_length, num_layers, hidden_size]
        all_layers_hidden_state_tensor = self.frame_feature_project(all_layers_hidden_state_tensor)

        # Layer attention
        layer_attn_weights_normalized = F.softmax(self.layer_attn(all_layers_hidden_state_tensor),
                                                  dim=2)  # [batch_size, sequence_length, num_layers, 1]
        layer_attn_weights_normalized_expanded = layer_attn_weights_normalized.expand(-1, -1, -1,
                                                                                      all_layers_hidden_state_tensor.size(
                                                                                          3))  # [batch_size, sequence_length, num_layers, hid_size]
        hidden_state_fused = torch.sum(all_layers_hidden_state_tensor * layer_attn_weights_normalized_expanded,
                                       dim=2)  # [batch_size, sequence_length, hid_size]
        # print("hidden_state_fused")
        # print(hidden_state_fused.size())
        # exit()
        last_hidden_state_projected = hidden_state_fused

        token_attn_score = self.token_self_attn(last_hidden_state_projected).squeeze()  # [batch_size, sequence_length]

        token_attn_score_resized_for_doc_frames = token_attn_score.unsqueeze(1).expand(-1, doc_frames_word_mask.size(1),
                                                                                       -1)  # [batch, n_doc_frames, seq_len]
        doc_frames_word_mask_multiplied = doc_frames_word_mask * token_attn_score_resized_for_doc_frames  # [batch, n_doc_frames, seq_len]
        # doc_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(doc_frames_word_mask_multiplied, -1)  # [batch, n_doc_frames, seq_len]

        doc_frames_word_mask_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)
        doc_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        num_doc_frame_words_expanded_seq_len = num_doc_frame_words.expand(-1, -1,
                                                                          doc_frames_word_mask.size(
                                                                              2))  # [batch, n_claim_frames, seq_len]

        doc_frames_word_mask_all_zero_mask = (
                num_doc_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        doc_frames_word_mask_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = doc_frames_word_mask[
            doc_frames_word_mask_all_zero_mask]
        doc_frames_word_mask_multiplied_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = \
            doc_frames_word_mask_multiplied[doc_frames_word_mask_all_zero_mask]

        doc_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            doc_frames_word_mask_multiplied_fixed_all_zeros,
            doc_frames_word_mask_fixed_all_zeros)  # [batch, n_doc_frames, seq_len]


        doc_frame_features_sum = torch.bmm(doc_frames_word_mask_multiplied_softmaxed,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask]
        # doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
        #    mean_pool_mask]  # [batch_size, n_frames, hid]

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        token_attn_score_resized_for_claim_frames = token_attn_score.unsqueeze(1).expand(-1,
                                                                                         claim_frames_word_mask.size(1),
                                                                                         -1)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_multiplied = claim_frames_word_mask * token_attn_score_resized_for_claim_frames  # [batch, n_claim_frames, seq_len]
        # claim_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(claim_frames_word_mask_multiplied, -1)  # [batch, n_claim_frames, seq_len]

        claim_frames_word_mask_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)
        claim_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]

        num_claim_frame_words_expanded_seq_len = num_claim_frame_words.expand(-1, -1,
                                                                              claim_frames_word_mask.size(
                                                                                  2))  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_all_zero_mask = (
                    num_claim_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = claim_frames_word_mask[
            claim_frames_word_mask_all_zero_mask]
        claim_frames_word_mask_multiplied_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = \
        claim_frames_word_mask_multiplied[claim_frames_word_mask_all_zero_mask]

        claim_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            claim_frames_word_mask_multiplied_fixed_all_zeros,
            claim_frames_word_mask_fixed_all_zeros)  # [batch, n_claim_frames, seq_len]

        claim_frame_features_sum = torch.bmm(claim_frames_word_mask_multiplied_softmaxed,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                                      claim_frame_features_sum.size(
                                                                          2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        # claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
        #    mean_pool_mask]  # [batch_size, n_claim_frames, hid]
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding

        # features of cosine similarity
        last_hidden_state_normalized = nn.functional.normalize(last_hidden_state, p=2, dim=2)
        similarity_matrix = torch.bmm(last_hidden_state_normalized,  torch.transpose(last_hidden_state_normalized, 1, 2))  # [batch, seq_len, seq_len]

        #similarity_matrix = self.cos_sim(last_hidden_state, last_hidden_state)  # [batch, seq_len, seq_len]
        # doc_words_mask [batch, seq_len]
        doc_words_mask_expanded = doc_words_mask.unsqueeze(1).expand(-1, similarity_matrix.size(1), -1)  # [batch, seq_len,seq_len]
        # inverted and mask to negative number
        doc_words_mask_expanded_inverted = doc_words_mask_expanded < 0.5 * -10000.0

        # similarity_matrix_masked = similarity_matrix * doc_words_mask_expanded  # [batch, seq_len, seq_len]
        similarity_matrix_masked = similarity_matrix + doc_words_mask_expanded_inverted  # [batch, seq_len, seq_len]
        similarity_max_pooled, _ = similarity_matrix_masked.max(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        # claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        claim_words_similarities_sum = torch.bmm(claim_frames_word_mask, similarity_max_pooled).squeeze(2)  # [batch, n_claim_frames]
        # mean_pool_mask [batch, n_claim_frames, hid]
        mean_pool_mask_shrinked = mean_pool_mask[:, :, 0]  # [batch, n_claim_frames]
        claim_words_precision = torch.zeros_like(claim_words_similarities_sum)  # [batch, n_claim_frames]
        # num_claim_frame_words [batch, n_claim_frames, 1]
        claim_words_precision[mean_pool_mask_shrinked] = claim_words_similarities_sum[mean_pool_mask_shrinked] / num_claim_frame_words.squeeze(-1)[
            mean_pool_mask_shrinked]  # [batch, n_claim_frames]
        """
        print("claim_words_precision")
        print(claim_words_precision.size())
        print(claim_words_precision[0])
        """
        claim_words_precision = claim_words_precision.unsqueeze(2)  # [batch, n_claim_frames, 1]

        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features, claim_words_precision], dim=2)  #  [batch, n_claim_frames, 2*hid_size+1]

        claim_frame_weights = self.frame_classification_features_attn(frame_classification_features)  #  [batch, n_claim_frames, 1]

        claim_frame_weights_normalized = F.softmax(claim_frame_weights, dim=1)  # [batch_size, n_claim_frames, 1]

        claim_frame_weights_normalized_expanded = claim_frame_weights_normalized.expand(-1, -1, frame_classification_features.size(2))  #  [batch, n_claim_frames, 2*hid_size+1]

        frame_classification_features_pooled = torch.sum(frame_classification_features * claim_frame_weights_normalized_expanded, dim=1)  #  [batch, 2*hid_size+1]

        #frame_classification_features_mean_pooled = frame_classification_features.mean(dim=1)  #  [batch, 2*hid_size+1]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]

        all_classification_feature = torch.cat([cls_representation, frame_classification_features_pooled], dim=1)  #  [batch, 3*hid_size+1]

        logits = self.classifier(self.dropout(all_classification_feature))  # [batch, num_classes]

        outputs = (logits, claim_attn_output_weights) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMeanMultiLabelAdapterClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMeanMultiLabelAdapterClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, self.num_labels)
        #self.classifier_attn = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax = MaskedSoftmax(dim=1)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        #self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilCosSimMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        """
        print("last_hidden_state")
        print(last_hidden_state[0, 0, :])
        print(last_hidden_state[0, 1, :])
        print(last_hidden_state[0, -1, :])
        """

        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        last_hidden_state_projected = self.frame_feature_project(
            last_hidden_state)  # [batch_size, sequence_length, hidden_size]

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        """
        print("doc_frames_word_mask")
        print(doc_frames_word_mask[0,0,:])
        print(doc_frames_word_mask[0,1,:])
        print("non empty doc frames")
        print((num_doc_frame_words[0, :, 0] > 0).sum())
        print(num_doc_frame_words[0, : 0])
        """

        doc_frame_features_sum = torch.bmm(doc_frames_word_mask,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
            mean_pool_mask]  # [batch_size, n_frames, hid]

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]
        claim_frame_features_sum = torch.bmm(claim_frames_word_mask,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                             claim_frame_features_sum.size(
                                                                 2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        """
        print("claim_attn_mask")  # [batch_size, n_doc_frames]
        print(claim_attn_mask.size())
        print(claim_attn_mask[0, :])
        print("num_claim_frame_words")
        print(num_claim_frame_words[0, 0, 0])
        print(num_claim_frame_words[0, 1, 0])
        print(num_claim_frame_words[0, 2, 0])
        print("claim_frame_features")
        print(claim_frame_features[0, 0, :])
        print(claim_frame_features[0, 1, :])
        print(claim_frame_features[0, 2, :])
        print(claim_frame_features[0, -1, :])
        """

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding

        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features], dim=2)  #  [batch, n_claim_frames, 2*hid_size]

        frame_classification_features_mean_pooled = frame_classification_features.mean(dim=1)  #  [batch, 2*hid_size]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]

        all_classification_feature = torch.cat([cls_representation, frame_classification_features_mean_pooled], dim=1)  #  [batch, 3*hid_size+1]

        logits = self.classifier(self.dropout(all_classification_feature))  # [batch, num_classes]

        outputs = (logits, claim_attn_output_weights) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMilCosSimMultiLabelAdapterWordAttnClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMilCosSimMultiLabelAdapterWordAttnClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        config.output_hidden_states = True
        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2 + 1, self.num_labels)
        self.classifier_attn = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax_dim_1 = MaskedSoftmax(dim=1)
        self.masked_softmax_dim_2 = MaskedSoftmax(dim=2)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_attn = nn.Linear(config.hidden_size, 1)
        self.token_self_attn = nn.Linear(config.hidden_size, 1)
        self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilCosSimMultiLabelAdapterWordAttnClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        all_layers_hidden_state = outputs[2]  # tuple of [batch_size, sequence_length, hidden_size]
        all_layers_hidden_state_tensor = torch.stack(all_layers_hidden_state, dim=2)  # [batch_size, sequence_length, num_layers, hidden_size]
        all_layers_hidden_state_tensor = self.frame_feature_project(all_layers_hidden_state_tensor)

        #print("all_layers_hidden_state_tensor")
        #print(all_layers_hidden_state_tensor.size())
        #embedding_layers_hidden_state = all_layers_hidden_state[0]
        """
        print("last_hidden_state")
        print(last_hidden_state[0, 0, :])
        print(last_hidden_state[0, 1, :])
        print(last_hidden_state[0, -1, :])
        """
        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        #hidden_state_fused = 0.5 * embedding_layers_hidden_state + 0.5 * last_hidden_state

        # Layer attention
        layer_attn_weights_normalized = F.softmax(self.layer_attn(all_layers_hidden_state_tensor), dim=2)  # [batch_size, sequence_length, num_layers, 1]
        layer_attn_weights_normalized_expanded = layer_attn_weights_normalized.expand(-1, -1, -1, all_layers_hidden_state_tensor.size(3))  # [batch_size, sequence_length, num_layers, hid_size]
        hidden_state_fused = torch.sum(all_layers_hidden_state_tensor * layer_attn_weights_normalized_expanded, dim=2)  # [batch_size, sequence_length, hid_size]
        #print("hidden_state_fused")
        #print(hidden_state_fused.size())
        #exit()
        last_hidden_state_projected = hidden_state_fused
        #last_hidden_state_projected = self.frame_feature_project(hidden_state_fused)  # [batch_size, sequence_length, hidden_size]

        token_attn_score = self.token_self_attn(hidden_state_fused).squeeze()  # [batch_size, sequence_length]

        token_attn_score_resized_for_doc_frames = token_attn_score.unsqueeze(1).expand(-1, doc_frames_word_mask.size(1), -1)  # [batch, n_doc_frames, seq_len]
        doc_frames_word_mask_multiplied = doc_frames_word_mask * token_attn_score_resized_for_doc_frames  # [batch, n_doc_frames, seq_len]
        #doc_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(doc_frames_word_mask_multiplied, -1)  # [batch, n_doc_frames, seq_len]

        doc_frames_word_mask_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)
        doc_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        num_doc_frame_words_expanded_seq_len = num_doc_frame_words.expand(-1, -1,
                                                                              doc_frames_word_mask.size(
                                                                                  2))  # [batch, n_claim_frames, seq_len]

        doc_frames_word_mask_all_zero_mask = (
                    num_doc_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        doc_frames_word_mask_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = doc_frames_word_mask[
            doc_frames_word_mask_all_zero_mask]
        doc_frames_word_mask_multiplied_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = \
        doc_frames_word_mask_multiplied[doc_frames_word_mask_all_zero_mask]

        doc_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(doc_frames_word_mask_multiplied_fixed_all_zeros,
                                                                              doc_frames_word_mask_fixed_all_zeros)  # [batch, n_doc_frames, seq_len]

        """
        print("doc_frames_word_mask_multiplied")
        print(doc_frames_word_mask_multiplied_softmaxed[0, 0, :])
        print(doc_frames_word_mask_multiplied_softmaxed[0, 1, :])
        """
        """
        print("token_attn_score_resized_for_doc_frames")
        print(token_attn_score_resized_for_doc_frames[0, 0, :])
        print("doc_frames_word_mask_multiplied")
        print(doc_frames_word_mask_multiplied_fixed_all_zeros[0, 0, :])
        print(doc_frames_word_mask_multiplied_fixed_all_zeros[0, -1, :])
        print("doc_frames_word_mask_multiplied_softmaxed")
        print(doc_frames_word_mask_multiplied_softmaxed[0,0,:])
        print(doc_frames_word_mask_multiplied_softmaxed[0, -1, :])
        #print(doc_frames_word_mask_multiplied_softmaxed[0,1,:])
        #print("non empty doc frames")
        #print((num_doc_frame_words[0, :, 0] > 0).sum())
        #print(num_doc_frame_words[0, : 0])
        """


        doc_frame_features_sum = torch.bmm(doc_frames_word_mask_multiplied_softmaxed,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask]
        #doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
        #    mean_pool_mask]  # [batch_size, n_frames, hid]

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        token_attn_score_resized_for_claim_frames = token_attn_score.unsqueeze(1).expand(-1,
                                                                 claim_frames_word_mask.size(1),
                                                                 -1)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_multiplied = claim_frames_word_mask * token_attn_score_resized_for_claim_frames  # [batch, n_claim_frames, seq_len]
        #claim_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(claim_frames_word_mask_multiplied, -1)  # [batch, n_claim_frames, seq_len]

        claim_frames_word_mask_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)
        claim_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]

        num_claim_frame_words_expanded_seq_len = num_claim_frame_words.expand(-1, -1,
                                                                      claim_frames_word_mask.size(
                                                                          2))  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_all_zero_mask = (num_claim_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = claim_frames_word_mask[
            claim_frames_word_mask_all_zero_mask]
        claim_frames_word_mask_multiplied_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = claim_frames_word_mask_multiplied[claim_frames_word_mask_all_zero_mask]

        claim_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(claim_frames_word_mask_multiplied_fixed_all_zeros, claim_frames_word_mask_fixed_all_zeros)  # [batch, n_claim_frames, seq_len]


        claim_frame_features_sum = torch.bmm(claim_frames_word_mask_multiplied_softmaxed,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                             claim_frame_features_sum.size(
                                                                 2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        #claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
        #    mean_pool_mask]  # [batch_size, n_claim_frames, hid]
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask]  # [batch_size, n_claim_frames, hid]
        """
        print("claim_frames_word_mask_multiplied_softmaxed")  # [batch_size, n_doc_frames]
        print(claim_frames_word_mask_multiplied_softmaxed[0, 0, :])
        print(claim_frames_word_mask_multiplied_softmaxed[0, 1, :])
        print()
        """
        """
        print("claim_frames_word_mask_fixed_all_zeros")
        print(claim_frames_word_mask_fixed_all_zeros[0, 0, :])
        print(claim_frames_word_mask_fixed_all_zeros[0, -1, :])
        print("claim_frames_word_mask_multiplied_fixed_all_zeros")
        print(claim_frames_word_mask_multiplied_fixed_all_zeros[0, 0, :])
        print(claim_frames_word_mask_multiplied_fixed_all_zeros[0, -1, :])
        print("claim_frames_word_mask_multiplied_softmaxed")  # [batch_size, n_doc_frames]
        print(claim_frames_word_mask_multiplied_softmaxed.size())
        print(claim_frames_word_mask_multiplied_softmaxed[0, 0, :])
        print(claim_frames_word_mask_multiplied_softmaxed[0, -1, :])
        print("num_claim_frame_words")
        print(num_claim_frame_words[0, 0, 0])
        print(num_claim_frame_words[0, 1, 0])
        print(num_claim_frame_words[0, 2, 0])
        print("claim_frame_features")
        print(claim_frame_features[0, 0, :])
        print(claim_frame_features[0, 1, :])
        print(claim_frame_features[0, 2, :])
        print(claim_frame_features[0, -1, :])
        """
        

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding

        # features of cosine similarity
        last_hidden_state_normalized = nn.functional.normalize(last_hidden_state, p=2, dim=2)
        similarity_matrix = torch.bmm(last_hidden_state_normalized,  torch.transpose(last_hidden_state_normalized, 1, 2))  # [batch, seq_len, seq_len]

        #similarity_matrix = self.cos_sim(last_hidden_state, last_hidden_state)  # [batch, seq_len, seq_len]
        # doc_words_mask [batch, seq_len]
        doc_words_mask_expanded = doc_words_mask.unsqueeze(1).expand(-1, similarity_matrix.size(1), -1)  # [batch, seq_len,seq_len]
        # inverted and mask to negative number
        doc_words_mask_expanded_inverted = doc_words_mask_expanded < 0.5 * -10000.0

        # similarity_matrix_masked = similarity_matrix * doc_words_mask_expanded  # [batch, seq_len, seq_len]
        similarity_matrix_masked = similarity_matrix + doc_words_mask_expanded_inverted  # [batch, seq_len, seq_len]
        similarity_max_pooled, _ = similarity_matrix_masked.max(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        # claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        claim_words_similarities_sum = torch.bmm(claim_frames_word_mask, similarity_max_pooled).squeeze(2)  # [batch, n_claim_frames]
        # mean_pool_mask [batch, n_claim_frames, hid]
        mean_pool_mask_shrinked = mean_pool_mask[:, :, 0]  # [batch, n_claim_frames]
        claim_words_precision = torch.zeros_like(claim_words_similarities_sum)  # [batch, n_claim_frames]
        # num_claim_frame_words [batch, n_claim_frames, 1]
        """
        print("similarity_max_pooled")
        print(similarity_max_pooled.size())
        print(similarity_max_pooled[0])
        print("claim_words_similarities_sum")
        print(claim_words_similarities_sum.size())
        print(claim_words_similarities_sum[0])
        print("claim_words_precision")
        print(claim_words_precision.size())
        print("mean_pool_mask_shrinked")
        print(mean_pool_mask_shrinked.size())
        print(mean_pool_mask_shrinked[0])
        """
        claim_words_precision[mean_pool_mask_shrinked] = claim_words_similarities_sum[mean_pool_mask_shrinked] / num_claim_frame_words.squeeze(-1)[
            mean_pool_mask_shrinked]  # [batch, n_claim_frames]
        """
        print("claim_words_precision")
        print(claim_words_precision.size())
        print(claim_words_precision[0])
        """
        claim_words_precision = claim_words_precision.unsqueeze(2)  # [batch, n_claim_frames, 1]

        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features, claim_words_precision], dim=2)  #  [batch, n_claim_frames, 2*hid_size+1]
        claim_frame_logits = self.classifier(self.dropout(frame_classification_features))  # [batch, n_claim_frames, num_classes]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]
        # expand cls representation
        cls_representation_expanded = cls_representation.unsqueeze(1).expand(-1, claim_frame_features.size(1), -1)  # [batch, n_claim_frames, hid_size]

        classifier_attn_weights = self.classifier_attn(self.dropout(torch.cat([cls_representation_expanded, frame_classification_features], dim=2)))  # [batch, n_claim_frames, num_classes]
        # claim_frames_padding_mask [batch, n_claim_frames]
        claim_frames_padding_mask_expanded = claim_frames_padding_mask.unsqueeze(2).expand(-1, -1, classifier_attn_weights.size(2))
        # claim_frames_padding_mask [batch, n_claim_frames, num_classes]
        classifier_attn_weights = self.masked_softmax_dim_1(classifier_attn_weights, mask=claim_frames_padding_mask_expanded)  # [batch, n_claim_frames, num_classes]
        logits = (claim_frame_logits * classifier_attn_weights).sum(dim=1)  # [batch, num_classes]

        #logits = self.classifier(claim_attn_output_mean_pooled)
        # [batch, n_claim_frames, hid] ->  # [batch, hid]

        outputs = (logits, claim_frame_logits, classifier_attn_weights, claim_attn_output_weights) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            #print("loss")
            #print(loss.item())
            #exit()
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMilCosSimMultiLabelAdapterWordAttnSimpleClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMilCosSimMultiLabelAdapterWordAttnSimpleClassifier, self).__init__(config)
        self.num_labels = config.num_labels
        config.output_hidden_states = True
        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2 + 1, self.num_labels)
        self.classifier_attn = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax_dim_1 = MaskedSoftmax(dim=1)
        self.masked_softmax_dim_2 = MaskedSoftmax(dim=2)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        #self.token_self_attn = nn.Linear(config.hidden_size, 1)
        self.token_self_attn = nn.Sequential( nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(), nn.Linear(config.hidden_size, 1) )
        self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilCosSimMultiLabelAdapterWordAttnSimpleClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        all_layers_hidden_state = outputs[2]
        #embedding_layers_hidden_state = all_layers_hidden_state[0]
        """
        print("last_hidden_state")
        print(last_hidden_state[0, 0, :])
        print(last_hidden_state[0, 1, :])
        print(last_hidden_state[0, -1, :])
        """
        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        #hidden_state_fused = 0.5 * embedding_layers_hidden_state + 0.5 * last_hidden_state
        hidden_state_fused = self.frame_feature_project(last_hidden_state)
        last_hidden_state_projected = hidden_state_fused
        #last_hidden_state_projected = self.frame_feature_project(
        #    hidden_state_fused)  # [batch_size, sequence_length, hidden_size]

        token_attn_score = self.token_self_attn(hidden_state_fused).squeeze()  # [batch_size, sequence_length]

        token_attn_score_resized_for_doc_frames = token_attn_score.unsqueeze(1).expand(-1, doc_frames_word_mask.size(1),
                                                                                       -1)  # [batch, n_doc_frames, seq_len]
        doc_frames_word_mask_multiplied = doc_frames_word_mask * token_attn_score_resized_for_doc_frames  # [batch, n_doc_frames, seq_len]
        # doc_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(doc_frames_word_mask_multiplied, -1)  # [batch, n_doc_frames, seq_len]

        doc_frames_word_mask_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)
        doc_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(doc_frames_word_mask)

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        num_doc_frame_words_expanded_seq_len = num_doc_frame_words.expand(-1, -1,
                                                                          doc_frames_word_mask.size(
                                                                              2))  # [batch, n_claim_frames, seq_len]

        doc_frames_word_mask_all_zero_mask = (
                num_doc_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        doc_frames_word_mask_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = doc_frames_word_mask[
            doc_frames_word_mask_all_zero_mask]
        doc_frames_word_mask_multiplied_fixed_all_zeros[doc_frames_word_mask_all_zero_mask] = \
            doc_frames_word_mask_multiplied[doc_frames_word_mask_all_zero_mask]

        doc_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            doc_frames_word_mask_multiplied_fixed_all_zeros,
            doc_frames_word_mask_fixed_all_zeros)  # [batch, n_doc_frames, seq_len]

        """
        print("doc_frames_word_mask_multiplied")
        print(doc_frames_word_mask_multiplied_softmaxed[0, 0, :])
        print(doc_frames_word_mask_multiplied_softmaxed[0, 1, :])
        """
        """
        print("token_attn_score_resized_for_doc_frames")
        print(token_attn_score_resized_for_doc_frames[0, 0, :])
        print("doc_frames_word_mask_multiplied")
        print(doc_frames_word_mask_multiplied_fixed_all_zeros[0, 0, :])
        print(doc_frames_word_mask_multiplied_fixed_all_zeros[0, -1, :])
        print("doc_frames_word_mask_multiplied_softmaxed")
        print(doc_frames_word_mask_multiplied_softmaxed[0,0,:])
        print(doc_frames_word_mask_multiplied_softmaxed[0, -1, :])
        #print(doc_frames_word_mask_multiplied_softmaxed[0,1,:])
        #print("non empty doc frames")
        #print((num_doc_frame_words[0, :, 0] > 0).sum())
        #print(num_doc_frame_words[0, : 0])
        """

        doc_frame_features_sum = torch.bmm(doc_frames_word_mask_multiplied_softmaxed,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask]
        # doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
        #    mean_pool_mask]  # [batch_size, n_frames, hid]

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        token_attn_score_resized_for_claim_frames = token_attn_score.unsqueeze(1).expand(-1,
                                                                                         claim_frames_word_mask.size(1),
                                                                                         -1)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_multiplied = claim_frames_word_mask * token_attn_score_resized_for_claim_frames  # [batch, n_claim_frames, seq_len]
        # claim_frames_word_mask_multiplied_softmaxed = nn.functional.softmax(claim_frames_word_mask_multiplied, -1)  # [batch, n_claim_frames, seq_len]

        claim_frames_word_mask_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)
        claim_frames_word_mask_multiplied_fixed_all_zeros = torch.ones_like(claim_frames_word_mask)

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]

        num_claim_frame_words_expanded_seq_len = num_claim_frame_words.expand(-1, -1,
                                                                              claim_frames_word_mask.size(
                                                                                  2))  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_all_zero_mask = (
                    num_claim_frame_words_expanded_seq_len != 0)  # [batch, n_claim_frames, seq_len]
        claim_frames_word_mask_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = claim_frames_word_mask[
            claim_frames_word_mask_all_zero_mask]
        claim_frames_word_mask_multiplied_fixed_all_zeros[claim_frames_word_mask_all_zero_mask] = \
        claim_frames_word_mask_multiplied[claim_frames_word_mask_all_zero_mask]

        claim_frames_word_mask_multiplied_softmaxed = self.masked_softmax_dim_2(
            claim_frames_word_mask_multiplied_fixed_all_zeros,
            claim_frames_word_mask_fixed_all_zeros)  # [batch, n_claim_frames, seq_len]

        claim_frame_features_sum = torch.bmm(claim_frames_word_mask_multiplied_softmaxed,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                                      claim_frame_features_sum.size(
                                                                          2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        # claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
        #    mean_pool_mask]  # [batch_size, n_claim_frames, hid]
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]
        """
        print("claim_frames_word_mask_multiplied_softmaxed")  # [batch_size, n_doc_frames]
        print(claim_frames_word_mask_multiplied_softmaxed[0, 0, :])
        print(claim_frames_word_mask_multiplied_softmaxed[0, 1, :])
        print()
        """
        """
        print("claim_frames_word_mask_fixed_all_zeros")
        print(claim_frames_word_mask_fixed_all_zeros[0, 0, :])
        print(claim_frames_word_mask_fixed_all_zeros[0, -1, :])
        print("claim_frames_word_mask_multiplied_fixed_all_zeros")
        print(claim_frames_word_mask_multiplied_fixed_all_zeros[0, 0, :])
        print(claim_frames_word_mask_multiplied_fixed_all_zeros[0, -1, :])
        print("claim_frames_word_mask_multiplied_softmaxed")  # [batch_size, n_doc_frames]
        print(claim_frames_word_mask_multiplied_softmaxed.size())
        print(claim_frames_word_mask_multiplied_softmaxed[0, 0, :])
        print(claim_frames_word_mask_multiplied_softmaxed[0, -1, :])
        print("num_claim_frame_words")
        print(num_claim_frame_words[0, 0, 0])
        print(num_claim_frame_words[0, 1, 0])
        print(num_claim_frame_words[0, 2, 0])
        print("claim_frame_features")
        print(claim_frame_features[0, 0, :])
        print(claim_frame_features[0, 1, :])
        print(claim_frame_features[0, 2, :])
        print(claim_frame_features[0, -1, :])
        """

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding

        # features of cosine similarity
        last_hidden_state_normalized = nn.functional.normalize(last_hidden_state, p=2, dim=2)
        similarity_matrix = torch.bmm(last_hidden_state_normalized,
                                      torch.transpose(last_hidden_state_normalized, 1, 2))  # [batch, seq_len, seq_len]

        # similarity_matrix = self.cos_sim(last_hidden_state, last_hidden_state)  # [batch, seq_len, seq_len]
        # doc_words_mask [batch, seq_len]
        doc_words_mask_expanded = doc_words_mask.unsqueeze(1).expand(-1, similarity_matrix.size(1),
                                                                     -1)  # [batch, seq_len,seq_len]
        # inverted and mask to negative number
        doc_words_mask_expanded_inverted = doc_words_mask_expanded < 0.5 * -10000.0

        # similarity_matrix_masked = similarity_matrix * doc_words_mask_expanded  # [batch, seq_len, seq_len]
        similarity_matrix_masked = similarity_matrix + doc_words_mask_expanded_inverted  # [batch, seq_len, seq_len]
        similarity_max_pooled, _ = similarity_matrix_masked.max(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        # claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        claim_words_similarities_sum = torch.bmm(claim_frames_word_mask, similarity_max_pooled).squeeze(
            2)  # [batch, n_claim_frames]
        # mean_pool_mask [batch, n_claim_frames, hid]
        mean_pool_mask_shrinked = mean_pool_mask[:, :, 0]  # [batch, n_claim_frames]
        claim_words_precision = torch.zeros_like(claim_words_similarities_sum)  # [batch, n_claim_frames]
        # num_claim_frame_words [batch, n_claim_frames, 1]
        """
        print("similarity_max_pooled")
        print(similarity_max_pooled.size())
        print(similarity_max_pooled[0])
        print("claim_words_similarities_sum")
        print(claim_words_similarities_sum.size())
        print(claim_words_similarities_sum[0])
        print("claim_words_precision")
        print(claim_words_precision.size())
        print("mean_pool_mask_shrinked")
        print(mean_pool_mask_shrinked.size())
        print(mean_pool_mask_shrinked[0])
        """
        claim_words_precision[mean_pool_mask_shrinked] = claim_words_similarities_sum[mean_pool_mask_shrinked] / \
                                                         num_claim_frame_words.squeeze(-1)[
                                                             mean_pool_mask_shrinked]  # [batch, n_claim_frames]
        """
        print("claim_words_precision")
        print(claim_words_precision.size())
        print(claim_words_precision[0])
        """
        claim_words_precision = claim_words_precision.unsqueeze(2)  # [batch, n_claim_frames, 1]

        # claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features, claim_words_precision],
                                                  dim=2)  # [batch, n_claim_frames, 2*hid_size+1]
        claim_frame_logits = self.classifier(
            self.dropout(frame_classification_features))  # [batch, n_claim_frames, num_classes]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]
        # expand cls representation
        cls_representation_expanded = cls_representation.unsqueeze(1).expand(-1, claim_frame_features.size(1),
                                                                             -1)  # [batch, n_claim_frames, hid_size]

        classifier_attn_weights = self.classifier_attn(self.dropout(
            torch.cat([cls_representation_expanded, frame_classification_features],
                      dim=2)))  # [batch, n_claim_frames, num_classes]
        # claim_frames_padding_mask [batch, n_claim_frames]
        claim_frames_padding_mask_expanded = claim_frames_padding_mask.unsqueeze(2).expand(-1, -1,
                                                                                           classifier_attn_weights.size(
                                                                                               2))
        # claim_frames_padding_mask [batch, n_claim_frames, num_classes]
        classifier_attn_weights = self.masked_softmax_dim_1(classifier_attn_weights,
                                                            mask=claim_frames_padding_mask_expanded)  # [batch, n_claim_frames, num_classes]
        logits = (claim_frame_logits * classifier_attn_weights).sum(dim=1)  # [batch, num_classes]

        # logits = self.classifier(claim_attn_output_mean_pooled)
        # [batch, n_claim_frames, hid] ->  # [batch, hid]

        outputs = (logits, claim_frame_logits, classifier_attn_weights, claim_attn_output_weights) + outputs[
                                                                                                     2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # print("loss")
            # print(loss.item())
            # exit()
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMeanMilCosSimMultiLabelAdapterClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMeanMilCosSimMultiLabelAdapterClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2 + 1, self.num_labels)
        #self.classifier_attn = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax = MaskedSoftmax(dim=1)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilCosSimMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        """
        print("last_hidden_state")
        print(last_hidden_state[0, 0, :])
        print(last_hidden_state[0, 1, :])
        print(last_hidden_state[0, -1, :])
        """

        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        last_hidden_state_projected = self.frame_feature_project(
            last_hidden_state)  # [batch_size, sequence_length, hidden_size]

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        """
        print("doc_frames_word_mask")
        print(doc_frames_word_mask[0,0,:])
        print(doc_frames_word_mask[0,1,:])
        print("non empty doc frames")
        print((num_doc_frame_words[0, :, 0] > 0).sum())
        print(num_doc_frame_words[0, : 0])
        """

        doc_frame_features_sum = torch.bmm(doc_frames_word_mask,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
            mean_pool_mask]  # [batch_size, n_frames, hid]

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]
        claim_frame_features_sum = torch.bmm(claim_frames_word_mask,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                             claim_frame_features_sum.size(
                                                                 2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding

        # features of cosine similarity
        last_hidden_state_normalized = nn.functional.normalize(last_hidden_state, p=2, dim=2)
        similarity_matrix = torch.bmm(last_hidden_state_normalized,  torch.transpose(last_hidden_state_normalized, 1, 2))  # [batch, seq_len, seq_len]

        #similarity_matrix = self.cos_sim(last_hidden_state, last_hidden_state)  # [batch, seq_len, seq_len]
        # doc_words_mask [batch, seq_len]
        doc_words_mask_expanded = doc_words_mask.unsqueeze(1).expand(-1, similarity_matrix.size(1), -1)  # [batch, seq_len,seq_len]
        # inverted and mask to negative number
        doc_words_mask_expanded_inverted = doc_words_mask_expanded < 0.5 * -10000.0

        #similarity_matrix_masked = similarity_matrix * doc_words_mask_expanded  # [batch, seq_len, seq_len]
        similarity_matrix_masked = similarity_matrix + doc_words_mask_expanded_inverted  # [batch, seq_len, seq_len]
        similarity_max_pooled, _ = similarity_matrix_masked.max(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        # claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        claim_words_similarities_sum = torch.bmm(claim_frames_word_mask, similarity_max_pooled).squeeze(2)  # [batch, n_claim_frames]
        # mean_pool_mask [batch, n_claim_frames, hid]
        mean_pool_mask_shrinked = mean_pool_mask[:, :, 0]  # [batch, n_claim_frames]
        claim_words_precision = torch.zeros_like(claim_words_similarities_sum)  # [batch, n_claim_frames]
        # num_claim_frame_words [batch, n_claim_frames, 1]

        claim_words_precision[mean_pool_mask_shrinked] = claim_words_similarities_sum[mean_pool_mask_shrinked] / num_claim_frame_words.squeeze(-1)[
            mean_pool_mask_shrinked]  # [batch, n_claim_frames]
        """
        print("claim_words_precision")
        print(claim_words_precision.size())
        print(claim_words_precision[0])
        """
        claim_words_precision = claim_words_precision.unsqueeze(2)  # [batch, n_claim_frames, 1]

        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features, claim_words_precision], dim=2)  #  [batch, n_claim_frames, 2*hid_size+1]
        claim_frame_logits = self.classifier(self.dropout(frame_classification_features))  # [batch, n_claim_frames, num_classes]

        #cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]
        # expand cls representation
        #cls_representation_expanded = cls_representation.unsqueeze(1).expand(-1, claim_frame_features.size(1), -1)  # [batch, n_claim_frames, hid_size]

        #classifier_attn_weights = self.classifier_attn(self.dropout(torch.cat([cls_representation_expanded, frame_classification_features], dim=2)))  # [batch, n_claim_frames, num_classes]
        # claim_frames_padding_mask [batch, n_claim_frames]
        claim_frames_padding_mask_expanded = claim_frames_padding_mask.unsqueeze(2).expand(-1, -1, claim_frame_logits.size(2))
        # claim_frames_padding_mask_expanded [batch, n_claim_frames, num_classes]

        # mean pooling of the claim_frame_logits
        claim_frame_logits_masked_sum = (claim_frame_logits * claim_frames_padding_mask_expanded).sum(dim=1)  # [batch, num_classes]
        claim_frame_logits_mean_pooled = claim_frame_logits_masked_sum / claim_frames_padding_mask_expanded.sum(dim=1)  # [batch, num_classes]
        logits = claim_frame_logits_mean_pooled  # [batch, num_classes]

        #classifier_attn_weights = self.masked_softmax(classifier_attn_weights, mask=claim_frames_padding_mask_expanded)  # [batch, n_claim_frames, num_classes]
        #logits = (claim_frame_logits * classifier_attn_weights).sum(dim=1)  # [batch, num_classes]

        #logits = self.classifier(claim_attn_output_mean_pooled)
        # [batch, n_claim_frames, hid] ->  # [batch, hid]

        outputs = (logits, claim_frame_logits, claim_attn_output_weights) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)




class BertClaimAttnMeanMilMultiLabelAdapterClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMeanMilMultiLabelAdapterClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        #self.classifier_attn = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax = MaskedSoftmax(dim=1)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilCosSimMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        """
        print("last_hidden_state")
        print(last_hidden_state[0, 0, :])
        print(last_hidden_state[0, 1, :])
        print(last_hidden_state[0, -1, :])
        """

        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        last_hidden_state_projected = self.frame_feature_project(
            last_hidden_state)  # [batch_size, sequence_length, hidden_size]

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        """
        print("doc_frames_word_mask")
        print(doc_frames_word_mask[0,0,:])
        print(doc_frames_word_mask[0,1,:])
        print("non empty doc frames")
        print((num_doc_frame_words[0, :, 0] > 0).sum())
        print(num_doc_frame_words[0, : 0])
        """

        doc_frame_features_sum = torch.bmm(doc_frames_word_mask,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
            mean_pool_mask]  # [batch_size, n_frames, hid]

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]
        claim_frame_features_sum = torch.bmm(claim_frames_word_mask,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                             claim_frame_features_sum.size(
                                                                 2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding

        # features of cosine similarity
        last_hidden_state_normalized = nn.functional.normalize(last_hidden_state, p=2, dim=2)
        similarity_matrix = torch.bmm(last_hidden_state_normalized,  torch.transpose(last_hidden_state_normalized, 1, 2))  # [batch, seq_len, seq_len]

        #similarity_matrix = self.cos_sim(last_hidden_state, last_hidden_state)  # [batch, seq_len, seq_len]
        # doc_words_mask [batch, seq_len]
        #doc_words_mask_expanded = doc_words_mask.unsqueeze(1).expand(-1, similarity_matrix.size(1), -1)  # [batch, seq_len,seq_len]
        # inverted and mask to negative number
        #doc_words_mask_expanded_inverted = doc_words_mask_expanded < 0.5 * -10000.0

        #similarity_matrix_masked = similarity_matrix + doc_words_mask_expanded_inverted  # [batch, seq_len, seq_len]
        #similarity_max_pooled, _ = similarity_matrix_masked.max(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        # claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        #claim_words_similarities_sum = torch.bmm(claim_frames_word_mask, similarity_max_pooled).squeeze(2)  # [batch, n_claim_frames]
        ## mean_pool_mask [batch, n_claim_frames, hid]
        #mean_pool_mask_shrinked = mean_pool_mask[:, :, 0]  # [batch, n_claim_frames]
        #claim_words_precision = torch.zeros_like(claim_words_similarities_sum)  # [batch, n_claim_frames]
        # num_claim_frame_words [batch, n_claim_frames, 1]

        #claim_words_precision[mean_pool_mask_shrinked] = claim_words_similarities_sum[mean_pool_mask_shrinked] / num_claim_frame_words.squeeze(-1)[
        #    mean_pool_mask_shrinked]  # [batch, n_claim_frames]
        #claim_words_precision = claim_words_precision.unsqueeze(2)  # [batch, n_claim_frames, 1]

        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features], dim=2)  #  [batch, n_claim_frames, 2*hid_size+1]
        claim_frame_logits = self.classifier(self.dropout(frame_classification_features))  # [batch, n_claim_frames, num_classes]

        #cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]
        # expand cls representation
        #cls_representation_expanded = cls_representation.unsqueeze(1).expand(-1, claim_frame_features.size(1), -1)  # [batch, n_claim_frames, hid_size]

        #classifier_attn_weights = self.classifier_attn(self.dropout(torch.cat([cls_representation_expanded, frame_classification_features], dim=2)))  # [batch, n_claim_frames, num_classes]
        # claim_frames_padding_mask [batch, n_claim_frames]
        claim_frames_padding_mask_expanded = claim_frames_padding_mask.unsqueeze(2).expand(-1, -1, claim_frame_logits.size(2))
        # claim_frames_padding_mask_expanded [batch, n_claim_frames, num_classes]

        # mean pooling of the claim_frame_logits
        claim_frame_logits_masked_sum = (claim_frame_logits * claim_frames_padding_mask_expanded).sum(dim=1)  # [batch, num_classes]
        claim_frame_logits_mean_pooled = claim_frame_logits_masked_sum / claim_frames_padding_mask_expanded.sum(dim=1)  # [batch, num_classes]
        logits = claim_frame_logits_mean_pooled  # [batch, num_classes]

        #classifier_attn_weights = self.masked_softmax(classifier_attn_weights, mask=claim_frames_padding_mask_expanded)  # [batch, n_claim_frames, num_classes]
        #logits = (claim_frame_logits * classifier_attn_weights).sum(dim=1)  # [batch, num_classes]

        #logits = self.classifier(claim_attn_output_mean_pooled)
        # [batch, n_claim_frames, hid] ->  # [batch, hid]

        outputs = (logits, claim_frame_logits, claim_attn_output_weights) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMaxMilCosSimMultiLabelAdapterClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMaxMilCosSimMultiLabelAdapterClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2 + 1, self.num_labels)
        #self.classifier_attn = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax = MaskedSoftmax(dim=1)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMaxMilCosSimMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        """
        print("last_hidden_state")
        print(last_hidden_state[0, 0, :])
        print(last_hidden_state[0, 1, :])
        print(last_hidden_state[0, -1, :])
        """

        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        last_hidden_state_projected = self.frame_feature_project(
            last_hidden_state)  # [batch_size, sequence_length, hidden_size]

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        """
        print("doc_frames_word_mask")
        print(doc_frames_word_mask[0,0,:])
        print(doc_frames_word_mask[0,1,:])
        print("non empty doc frames")
        print((num_doc_frame_words[0, :, 0] > 0).sum())
        print(num_doc_frame_words[0, : 0])
        """

        doc_frame_features_sum = torch.bmm(doc_frames_word_mask,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
            mean_pool_mask]  # [batch_size, n_frames, hid]

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]
        claim_frame_features_sum = torch.bmm(claim_frames_word_mask,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                             claim_frame_features_sum.size(
                                                                 2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding

        # features of cosine similarity
        last_hidden_state_normalized = nn.functional.normalize(last_hidden_state, p=2, dim=2)
        similarity_matrix = torch.bmm(last_hidden_state_normalized,  torch.transpose(last_hidden_state_normalized, 1, 2))  # [batch, seq_len, seq_len]

        #similarity_matrix = self.cos_sim(last_hidden_state, last_hidden_state)  # [batch, seq_len, seq_len]
        # doc_words_mask [batch, seq_len]
        doc_words_mask_expanded = doc_words_mask.unsqueeze(1).expand(-1, similarity_matrix.size(1), -1)  # [batch, seq_len,seq_len]

        similarity_matrix_masked = similarity_matrix * doc_words_mask_expanded  # [batch, seq_len, seq_len]
        similarity_max_pooled, _ = similarity_matrix_masked.max(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        # claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        claim_words_similarities_sum = torch.bmm(claim_frames_word_mask, similarity_max_pooled).squeeze(2)  # [batch, n_claim_frames]
        # mean_pool_mask [batch, n_claim_frames, hid]
        mean_pool_mask_shrinked = mean_pool_mask[:, :, 0]  # [batch, n_claim_frames]
        claim_words_precision = torch.zeros_like(claim_words_similarities_sum)  # [batch, n_claim_frames]
        # num_claim_frame_words [batch, n_claim_frames, 1]

        claim_words_precision[mean_pool_mask_shrinked] = claim_words_similarities_sum[mean_pool_mask_shrinked] / num_claim_frame_words.squeeze(-1)[
            mean_pool_mask_shrinked]  # [batch, n_claim_frames]
        """
        print("claim_words_precision")
        print(claim_words_precision.size())
        print(claim_words_precision[0])
        """
        claim_words_precision = claim_words_precision.unsqueeze(2)  # [batch, n_claim_frames, 1]

        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features, claim_words_precision], dim=2)  #  [batch, n_claim_frames, 2*hid_size+1]
        claim_frame_logits = self.classifier(self.dropout(frame_classification_features))  # [batch, n_claim_frames, num_classes]

        #cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]
        # expand cls representation
        #cls_representation_expanded = cls_representation.unsqueeze(1).expand(-1, claim_frame_features.size(1), -1)  # [batch, n_claim_frames, hid_size]

        #classifier_attn_weights = self.classifier_attn(self.dropout(torch.cat([cls_representation_expanded, frame_classification_features], dim=2)))  # [batch, n_claim_frames, num_classes]
        # claim_frames_padding_mask [batch, n_claim_frames]
        claim_frames_padding_mask_expanded = claim_frames_padding_mask.unsqueeze(2).expand(-1, -1, claim_frame_logits.size(2))
        # claim_frames_padding_mask_expanded [batch, n_claim_frames, num_classes]

        # first invert the 1 and 0, and then set the one values to a very negative number
        claim_frames_padding_mask_expanded_inverted = claim_frames_padding_mask_expanded < 0.5 * -100000.0

        # max pooling of the claim_frame_logits
        claim_frame_logits_max_pooled, _ = (claim_frame_logits + claim_frames_padding_mask_expanded_inverted).max(dim=1)  # [batch, num_classes]
        #claim_frame_logits_mean_pooled = claim_frame_logits_masked_sum / claim_frames_padding_mask_expanded.sum(dim=1)  # [batch, num_classes]
        logits = claim_frame_logits_max_pooled  # [batch, num_classes]

        #classifier_attn_weights = self.masked_softmax(classifier_attn_weights, mask=claim_frames_padding_mask_expanded)  # [batch, n_claim_frames, num_classes]
        #logits = (claim_frame_logits * classifier_attn_weights).sum(dim=1)  # [batch, num_classes]

        #logits = self.classifier(claim_attn_output_mean_pooled)
        # [batch, n_claim_frames, hid] ->  # [batch, hid]

        outputs = (logits, claim_frame_logits, claim_attn_output_weights) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnSepMilCosSimMultiLabelAdapterClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnSepMilCosSimMultiLabelAdapterClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2 + 1, self.num_labels)
        self.classifier_attn = nn.Linear(config.hidden_size * 3 + 1, self.num_labels)
        self.criterion = nn.BCEWithLogitsLoss()
        self.masked_softmax = MaskedSoftmax(dim=1)
        self.cos_sim = nn.CosineSimilarity(dim=2)

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        #self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        self.predicate_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        self.np_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        #self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMilCosSimMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, doc_frames_np_word_mask, doc_frames_predicate_word_mask, claim_frames_word_mask, claim_frames_np_word_mask, claim_frames_predicate_word_mask, claim_attn_mask,
                claim_frames_padding_mask, doc_words_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        """
        print("last_hidden_state")
        print(last_hidden_state[0, 0, :])
        print(last_hidden_state[0, 1, :])
        print(last_hidden_state[0, -1, :])
        """

        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        #last_hidden_state_projected = self.frame_feature_project(last_hidden_state)  # [batch_size, sequence_length, hidden_size]

        num_doc_frame_words = doc_frames_np_word_mask.sum(dim=-1, keepdims=True) + doc_frames_predicate_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        #doc_frame_features_sum = torch.bmm(doc_frames_word_mask, last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        # doc_frames_np_word_mask: [batch, n_doc_frames, seq_len]
        # doc_frames_predicate_word_mask: [batch, n_doc_frames, seq_len]
        doc_frame_np_features_sum = self.np_feature_project(torch.bmm(doc_frames_np_word_mask, last_hidden_state))  # [batch_size, n_doc_frames, hid]
        doc_frame_predicate_features_sum = self.predicate_feature_project(torch.bmm(doc_frames_predicate_word_mask, last_hidden_state))  # [batch_size, n_doc_frames, hid]
        doc_frame_features_sum = doc_frame_np_features_sum + doc_frame_predicate_features_sum  # [batch_size, n_doc_frames, hid]

        """
        print("doc_frames_predicate_word_mask")
        print(doc_frames_word_mask.size())
        print(doc_frames_word_mask[0, 0, :])
        print(doc_frames_predicate_word_mask.size())
        print(doc_frames_predicate_word_mask[0,0,:])
        print(doc_frames_np_word_mask.size())
        print(doc_frames_np_word_mask[0, 0, :])

        print("claim_frames_predicate_word_mask")
        print(claim_frames_word_mask.size())
        print(claim_frames_word_mask[0, 0, :])
        print(claim_frames_predicate_word_mask.size())
        print(claim_frames_predicate_word_mask[0, 0, :])
        print(claim_frames_np_word_mask.size())
        print(claim_frames_np_word_mask[0, 0, :])
        """

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
            mean_pool_mask]  # [batch_size, n_frames, hid]

        # claim_frames_np_word_mask: [batch, n_claim_frames, seq_len]
        # claim_frames_predicate_word_mask: [batch, n_claim_frames, seq_len]
        num_claim_frame_words = claim_frames_np_word_mask.sum(dim=-1, keepdims=True) + claim_frames_predicate_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_claim_frames, 1]
        #claim_frame_features_sum = torch.bmm(claim_frames_word_mask, last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]

        claim_frame_np_features_sum = self.np_feature_project(
            torch.bmm(claim_frames_np_word_mask, last_hidden_state))  # [batch_size, n_claim_frames, hid]
        claim_frame_predicate_features_sum = self.predicate_feature_project(
            torch.bmm(claim_frames_predicate_word_mask, last_hidden_state))  # [batch_size, n_claim_frames, hid]
        claim_frame_features_sum = claim_frame_np_features_sum + claim_frame_predicate_features_sum  # [batch_size, n_claim_frames, hid]

        num_claim_frame_words_expanded = num_claim_frame_words.expand(-1, -1,
                                                             claim_frame_features_sum.size(2))  # [batch, n_claim_frames, hid]
        mean_pool_mask = (num_claim_frame_words_expanded != 0)  # [batch, n_claim_frames, hid]
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words_expanded[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding

        # features of cosine similarity
        last_hidden_state_normalized = nn.functional.normalize(last_hidden_state, p=2, dim=2)
        similarity_matrix = torch.bmm(last_hidden_state_normalized,  torch.transpose(last_hidden_state_normalized, 1, 2))  # [batch, seq_len, seq_len]

        #similarity_matrix = self.cos_sim(last_hidden_state, last_hidden_state)  # [batch, seq_len, seq_len]
        # doc_words_mask [batch, seq_len]
        doc_words_mask_expanded = doc_words_mask.unsqueeze(1).expand(-1, similarity_matrix.size(1), -1)  # [batch, seq_len,seq_len]

        similarity_matrix_masked = similarity_matrix * doc_words_mask_expanded  # [batch, seq_len, seq_len]
        similarity_max_pooled, _ = similarity_matrix_masked.max(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        # claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        claim_words_similarities_sum = torch.bmm(claim_frames_word_mask, similarity_max_pooled).squeeze(2)  # [batch, n_claim_frames]
        # mean_pool_mask [batch, n_claim_frames, hid]
        mean_pool_mask_shrinked = mean_pool_mask[:, :, 0]  # [batch, n_claim_frames]
        claim_words_precision = torch.zeros_like(claim_words_similarities_sum)  # [batch, n_claim_frames]
        # num_claim_frame_words [batch, n_claim_frames, 1]

        claim_words_precision[mean_pool_mask_shrinked] = claim_words_similarities_sum[mean_pool_mask_shrinked] / num_claim_frame_words.squeeze(-1)[
            mean_pool_mask_shrinked]  # [batch, n_claim_frames]
        """
        print("claim_words_precision")
        print(claim_words_precision.size())
        print(claim_words_precision[0])
        """
        claim_words_precision = claim_words_precision.unsqueeze(2)  # [batch, n_claim_frames, 1]

        #claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        frame_classification_features = torch.cat([claim_attn_output, claim_frame_features, claim_words_precision], dim=2)  #  [batch, n_claim_frames, 2*hid_size+1]
        claim_frame_logits = self.classifier(self.dropout(frame_classification_features))  # [batch, n_claim_frames, num_classes]

        cls_representation = last_hidden_state[:, 0, :]  # [batch, hid_size]
        # expand cls representation
        cls_representation_expanded = cls_representation.unsqueeze(1).expand(-1, claim_frame_features.size(1), -1)  # [batch, n_claim_frames, hid_size]

        classifier_attn_weights = self.classifier_attn(self.dropout(torch.cat([cls_representation_expanded, frame_classification_features], dim=2)))  # [batch, n_claim_frames, num_classes]
        # claim_frames_padding_mask [batch, n_claim_frames]
        claim_frames_padding_mask_expanded = claim_frames_padding_mask.unsqueeze(2).expand(-1, -1, classifier_attn_weights.size(2))
        # claim_frames_padding_mask [batch, n_claim_frames, num_classes]
        classifier_attn_weights = self.masked_softmax(classifier_attn_weights, mask=claim_frames_padding_mask_expanded)  # [batch, n_claim_frames, num_classes]
        logits = (claim_frame_logits * classifier_attn_weights).sum(dim=1)  # [batch, num_classes]

        #logits = self.classifier(claim_attn_output_mean_pooled)
        # [batch, n_claim_frames, hid] ->  # [batch, hid]

        outputs = (logits, claim_frame_logits) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            #print("loss")
            #print(loss)
            #exit()
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMultiLabelAdapterClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMultiLabelAdapterClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertAdapterModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        #nn.init.xavier_uniform_(self.classifier.weight)
        self.criterion = nn.BCEWithLogitsLoss()

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cos_sim = nn.CosineSimilarity(dim=2)
        # self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMultiLabelAdapterClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        """
        print("last_hidden_state")
        print(last_hidden_state[0, 0, :])
        print(last_hidden_state[0, 1, :])
        print(last_hidden_state[0, -1, :])
        """

        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        last_hidden_state_projected = self.frame_feature_project(
            last_hidden_state)  # [batch_size, sequence_length, hidden_size]

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        """
        print("doc_frames_word_mask")
        print(doc_frames_word_mask[0,0,:])
        print(doc_frames_word_mask[0,1,:])
        print("non empty doc frames")
        print((num_doc_frame_words[0, :, 0] > 0).sum())
        print(num_doc_frame_words[0, : 0])
        """

        doc_frame_features_sum = torch.bmm(doc_frames_word_mask,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
            mean_pool_mask]  # [batch_size, n_frames, hid]

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)
        claim_frame_features_sum = torch.bmm(claim_frames_word_mask,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words = num_claim_frame_words.expand(-1, -1,
                                                             claim_frame_features_sum.size(
                                                                 2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_claim_frame_words != 0)
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             doc_frame_features_t,
                                                                             key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]


        claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        claim_attn_output_sum = torch.bmm(claim_frames_padding_mask.unsqueeze(1), claim_attn_output).squeeze(
            1)  # [batch, hid_size]

        num_claim_frames = claim_frames_padding_mask.sum(1, keepdims=True)  # [batch, 1]
        claim_attn_output_mean_pooled = claim_attn_output_sum / num_claim_frames

        # features of cosine similarity
        """
        # similarity_matrix = torch.bmm(last_hidden_state,  torch.transpose(last_hidden_state, 1, 2))
        similarity_matrix = self.cos_sim(last_hidden_state, last_hidden_state)
        # [batch, seq_len, seq_len]
        # claim_text_mask: [batch, n_claim_subwords, seq_len]
        claim_text_similarity = torch.bmm(claim_text_mask, similarity_matrix)
        # claim_text_similarity: [batch, n_claim_subwords, seq_len]
        # doc_text_mask: [batch, seq_len]
        doc_text_mask = doc_text_mask.unsqueeze(1).expand(-1, claim_text_mask.size(1), -1)
        # doc_text_mask: [batch, n_claim_subwords, seq_len]
        claim_text_similarity_masked = claim_text_similarity * doc_text_mask
        # claim_text_similarity_masked: [batch, n_claim_subwords, seq_len]
        claim_text_similarity_max_pooled = claim_text_similarity_masked.max(dim=2)  # [batch, n_claim_subwords]
        """

        logits = self.classifier(claim_attn_output_mean_pooled)
        # [batch, n_claim_frames, hid] ->  # [batch, hid]

        outputs = (logits, claim_attn_output_weights) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # exit()
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertClaimAttnMultiLabelClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertClaimAttnMultiLabelClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        #nn.init.xavier_uniform_(self.classifier.weight)
        self.criterion = nn.BCEWithLogitsLoss()

        self.num_attn_heads = config.n_claim_attn_heads
        self.claim_frame_attn = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_attn_heads)
        self.frame_feature_project = nn.Linear(config.hidden_size, config.hidden_size)
        self.claim_frame_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        #self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertClaimAttnMultiLabelClassifier")

    def set_loss_pos_weight(self, pos_weight):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("set positive weights to:")
        print(pos_weight.detach().cpu().numpy())

    def forward(self, input_ids, doc_frames_word_mask, claim_frames_word_mask, claim_attn_mask,
                claim_frames_padding_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        """
        print("last_hidden_state")
        print(last_hidden_state[0, 0, :])
        print(last_hidden_state[0, 1, :])
        print(last_hidden_state[0, -1, :])
        """

        # code for aggregating representation for each frame
        # doc_frames_word_mask: [batch, n_doc_frames, seq_len], claim_frames_word_mask: [batch, n_claim_frames, seq_len]
        # separate frame_mask into verb and args
        last_hidden_state_projected = self.frame_feature_project(
            last_hidden_state)  # [batch_size, sequence_length, hidden_size]

        num_doc_frame_words = doc_frames_word_mask.sum(dim=-1, keepdims=True)  # [batch, n_doc_frames, 1]

        """
        print("doc_frames_word_mask")
        print(doc_frames_word_mask[0,0,:])
        print(doc_frames_word_mask[0,1,:])
        print("non empty doc frames")
        print((num_doc_frame_words[0, :, 0] > 0).sum())
        print(num_doc_frame_words[0, : 0])
        """

        doc_frame_features_sum = torch.bmm(doc_frames_word_mask,
                                           last_hidden_state_projected)  # [batch_size, n_doc_frames, hid]

        num_doc_frame_words = num_doc_frame_words.expand(-1, -1,
                                                         doc_frame_features_sum.size(2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_doc_frame_words != 0)
        doc_frame_features = torch.zeros_like(doc_frame_features_sum)
        doc_frame_features[mean_pool_mask] = doc_frame_features_sum[mean_pool_mask] / num_doc_frame_words[
            mean_pool_mask]  # [batch_size, n_frames, hid]

        """
        print("doc_frame_features")
        print(doc_frame_features[0, 0, :])
        print(doc_frame_features[0, 1, :])
        print(doc_frame_features[0, 2, :])
        """

        num_claim_frame_words = claim_frames_word_mask.sum(dim=-1, keepdims=True)
        claim_frame_features_sum = torch.bmm(claim_frames_word_mask,
                                             last_hidden_state_projected)  # [batch_size, n_claim_frames, hid]
        num_claim_frame_words = num_claim_frame_words.expand(-1, -1,
                                                             claim_frame_features_sum.size(
                                                                 2))  # [batch, n_doc_frames, hid]
        mean_pool_mask = (num_claim_frame_words != 0)
        claim_frame_features = torch.ones_like(claim_frame_features_sum)
        claim_frame_features[mean_pool_mask] = claim_frame_features_sum[mean_pool_mask] / num_claim_frame_words[
            mean_pool_mask]  # [batch_size, n_claim_frames, hid]

        """
        print("claim_attn_mask")  # [batch_size, n_doc_frames]
        print(claim_attn_mask.size())
        print(claim_attn_mask[0, :])
        print("num_claim_frame_words")
        print(num_claim_frame_words[0, 0, 0])
        print(num_claim_frame_words[0, 1, 0])
        print(num_claim_frame_words[0, 2, 0])
        print("claim_frame_features")
        print(claim_frame_features[0, 0, :])
        print(claim_frame_features[0, 1, :])
        print(claim_frame_features[0, 2, :])
        print(claim_frame_features[0, -1, :])
        """

        claim_frame_features_t = torch.transpose(claim_frame_features, 0, 1)
        doc_frame_features_t = torch.transpose(doc_frame_features, 0, 1)
        # claim_attn_mask = claim_attn_mask.unsqueeze(1).repeat(1, self.num_attn_heads, 1, 1)
        # _, _, n_claim_frames, n_doc_frames = claim_attn_mask.size()
        # claim_attn_mask = claim_attn_mask.view(claim_attn_mask.size(0) * self.num_attn_heads, n_claim_frames, n_doc_frames)
        claim_attn_output, claim_attn_output_weights = self.claim_frame_attn(claim_frame_features_t,
                                                                            doc_frame_features_t,
                                                                            doc_frame_features_t,
                                                                            key_padding_mask=claim_attn_mask)
        claim_attn_output = torch.transpose(claim_attn_output, 0, 1)  # [batch, n_claim_frames, hid]

        # claim_frames_padding_mask [batch, n_claim_frames]
        # claim_attn_output [batch_size, n_claim_frames, hid]
        # mean pooling of the claim attn output for every claim frame while ignoring the padding
        """
        claim_attn_output_nan_mask = claim_frames_padding_mask.unsqueeze(2).expand(-1, -1, claim_attn_output.size(2))  # [batch_size, n_claim_frames, hid]
        claim_attn_output_nan_mask = (claim_attn_output_nan_mask != 0)
        claim_attn_output_no_nan = torch.zeros_like(claim_attn_output)
        claim_attn_output_no_nan[claim_attn_output_nan_mask] = claim_attn_output[claim_attn_output_nan_mask]
        claim_attn_output_no_nan = self.layer_norm(self.dropout(claim_attn_output_no_nan) + claim_frame_features)
        """

        """
        print("claim_attn_output_no_nan")
        print(claim_attn_output_no_nan[0, 0, :].detach().cpu().numpy())
        print(claim_attn_output_no_nan[0, 1, :].detach().cpu().numpy())
        print(claim_attn_output_no_nan[0, 2, :].detach().cpu().numpy())
        print(claim_attn_output_no_nan[0, 3, :].detach().cpu().numpy())
        print()
        print()
        """

        claim_attn_output = self.claim_frame_layer_norm(self.dropout(claim_attn_output) + claim_frame_features)
        claim_attn_output_sum = torch.bmm(claim_frames_padding_mask.unsqueeze(1), claim_attn_output).squeeze(
            1)  # [batch, hid_size]

        num_claim_frames = claim_frames_padding_mask.sum(1, keepdims=True)  # [batch, 1]
        claim_attn_output_mean_pooled = claim_attn_output_sum / num_claim_frames

        logits = self.classifier(claim_attn_output_mean_pooled)
        # [batch, n_claim_frames, hid] ->  # [batch, hid]

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # print("logits")
            # print(logits.detach().cpu().numpy())
            # print("labels")
            # print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            # exit()
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class MaskedSoftmax(nn.Module):
    def __init__(self, dim):
        super(MaskedSoftmax, self).__init__()
        self.dim = dim

    def forward(self, logit, mask=None):
        if mask is None:
            dist = nn.functional.softmax(logit - torch.max(logit, dim=self.dim, keepdim=True)[0], dim=self.dim)
        else:
            dist_ = nn.functional.softmax(logit - torch.max(logit, dim=self.dim, keepdim=True)[0], dim=self.dim) * mask
            normalization_factor = dist_.sum(self.dim, keepdim=True)
            dist = dist_ / normalization_factor
        return dist


class BertSRLAttnFCMultiLabelClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSRLAttnFCMultiLabelClassifier, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        #nn.init.xavier_uniform_(self.classifier.weight)
        self.criterion = nn.BCEWithLogitsLoss()

        self.num_verb_attn_heads = 6
        self.verb_attention = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_verb_attn_heads)
        self.cls_attention = nn.MultiheadAttention(config.hidden_size, num_heads=self.num_verb_attn_heads)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        # self.apply(self.init_weights)
        # self.init_weights()
        self.post_init()
        print("BertSRLAttnFCMultiLabelClassifier")

    def forward(self, input_ids, verb_attn_mask, cls_attn_mask, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        # run through bert
        #print("input_ids")
        #print(input_ids.size())
        #print("token_type_ids")
        #print(token_type_ids.size())
        #print(token_type_ids[0].detach().cpu().numpy())
        # verb_attn_mask [batch, seq_len, seq_len]
        # cls_attn_mask [batch, seq_len]
        #print("verb_attn_mask")
        #print(verb_attn_mask.size())
        #print(verb_attn_mask[0,0].detach().cpu().numpy())
        #print("cls_attn_mask")
        #print(cls_attn_mask.size())
        #print(cls_attn_mask[0].detach().cpu().numpy())
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, head_mask=head_mask)
        cls_output = outputs[1]  # [batch, hidden_size]
        #print("cls_output")
        #print(cls_output.size())

        # select verb representation
        last_hidden_state = outputs[0]  # [batch_size, sequence_length, hidden_size]
        last_hidden_state = torch.transpose(last_hidden_state, 0, 1)
        #batch_size, seq_len, _ = verb_attn_mask.size()
        #verb_attn_mask = verb_attn_mask.expand(batch_size * self.num_verb_attn_heads, -1, -1)
        verb_attn_mask = verb_attn_mask.repeat(self.num_verb_attn_heads, 1, 1)
        #verb_attn_mask = torch.transpose(verb_attn_mask, 0, 1)
        # batch first?
        #print("last_hidden_state")
        #print(last_hidden_state.size())
        #print("verb_attn_mask")
        #print(verb_attn_mask.size())
        verb_attn_output, verb_attn_output_weights = self.verb_attention(last_hidden_state, last_hidden_state,
                                                                         last_hidden_state, attn_mask=verb_attn_mask)
        verb_attn_output = self.layer_norm(self.dropout(verb_attn_output) + last_hidden_state)
        # verb_attn_output: [seq_len, batch_size, hidden_size]

        verb_attn_output_fc = self.layer_norm(self.dropout(self.fc(verb_attn_output)) + verb_attn_output)

        cls_output_ = cls_output.unsqueeze(0)  # [1, batch, hidden_size]

        # concat cls representation with verb attn output
        verb_attn_output_ = torch.cat([cls_output_, verb_attn_output_fc[1:, :, :]], dim=0)

        cls_attn_output, cls_attn_output_weights = self.cls_attention(cls_output_, verb_attn_output_, verb_attn_output_, key_padding_mask=cls_attn_mask)
        # cls_attn_output  [1, batch, hidden_size]
        #print("cls_attn_output")
        #print(cls_attn_output.size())
        #print(cls_attn_output[0,0].detach().cpu().numpy())
        #exit()
        cls_attn_output = self.layer_norm(self.dropout(cls_attn_output.squeeze(0))) + self.dropout(cls_output)
        #print("cls_attn_output")
        #print(cls_attn_output.size())
        #print(cls_attn_output[0].detach().cpu().numpy())
        logits = self.classifier(cls_attn_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            #print("logits")
            #print(logits.detach().cpu().numpy())
            #print("labels")
            #print(labels.detach().cpu().numpy())
            loss = self.criterion(logits, labels)
            #print("loss")
            #print(loss.detach().cpu().numpy())
            #exit()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertPointer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertPointer, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # classifiers
        self.ext_start_classifier = nn.Linear(config.hidden_size, 1, bias=False)
        self.ext_end_classifier = nn.Linear(config.hidden_size, 1, bias=False)
        self.aug_start_classifier = nn.Linear(config.hidden_size, 1, bias=False)
        self.aug_end_classifier = nn.Linear(config.hidden_size, 1, bias=False)

        self.label_classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, 
                ext_mask=None, ext_start_labels=None, ext_end_labels=None,
                aug_mask=None, aug_start_labels=None, aug_end_labels=None,
                loss_lambda=1.0):
        # run through bert
        bert_outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, head_mask=head_mask)

        # label classifier
        pooled_output = bert_outputs[1]
        pooled_output = self.dropout(pooled_output)
        label_logits = self.label_classifier(pooled_output)

        # extraction classifier
        output = bert_outputs[0]
        ext_mask = ext_mask.unsqueeze(-1)
        ext_start_logits = self.ext_start_classifier(output) * ext_mask
        ext_end_logits = self.ext_end_classifier(output) * ext_mask

        # augmentation classifier
        output = bert_outputs[0]
        aug_mask = aug_mask.unsqueeze(-1)
        aug_start_logits = self.aug_start_classifier(output) * aug_mask
        aug_end_logits = self.aug_end_classifier(output) * aug_mask

        span_logits = (ext_start_logits, ext_end_logits, aug_start_logits, aug_end_logits,)
        outputs = (label_logits,) + span_logits + bert_outputs[2:]

        if labels is not None and \
                ext_start_labels is not None and ext_end_labels is not None and \
                aug_start_labels is not None and aug_end_labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(label_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()

                # label loss
                labels_loss = loss_fct(label_logits.view(-1, self.num_labels), labels.view(-1))

                # extraction loss
                ext_start_loss = loss_fct(ext_start_logits.squeeze(), ext_start_labels)
                ext_end_loss = loss_fct(ext_end_logits.squeeze(), ext_end_labels)

                # augmentation loss
                aug_start_loss = loss_fct(aug_start_logits.squeeze(), aug_start_labels)
                aug_end_loss = loss_fct(aug_end_logits.squeeze(), aug_end_labels)

                span_loss = (ext_start_loss + ext_end_loss + aug_start_loss + aug_end_loss) / 4

                # combined loss
                loss = labels_loss + loss_lambda * span_loss

            outputs = (loss, labels_loss, span_loss, ext_start_loss, ext_end_loss, aug_start_loss, aug_end_loss) + outputs

        return outputs  # (loss), (logits), (hidden_states), (attentions)

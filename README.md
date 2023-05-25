# Interpretable Automatic Fine-grained Inconsistency Detection in Text Summarization

This repository contains the source code for our ACL Findings 2023 paper: [Interpretable Automatic Fine-grained Inconsistency Detection in Text Summarization](https://arxiv.org/pdf/2305.14548).

If you use our source code, please cite our paper
```
@inproceedings{finegrainfact,
    title={Interpretable Automatic Fine-grained Inconsistency Detection in Text Summarization},
    author={Chan, Hou Pong and Zeng, Qi and Ji, Heng},
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = {July},
    year = "2023",
    publisher = "Association for Computational Linguistics",
    }
```

## Aggrefact-United Dataset

We conduct experiments on the Aggrefact-United dataset. If you use this dataset, please cite their [paper](https://arxiv.org/pdf/2205.12854v1.pdf).

The original dataset contains 5,496 samples. We remove the duplicated annotations and obtain 4,489 samples. 
Then we randomly split data samples into train/validation/test sets of size 3,689/300/500. 
After that, we use the SRL tool from Allennlp to parse the document and summary. 
This repository contains our preprocessed data splits. 
The training and validation sets are in `data/aggrefact-deduplicated-final`. The test set is in `data/aggrefact-deduplicated-final`.

## Environment setup
```
conda create -n finegrainfact python=3.7.13
pip3 install -r requirements.txt
```

## Training
```
# please change the CODE_PATH, DATA_PATH, OUTPUT_PATH variables in the below script file before running it.
bash modeling/scripts/aggrefact-train-finegrainfact-model.sh 2>&1 | tee ./logs/aggrefact-train-finegrainfact-model.log
```
After the training process is completed, you can find the path to the best checkpoint by searching `Best bacc chkpt path:` in the log file `./logs/aggrefact-train-finegrainfact-model.log`.

## Inference
Run the following script. 
```
# please change the CODE_PATH, DATA_PATH, CKPT_PATH variables in the below script file before running it.
bash modeling/scripts/aggrefact-finetune-finegrainfact-model.sh
```

## Evaluation of Document Fact Highlights
Our preprocessed Fever 2.0 dataset is in `./data/fever2`.
Run the following script. 
```
# please change the CODE_PATH, DATA_PATH, CKPT_PATH variables in the below script file before running it.
bash modeling/scripts/fever2-inference-finegrainfact-model.sh
```

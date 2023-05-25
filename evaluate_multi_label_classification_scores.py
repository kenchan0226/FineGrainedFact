import numpy as np
import json
import jsonlines
import argparse
from modeling.utils import multi_label_acc_and_f1, classification_metric_all, classification_metric_all_macro, multi_label_acc_auc_f1, multi_label_bacc_auc_f1, multi_label_bacc_auc_f1_macro
from sklearn.metrics import classification_report


SOTA = ['BART', 'PegasusDynamic', 'T5', 'Pegasus']
XFORMER = ['BertSum', 'BertExtAbs', 'BertExt', 'GPT2', 'BERTS2S', 'TranS2S']
OLD = ['FastAbsRl','TConvS2S', 'PtGen', 'PtGenCoverage',
            'Summa', 'BottomUp', 'Seq2Seq', 'TextRank', 'missing', 'ImproveAbs', 'NEUSUM',
            'ClosedBookDecoder', 'RNES', 'BanditSum', 'ROUGESal', 'MultiTask', 'UnifiedExtAbs']


def main(model_output_file, src_jsonl_file, is_entire):
    score_output_file = model_output_file.replace(".jsonl", "-scores.csv")
    src_sample_list = []
    with jsonlines.open(src_jsonl_file) as f:
        for line in f:
            src_sample_list.append(line)
    #output_sample_list = json.load(open(model_output_file))
    output_sample_list = []
    with jsonlines.open(model_output_file) as f:
        for line in f:
            output_sample_list.append(line)

    assert len(src_sample_list) == len(output_sample_list)

    # "pred_scores": [0.12050816416740417, 0.15728257596492767, 0.40704405307769775, 0.14202255010604858, 0.23076584935188293]
    # ["incorrect", "extrinsic-NP", "intrinsic-NP", "extrinsic-predicate", "intrinsic-predicate"]
    error_name_list = ["extrinsic-NP", "intrinsic-NP", "extrinsic-predicate", "intrinsic-predicate"]

    #print("****** threshold: ******")
    #print(all_types_threshold)

    preds = []
    preds_scores = []
    gold_label_ids = []

    pred_error_class_label_all_dict = {"all": [], "xsum": [], "cnndm": [], "old": [], "xformer": [], "sota": [], "reference": []}
    pred_error_class_score_all_dict = {"all": [], "xsum": [], "cnndm": [], "old": [], "xformer": [],
                                                "sota": [], "reference": []}
    gold_error_class_label_all_dict = {"all": [], "xsum": [], "cnndm": [], "old": [], "xformer": [], "sota": [], "reference": []}

    for sample_i, (src_sample, output_sample) in enumerate(zip(src_sample_list, output_sample_list)):
        pred_score = np.asarray(output_sample["pred_scores"])
        pred = pred_score >= 0.5
        gold = [False, False, False, False]

        for error in src_sample["error_type"]:
            if error == "extrinsic-NP":
                gold[0] = True
            elif error == "intrinsic-NP":
                gold[1] = True
            elif error == "extrinsic-predicate":
                gold[2] = True
            elif error == "intrinsic-predicate":
                gold[3] = True
            elif error == "extrinsic-entire_sent" and is_entire:
                gold[0] = True
                gold[2] = True
            elif error == "intrinsic-entire_sent" and is_entire:
                gold[1] = True
                gold[3] = True
            elif error == "entire_sent" and is_entire:
                gold[0] = True
                gold[1] = True
                gold[2] = True
                gold[3] = True
        gold = np.asarray(gold)

        #preds.append(pred)
        #preds_scores.append(pred_score)
        #gold_label_ids.append(gold)

        origin = src_sample["origin"]
        model_name = src_sample["model_name"]

        # model type
        if model_name in OLD:
            model_category = "old"
        elif model_name in XFORMER:
            model_category = "xformer"
        elif model_name in SOTA:
            model_category = "sota"
        elif model_name == "Gold":
            model_category = "reference"
        else:
            raise ValueError
        #else:
        #    model_category = None

        key_list = ["all", origin]
        if model_category:
            key_list.append(model_category)

        for key in key_list:
            pred_error_class_label_all_dict[key].append(pred)
            pred_error_class_score_all_dict[key].append(pred_score)
            gold_error_class_label_all_dict[key].append(gold)

    scores_dict = {}
    for k in pred_error_class_label_all_dict.keys():
        pred_error_class_label_all_dict[k] = np.vstack(pred_error_class_label_all_dict[k])
        pred_error_class_score_all_dict[k] = np.vstack(pred_error_class_score_all_dict[k])
        gold_error_class_label_all_dict[k] = np.vstack(gold_error_class_label_all_dict[k])

        scores_dict[k] = multi_label_bacc_auc_f1_macro(preds=pred_error_class_label_all_dict[k],
                                                preds_scores=pred_error_class_score_all_dict[k],
                                                labels=gold_error_class_label_all_dict[k])
        #scores_dict[k] = multi_label_acc_auc_f1(preds=pred_error_class_label_all_dict[k],
        #                       preds_scores=pred_error_class_score_all_dict[k],
        #                       labels=gold_error_class_label_all_dict[k])


    for i, error_type_name in enumerate(error_name_list):
        pred_error_class_label = pred_error_class_label_all_dict["all"][:, i]
        pred_error_class_score = pred_error_class_score_all_dict["all"][:, i]
        gold_error_class_label = gold_error_class_label_all_dict["all"][:, i]
        scores_dict[error_type_name] = classification_metric_all_macro(preds=pred_error_class_label, preds_scores=pred_error_class_score, labels=gold_error_class_label)
        #print(error_type_name + ":")
        #print(classification_metric_all(preds=pred_error_class_label, labels=gold_error_class_label))

    heading_list = ["SOTA F1", "SOTA BACC", "SOTA AUC", "XFORMER F1", "XFORMER BACC", "XFORMER AUC", "OLD F1", "OLD BACC", "OLD AUC", "REF_F1", "REF_BACC", "REF_AUC", "ALL_F1", "ALL_BACC", "ALL_AUC"]
    result_list = [scores_dict["sota"]["f1"], scores_dict["sota"]["bacc"], scores_dict["sota"]["auc"], scores_dict["xformer"]["f1"], scores_dict["xformer"]["bacc"], scores_dict["xformer"]["auc"],
                   scores_dict["old"]["f1"], scores_dict["old"]["bacc"], scores_dict["old"]["auc"], scores_dict["reference"]["f1"], scores_dict["reference"]["bacc"], scores_dict["reference"]["auc"],
                   scores_dict["all"]["f1"], scores_dict["all"]["bacc"], scores_dict["all"]["auc"]]

    #print("******** Results ********")
    #print("\t".join(heading_list))
    result_list = ["{:.5f}".format(result) for result in result_list]
    #print("\t".join(result_list))

    with open(score_output_file, "w") as f_out:
        f_out.write( ",".join(heading_list) + "\n" )  # write heading
        f_out.write( ",".join(result_list) )  # write scores
    print("Scores written to {}".format(score_output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_file", default=None, type=str,
                        help="")
    parser.add_argument("--src_jsonl_file", default=None, type=str,
                        help="")
    parser.add_argument("--is_entire", action="store_true",
                        help="")
    args = parser.parse_args()
    #filename = "/shared/nas/data/users/hpchan/projects/PolNeAR/software/dev.json"
    main(args.model_output_file, args.src_jsonl_file, args.is_entire)

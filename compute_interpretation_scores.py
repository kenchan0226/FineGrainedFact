import numpy as np
#import json
import jsonlines
import argparse
#from modeling.utils import multi_label_acc_and_f1, classification_metric_all, classification_metric_all_macro, multi_label_acc_auc_f1, multi_label_bacc_auc_f1, multi_label_bacc_auc_f1_macro
#from sklearn.metrics import classification_report


def compute_f1_score(recall, precision):
    if recall == 0 and precision == 0:
        return 0.0
    else:
        return 2 * recall * precision / (recall + precision)


def main(model_output_file, src_jsonl_file):
    #score_output_file = model_output_file.replace(".jsonl", "-scores.csv")
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

    sample_cnt = 0
    recall_at_1_total = 0
    recall_at_2_total = 0
    recall_at_3_total = 0
    recall_at_4_total = 0
    recall_at_5_total = 0

    precision_at_1_total = 0
    precision_at_2_total = 0
    precision_at_3_total = 0
    precision_at_4_total = 0
    precision_at_5_total = 0

    num_evidence_doc_frames_total = 0

    skip_cnt = 0

    for sample_i, (src_sample, output_sample) in enumerate(zip(src_sample_list, output_sample_list)):
        claim_frames_word_ids = src_sample["claim_frames_word_ids"]
        summary_word_list = src_sample["summ"].split(" ")
        claim_frame_list = []
        for claim_frame in claim_frames_word_ids:
            all_word_ids = claim_frame["V"] + claim_frame["NP"]
            all_word_ids.sort()
            claim_frame_words = [summary_word_list[word_id] for word_id in all_word_ids]
            claim_frame_list.append(" ".join(claim_frame_words))
        num_claim_frames = len(claim_frames_word_ids)
        doc_frames_word_ids = src_sample["doc_frames_word_ids"]
        #print("Doc frames")
        #doc_word_list = src_sample["doc"].split(" ")
        doc_frame_word_id_list = []
        for doc_frame in doc_frames_word_ids:
            all_word_ids = doc_frame["V"] + doc_frame["NP"]
            all_word_ids.sort()
            doc_frame_word_id_list.append(all_word_ids)

        num_doc_frames = len(doc_frames_word_ids)
        if "claim_attn_output_weights" in output_sample.keys():  # for our model
            claim_attn_output_weights = output_sample["claim_attn_output_weights"]  # [tgt_len, src_len]
            claim_attn_output_weights = np.asarray(claim_attn_output_weights)
            claim_attn_output_weights = claim_attn_output_weights[:num_claim_frames, :num_doc_frames]
            # claim_attn_output_weights [num_claim_frames, num_doc_frames]
            doc_frame_total_attn_weights = np.sum(claim_attn_output_weights, axis=0)  # [num_doc_frames]
        else:  # for baseline models
            doc_frame_total_attn_weights = np.asarray(output_sample["doc_frame_total_attn_weights"])
        doc_frame_ids_sort_by_scores = doc_frame_total_attn_weights.argsort()[::-1]

        doc_frame_word_list_sorted_by_scores = []
        for doc_frame_id in doc_frame_ids_sort_by_scores:
            doc_frame_word_list_sorted_by_scores.append(doc_frame_word_id_list[doc_frame_id])

        doc_frame_word_list_sorted_by_scores_at_1_all = doc_frame_word_list_sorted_by_scores[0]
        doc_frame_word_list_sorted_by_scores_at_2_all = []
        for word_id_list in doc_frame_word_list_sorted_by_scores[:2]:
            doc_frame_word_list_sorted_by_scores_at_2_all += word_id_list
        doc_frame_word_list_sorted_by_scores_at_3_all = []
        for word_id_list in doc_frame_word_list_sorted_by_scores[:3]:
            doc_frame_word_list_sorted_by_scores_at_3_all += word_id_list
        doc_frame_word_list_sorted_by_scores_at_5_all = []
        for word_id_list in doc_frame_word_list_sorted_by_scores[:5]:
            doc_frame_word_list_sorted_by_scores_at_5_all += word_id_list

        evidence_sentences_word_ids = src_sample["evidence_sentences_word_ids"]

        ground_truth_word_ids = []
        for start_word_id, end_word_id in evidence_sentences_word_ids:
            ground_truth_word_ids += list(range(start_word_id,end_word_id))

        evidence_doc_frames_ids = src_sample["evidence_doc_frames_ids"]

        if len(evidence_doc_frames_ids) == 0:
            skip_cnt += 1
            continue

        # recall, p, f1
        match_at_1 = 0
        match_at_2 = 0
        match_at_3 = 0
        match_at_4 = 0
        match_at_5 = 0

        for evidence_doc_frames_id in evidence_doc_frames_ids:
            if evidence_doc_frames_id == doc_frame_ids_sort_by_scores[0]:
                match_at_1 += 1
            if evidence_doc_frames_id in doc_frame_ids_sort_by_scores[:2]:
                match_at_2 += 1
            if evidence_doc_frames_id in doc_frame_ids_sort_by_scores[:3]:
                match_at_3 += 1
            if evidence_doc_frames_id in doc_frame_ids_sort_by_scores[:4]:
                match_at_4 += 1
            if evidence_doc_frames_id in doc_frame_ids_sort_by_scores[:5]:
                match_at_5 += 1

        num_evidence_doc_frames_total += len(evidence_doc_frames_ids)

        recall_at_1 = match_at_1 / len(evidence_doc_frames_ids)
        recall_at_2 = match_at_2 / len(evidence_doc_frames_ids)
        recall_at_3 = match_at_3 / len(evidence_doc_frames_ids)
        recall_at_4 = match_at_4 / len(evidence_doc_frames_ids)
        recall_at_5 = match_at_5 / len(evidence_doc_frames_ids)
        recall_at_1_total += recall_at_1
        recall_at_2_total += recall_at_2
        recall_at_3_total += recall_at_3
        recall_at_4_total += recall_at_4
        recall_at_5_total += recall_at_5

        precision_at_1 = match_at_1 / 1
        precision_at_2 = match_at_2 / 2
        precision_at_3 = match_at_3 / 3
        precision_at_4 = match_at_4 / 4
        precision_at_5 = match_at_5 / 5
        precision_at_1_total += precision_at_1
        precision_at_2_total += precision_at_2
        precision_at_3_total += precision_at_3
        precision_at_4_total += precision_at_4
        precision_at_5_total += precision_at_5

        sample_cnt += 1

    recall_at_1_macro_avg = recall_at_1_total/sample_cnt
    recall_at_2_macro_avg = recall_at_2_total/sample_cnt
    recall_at_3_macro_avg = recall_at_3_total/sample_cnt
    recall_at_4_macro_avg = recall_at_4_total/sample_cnt
    recall_at_5_macro_avg = recall_at_5_total/sample_cnt

    precision_at_1_macro_avg = precision_at_1_total / sample_cnt
    precision_at_2_macro_avg = precision_at_2_total / sample_cnt
    precision_at_3_macro_avg = precision_at_3_total / sample_cnt
    precision_at_4_macro_avg = precision_at_4_total / sample_cnt
    precision_at_5_macro_avg = precision_at_5_total / sample_cnt

    f1_at_1_macro_avg = compute_f1_score(recall_at_1_macro_avg, precision_at_1_macro_avg)
    f1_at_2_macro_avg = compute_f1_score(recall_at_2_macro_avg, precision_at_2_macro_avg)
    f1_at_3_macro_avg = compute_f1_score(recall_at_3_macro_avg, precision_at_3_macro_avg)
    f1_at_4_macro_avg = compute_f1_score(recall_at_4_macro_avg, precision_at_4_macro_avg)
    f1_at_5_macro_avg = compute_f1_score(recall_at_5_macro_avg, precision_at_5_macro_avg)

    print("recall_at_1", recall_at_1_macro_avg)
    print("recall_at_2", recall_at_2_macro_avg)
    print("recall_at_3", recall_at_3_macro_avg)
    print("recall_at_4", recall_at_4_macro_avg)
    print("recall_at_5", recall_at_5_macro_avg)

    print("precision_at_1", precision_at_1_macro_avg)
    print("precision_at_2", precision_at_2_macro_avg)
    print("precision_at_3", precision_at_3_macro_avg)
    print("precision_at_4", precision_at_4_macro_avg)
    print("precision_at_3", precision_at_5_macro_avg)

    print("f1_at_1_macro_avg", f1_at_1_macro_avg)
    print("f1_at_2_macro_avg", f1_at_2_macro_avg)
    print("f1_at_3_macro_avg", f1_at_3_macro_avg)
    print("f1_at_4_macro_avg", f1_at_4_macro_avg)
    print("f1_at_5_macro_avg", f1_at_5_macro_avg)

    print("skip_cnt", skip_cnt)
    print("sample cnt", sample_cnt)
    print("avg no. of evidence doc frames: ", num_evidence_doc_frames_total/sample_cnt)

    heading_list = ["R@1","R@2","R@3","R@4","R@5","P@1","P@2","P@3","P@4","P@5","F1@1","F1@2","F1@3","F1@4","F1@5"]
    result_list = [recall_at_1_macro_avg, recall_at_2_macro_avg, recall_at_3_macro_avg, recall_at_4_macro_avg, recall_at_5_macro_avg,
                   precision_at_1_macro_avg, precision_at_2_macro_avg, precision_at_3_macro_avg, precision_at_4_macro_avg, precision_at_5_macro_avg,
                   f1_at_1_macro_avg, f1_at_2_macro_avg, f1_at_3_macro_avg, f1_at_4_macro_avg, f1_at_5_macro_avg]
    result_list = ["{:.5f}".format(r) for r in result_list]
    score_output_file = model_output_file.replace(".jsonl", "-scores.csv")
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
    args = parser.parse_args()
    #filename = "/shared/nas/data/users/hpchan/projects/PolNeAR/software/dev.json"
    main(args.model_output_file, args.src_jsonl_file)

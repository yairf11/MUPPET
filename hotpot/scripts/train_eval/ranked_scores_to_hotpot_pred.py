import argparse
from ast import literal_eval
import json
import pandas as pd
import numpy as np


def df_to_pred(df, filename, return_results=False, return_titles=False):
    span_q_col = "span_question_scores"
    yes_no_q_col = "yes_no_question_scores"
    yes_no_scores_col = "yes_no_confidence_scores"
    span_scores_col = "predicted_score"
    group_cols = ['question_id']

    answer_dict = {}
    sp_dict = {}
    titles_dict = {}

    for gid, group in df[[span_scores_col, span_q_col, yes_no_q_col, yes_no_scores_col,
                                 'text_answer', 'predicted_sp', 'titles'] +
                                group_cols].groupby(group_cols):
        max_span_question_score = group[span_q_col].expanding().apply(lambda x: x.max(), raw=True).values
        max_yes_no_question_score = group[yes_no_q_col].expanding().apply(lambda x: x.max(), raw=True).values
        is_yes_no_expanding = np.stack([max_span_question_score, max_yes_no_question_score], axis=0).argmax(axis=0)
        span_used_predictions = group[span_scores_col].expanding().apply(lambda x: x.argmax(), raw=True)
        yes_no_used_predictions = group[yes_no_scores_col].expanding().apply(lambda x: x.argmax(), raw=True)
        pred_idx = int(yes_no_used_predictions.values[-1]) if is_yes_no_expanding[-1] else int(
            span_used_predictions.values[-1])
        sp_dict[gid] = group['predicted_sp'].values[pred_idx]
        if type(sp_dict[gid]) == str:
            sp_dict[gid] = literal_eval(sp_dict[gid])
        if return_results and return_titles:
            titles_dict[gid] = group['titles'].values[pred_idx]
            if type(titles_dict[gid]) == str:
                titles_dict[gid] = literal_eval(titles_dict[gid])
        answer_dict[gid] = group['text_answer'].values[pred_idx]

    if return_results and return_titles:
        return answer_dict, sp_dict, titles_dict
    elif return_results:
        return answer_dict, sp_dict

    with open(filename, 'w') as f:
        json.dump({'answer': answer_dict, 'sp': sp_dict}, f)


def main():
    parser = argparse.ArgumentParser(description="produces a prediction file from a ranked scores file")
    parser.add_argument('answers', help='answer file')
    parser.add_argument('out_file', help="file to write the predictions to")
    args = parser.parse_args()

    print("Loading answers..")
    answer_df = pd.read_csv(args.answers)
    df_to_pred(answer_df, args.out_file)


if __name__ == "__main__":
    main()

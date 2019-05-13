import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd

from hotpot.utils import print_table


def compute_ranked_scores(df, max_over, target_score, group_cols):
    scores = []
    for _, group in df[[max_over, target_score] + group_cols].groupby(group_cols):
        if target_score == max_over:
            scores.append(group[target_score].cummax().values)
        else:
            used_predictions = group[max_over].expanding().apply(lambda x: x.argmax())
            scores.append(group[target_score].iloc[used_predictions].values)

    max_para = max(len(x) for x in scores)
    summed_scores = np.zeros(max_para)
    for s in scores:
        summed_scores[:len(s)] += s
        summed_scores[len(s):] += s[-1]
    return summed_scores / len(scores)


def compute_ranked_scores_with_yes_no(df, span_q_col, yes_no_q_col, yes_no_scores_col,
                                      span_scores_col, span_target_score, group_cols):
    """ Computes ranked scores with yes/no option.
    Note that there is a hidden assumption that the choice of whether to produce a yes/no answer or a span answer is
    based solely on the question, thus all paragraphs for a given question decide unanimously on the answer type"""

    scores = []

    for _, group in df[[span_scores_col, span_target_score, span_q_col, yes_no_q_col, yes_no_scores_col] +
                       group_cols].groupby(group_cols):
        max_span_question_score = group[span_q_col].expanding().apply(lambda x: x.max(), raw=True).values
        max_yes_no_question_score = group[yes_no_q_col].expanding().apply(lambda x: x.max(), raw=True).values
        is_yes_no_expanding = np.stack([max_span_question_score, max_yes_no_question_score], axis=0).argmax(axis=0)
        span_used_predictions = group[span_scores_col].expanding().apply(lambda x: x.argmax(), raw=True)
        yes_no_used_predictions = group[yes_no_scores_col].expanding().apply(lambda x: x.argmax(), raw=True)
        final_scores = group[span_target_score].iloc[span_used_predictions].values * (1 - is_yes_no_expanding) + \
                       group[span_target_score].iloc[yes_no_used_predictions].values * is_yes_no_expanding
        scores.append(final_scores)

    max_para = max(len(x) for x in scores)
    summed_scores = np.zeros(max_para)
    for s in scores:
        summed_scores[:len(s)] += s
        summed_scores[len(s):] += s[-1]
    return summed_scores / len(scores)


def show_scores_table(df, cols):
    rows = [["Rank"] + cols]
    for i in range(len(df)):
        rows.append(["%d" % (i + 1)] + ["%.4f" % df[k].iloc[i] for k in cols])
    print_table(rows)


def main():
    parser = argparse.ArgumentParser(description=
                                     "Compute scores as more paragraphs are used, using "
                                     "a per-paragraph csv file as built from our evaluation scripts ")
    parser.add_argument('answers', help='answer file(s)', nargs="+")
    parser.add_argument('--hotpot', action='store_true')
    parser.add_argument('--sp', action='store_true', help="show supporting facts")
    parser.add_argument('--br-cp', action='store_true', help='show bridge and comparison results')
    args = parser.parse_args()

    print("Loading answers..")
    answer_dfs = []
    for filename in args.answers:
        answer_dfs.append(pd.read_csv(filename))

    print("Computing ranks...")
    group_by = ["question_id"]

    def get_ranked_scores(df, score_name):
        filtered_df = df[df.type == 'comparison'] if "Cp" in score_name else \
            df[df.type == 'bridge'] if "Br" in score_name else df
        target_prefix = 'joint' if 'joint' in score_name else 'sp' if 'sp' in score_name else 'text'
        target_score = f"{target_prefix}_{'em' if 'EM' in score_name else 'f1'}"
        return compute_ranked_scores_with_yes_no(filtered_df, span_q_col="span_question_scores",
                                                 yes_no_q_col="yes_no_question_scores",
                                                 yes_no_scores_col="yes_no_confidence_scores",
                                                 span_scores_col="predicted_score", span_target_score=target_score,
                                                 group_cols=group_by)

    data = OrderedDict()
    for i, answer_df in enumerate(answer_dfs):
        answer_df.sort_values(["question_id", "rank"], inplace=True)
        if not args.hotpot:
            model_scores = compute_ranked_scores(answer_df, "predicted_score", "text_em", group_by)
            data["answers_%d_em" % i] = model_scores
            model_scores = compute_ranked_scores(answer_df, "predicted_score", "text_f1", group_by)
            data["answers_%d_f1" % i] = model_scores
        else:
            pref = f"df_{i}_"
            score_names = ["EM", "F1"]
            if args.br_cp:
                score_names.extend(["Br EM", "Br F1", "Cp EM", "Cp F1"])
            if args.sp:
                score_names.extend([f"{prefix} {name}" for prefix in ['sp', 'joint'] for name in score_names])
            for score in score_names:
                data[pref + score] = get_ranked_scores(answer_df, score)

    show_scores_table(pd.DataFrame(data),
                      sorted(data.keys(), key=lambda x: (0, x) if x.lower().endswith("em") else (1, x)))


if __name__ == "__main__":
    main()

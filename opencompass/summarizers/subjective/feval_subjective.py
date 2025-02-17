# flake8: noqa: E501
import csv
import os, json
import os.path as osp
import statistics
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
from mmengine import ConfigDict

try:
    from prettytable import from_csv
except ImportError:
    from_csv = None

from opencompass.utils import model_abbr_from_cfg

from .subjective_post_process import post_process_autoj, post_process_judgelm
from .utils import get_judgeanswer_and_reference, get_outdir

# CATEGORIES = {
#     "业务专精能力":["业务专精能力"],
#     "中文通用基础能力":["中文通用基础能力"],
#     "金融能力":["金融能力"]
# }

# All_Dimensions = [
#     '事实正确性', '满足用户需求', '安全无害', '清晰度', '逻辑性', '完备性', '创造性', '可负责程度', '逻辑连贯性',
#     '公平与可负责程度', '丰富度', '综合得分'
# ]

# MAPPING = {
#     '事实与解释型回答': ['事实正确性', '满足用户需求', '清晰度', '完备性'],
#     '逻辑推理型回答': ['事实正确性', '满足用户需求', '逻辑连贯性', '完备性'],
#     '生成型回答': ['事实正确性', '满足用户需求', '逻辑连贯性', '创造性', '丰富度'],
#     '建议型回答': ['事实正确性', '满足用户需求', '公平与可负责程度', '创造性']
# }


# def detect_mapping(text):
#     if '清晰度' in text and '完备性' in text:
#         return '事实与解释型回答'
#     elif '完备性' in text and '逻辑连贯性' in text:
#         return '逻辑推理型回答'
#     elif '创造性' in text and '丰富度' in text:
#         return '生成型回答'
#     elif '创造性' in text and '公平与可负责程度' in text:
#         return '建议型回答'
#     else:
#         return None


def extract_rating(text):
    pattern = r'({.*?})(?![^{]*{)'  # match last brackets
    match = re.search(pattern, text)

    if match:
        dictionary_str = match.group(1)
        return json.loads(dictionary_str)

    else:
        return None

def check_rating(rating, all_dimensions):
    for k, v in rating.items():
        if isinstance(v, (int, float)) and k in all_dimensions:  # 确保值是数字
            if v >= 0 and v <= 10:
                pass
            else:
                return None
        else:
            return None
    return rating


def post_process_alignbench(judgement: str):
    """Input a string like below:
    xxx{\n\"鲁棒性\": \"良好,70\",\n\"准确性\": \"良好,72\",\n\"翔实性\": \"良好,73\"\n}
    and extract each score
    """

    judgement = judgement.replace('\n', '')
    rating = extract_rating(judgement)
    rating = {k:int(v.split(",")[1].strip()) for k,v in rating.items()}
    if rating is not None:
        rating['综合得分'] = round(statistics.mean(rating.values()), 2)
        return {'rating': rating}
    else:
        return {'rating': {'综合得分':0}}


def get_dimension_results(judged_answers, references, fout, fout_flag, model, ds_name):
    dimension_ratings = defaultdict(int)
    dimension_counts = defaultdict(int)
    for ans, ref in zip(judged_answers, references):
        for k, v in ans['rating'].items():
            if k != '综合得分' or k != 'Overall Score':
                dimension_ratings[k] += v
                dimension_counts[k] += 1
            else:
                if k == '综合得分':
                    dimension_ratings['综合得分'] += ans['score']
                    dimension_counts['综合得分'] += 1
                else:
                    dimension_ratings['Overall Score'] += ans['score']
                    dimension_counts['Overall Score'] += 1

    dimension_avg_ratings = defaultdict(float)
    for dimension, total_score in dimension_ratings.items():
        s = total_score / dimension_counts[dimension]
        s = round(s, 2)
        dimension_avg_ratings[dimension] = s

    scores = {model: dimension_avg_ratings}
    rows = list(scores.keys())
    columns = list(scores[rows[0]].keys())
    with open(fout, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if fout_flag == 0 and os.path.getsize(fout) == 0:
            writer.writerow(['模型', '维度'] + columns)

        for row in rows:
            writer.writerow([row, ds_name] +
                            [scores[row][column] for column in columns])


def get_capability_results(judged_answers,
                           references,
                           fout,
                           fout_flag,
                           model):
    capability_ratings = defaultdict(int)
    capability_counts = defaultdict(int)
    for ans, ref in zip(judged_answers, references):
        capability_ratings[ref['capability']] += ans['rating']['综合得分']
        capability_counts[ref['capability']] += 1

    capability_avg_ratings = defaultdict(float)

    for capability, total_score in capability_ratings.items():
        s = total_score / capability_counts[capability]
        s = round(s, 2)
        capability_avg_ratings[capability] = s

    capability_avg_ratings['总分'] = statistics.mean(capability_avg_ratings.values())

    scores = {model: capability_avg_ratings}
    with open(fout, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if fout_flag == 0 and os.path.getsize(fout) == 0:

            header = ['模型']
            for category,_ in capability_avg_ratings.items():
                header.append(category)
            writer.writerow(header)

        row = [model]
        # row.append(scores[model]['总分'])
        for _, score in capability_avg_ratings.items():
            row.append(score)
        writer.writerow(row)

    scores = scores[model]

    # Creating a new dictionary with '总分' as the first item
    scores.pop('总分')
    updated_scores = {}
    updated_scores.update(scores)
    return updated_scores


class FEvalSubjectiveSummerizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict, judge_type='general') -> None:
        self.tasks = []
        self.cfg = config
        self.eval_model_cfgs = self.cfg['eval']['partitioner']['models']
        self.eval_model_abbrs = [
            model_abbr_from_cfg(model) for model in self.eval_model_cfgs
        ]
        self.judge_models = self.cfg.get('judge_models', None)
        self.judge_type = judge_type
        assert self.judge_type in [
            'general', 'autoj', 'judgelm'
        ]
        self.judge_map = {
            'general': post_process_alignbench,
            'autoj': post_process_autoj,
            'judgelm': post_process_judgelm
        }
        self.judge_function = self.judge_map[self.judge_type]

    def summarize(self,
                  time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """
        all_scores = {}
        for judge_model in self.judge_models:
            score_by_judgemodel = {}
            judge_abbr = model_abbr_from_cfg(judge_model)
            dataset_cfgs = self.cfg['datasets']
            dataset = dataset_cfgs[0]
            fout_flag, fout_flag2 = 0, 0
            output_dir, results_folder = get_outdir(self.cfg, time_str)
            if self.judge_type == 'general':
                fout = osp.join(
                    output_dir,
                    'F-Eval-主观题-judged-by--' + judge_abbr + '-dimension.csv')
            fout2 = osp.join(
                output_dir,
                'F-Eval-主观题-judged-by--' + judge_abbr + '-capability-'+ dataset['name'] +'.csv')
            
            
                
            for eval_model_abbr in self.eval_model_abbrs:
                subdir = eval_model_abbr + '_judged-by--' + judge_abbr
                subdir_path = os.path.join(results_folder, subdir)
                model = eval_model_abbr
                if os.path.isdir(subdir_path):
                    judged_answers, references = get_judgeanswer_and_reference(
                        dataset, subdir_path, self.judge_function)
                    if len(judged_answers) == 0:
                        score_by_judgemodel[model] = None
                        continue
                    if self.judge_type == 'general':
                        get_dimension_results(judged_answers, references, fout,
                                            fout_flag, model, dataset['name'])
                        fout_flag += 1
                    scores = get_capability_results(judged_answers, references,
                                                    fout2, fout_flag2, model)

                    score_by_judgemodel[model] = scores
                    fout_flag2 += 1
                else:
                    score_by_judgemodel[model] = None
                    print(subdir_path + ' is not exist! please check!')

                all_scores[judge_abbr] = score_by_judgemodel
        return {'F_Eval主观评测': all_scores}

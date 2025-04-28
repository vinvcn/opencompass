# flake8: noqa: E501
import csv
import json
import pandas as pd
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np
from datasets import Dataset, DatasetDict
from mmengine import ConfigDict
from itertools import islice

from opencompass.registry import DICT_POSTPROCESSORS, LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset
from .utils import get_judgeanswer_and_reference


def prompt_construct(sample):
    # dimensions = config.category2dimensions(sample['others']['subcategory'])
    base_prompt = """# 角色
你是一个精确、公正的评分系统，专注于根据鲁棒性、准确性和翔实性三个维度评估答案的质量。

## 背景
用户需要建立一个标准化的评分流程，以便可以依赖这个系统来评估不同问题的答案在鲁棒性、准确性和翔实性三个方面的分数。

## 工作流程
1. 首先，接收用户提交的【问题】、【参考答案】和【待打分的答案】。
2. 其次，分析【待打分的答案】在鲁棒性、准确性和翔实性三个评分维度的表现。具体标准如下：
   - 鲁棒性：考察【待打分的答案】是否按照【问题】的指令要求作答，是否符合预期，即指令遵循能力。
   - 准确性：检查【待打分的答案】中的数字、指标、信息等是否符合事实准确，以及是否与【参考答案】中的对应结果一致，以此考察回答的准确性。
   - 翔实性：评估【待打分的答案】内容是否存在重复、是否全面充分翔实、是否语言连贯。
3. 然后，根据已知的三个评分维度的打分等级和范围进行打分：
   - 差：完全不符合三个评分维度要求的；打分范围：0至25分。
   - 一般：符合小部分要求的；打分范围：26至50分。
   - 良好：符合大部分要求的；打分范围：50至75分。
   - 优秀：完全符合评分维度要求的；打分范围：76至100分。
4. 最后，通过所给的所有【问题】、【参考答案】、【待打分的答案】，逐步的去分析他们的鲁棒性、准确性和翔实性，给出具体打分等级和对应的明确分值。

## 输出格式
- 以json格式展示三个评分的维度，模板：
{{
"鲁棒性": "<鲁棒性等级>,<鲁棒性评分>",
"准确性": "<准确性等级>,<准确性评分>",
"翔实性": "<翔实性等级>,<翔实性评分>"
}}

## 限制与要求
1. 你的输出应该只包含json模板内容，不应该包含模板以外的其他内容。
2. 要求严格打分，打分粒度要细且有差距，不要出现满分。

【问题：{问题}】

【参考答案：{参考答案}】

"""
    prompt = base_prompt.format(问题=sample['题目'],
                                参考答案=sample['标准答案'])

    return prompt


@LOAD_DATASET.register_module()
class FEvalSubjectiveDataset(BaseDataset):
    
    def _load_excel(self, path, name):
        all_sheets_df = pd.read_excel(path, sheet_name=None)
        for sheet_name, sheet in all_sheets_df.items():
            sheet.fillna('', inplace=True)
            if name == sheet_name:
                return list(sheet.iterrows())
            
    def load(self,
             path: str,
             name: str,
             limit:int = 0,
             *args,
             **kwargs):
        path = get_data_path(path)
        raw_dataset = self._load_excel(path, name)
        if limit > 0:
            raw_dataset = islice(raw_dataset, limit)
        dataset = []
        for raw_item in raw_dataset:
            raw_item = raw_item[1]
            prefix = prompt_construct(raw_item)
            data = {}
            data['question'] = raw_item['题目']
            data['critiquellm_prefix'] = prefix
            data['ref'] = str(raw_item['标准答案'])
            data['judge'] = { 'gold': str(raw_item.get('标准答案', '')), 'capability': name }
            dataset.append(data)
        dataset = Dataset.from_list(dataset)
        return dataset


# CATEGORIES = {
#     '中文推理': ['数学计算', '逻辑推理'],
#     '中文语言': ['基本任务', '中文理解', '综合问答', '文本写作', '角色扮演', '专业能力'],
# }

# # All_Dimensions = [
# #     '事实正确性', '满足用户需求', '安全无害', '清晰度', '逻辑性', '完备性', '创造性', '可负责程度', '逻辑连贯性',
# #     '公平与可负责程度', '丰富度', '综合得分'
# # ]

# All_Dimensions = [
#     '事实正确性', '满足用户需求'
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


# def extract_missing_rating(text, search_type):
#     searching_keys = MAPPING[search_type]
#     result_dict = {}
#     for k in searching_keys:
#         matches = re.findall(rf'{k}.*?\n', text)
#         result_dict[k] = None
#         for match in reversed(matches):
#             if re.findall(r'\d{1,2}', match):
#                 result_dict[k] = int(re.findall(r'\d{1,2}', match)[-1])
#                 break
#     overall_number = re.findall('\d{1,2}', text)
#     try:
#         result_dict['综合得分'] = int(overall_number[-1])
#     except:
#         return {}
#     return result_dict


# def extract_rating(text):
#     pattern = r'{(.*?)}(?![^{]*{)'  # match last brackets
#     match = re.search(pattern, text)

#     if match:
#         dictionary_str = match.group(1)
#         kv_pattern = r"'(.*?)': (\d+)"
#         matches = re.findall(kv_pattern, dictionary_str)
#         result_dict = {key: int(value) for key, value in matches}
#         return result_dict
#     else:
#         match_type = detect_mapping(text=text)
#         if match_type is not None:
#             return extract_missing_rating(text=text, search_type=match_type)
#         else:
#             return None


# def check_rating(rating, all_dimensions):
#     for k, v in rating.items():
#         if isinstance(v, (int, float)) and k in all_dimensions:  # 确保值是数字
#             if v >= 0 and v <= 10:
#                 pass
#             else:
#                 return None
#         else:
#             return None
#     return rating


# def post_process_alignbench(judgement: dict,
#                             all_dimensions=All_Dimensions,
#                             possible_keys=['综合得分']):
#     """Input a dict item must contain string like below:

#     xxx{'事实正确性': 1, '满足用户需求': 1, '清晰度': 2, '完备性': 1, '综合得分': 1}xxx,
#     and extract each score
#     """
#     judgement = judgement['prediction']

#     def extract_score(text):
#         keys_pattern = '|'.join(map(re.escape, possible_keys))
#         pattern = rf"({'|'.join(possible_keys)}): (\d+(\.\d{{1,2}})?)"
#         match = re.search(pattern, text)
#         if match:
#             try:
#                 return float(match.group(1))
#             except ValueError:
#                 return -1
#         return -1

#     # judgement = judgement.replace('\n', '')
#     rating = extract_rating(judgement)

#     if rating is not None:
#         score = -1
#         for key in possible_keys:
#             score = rating.get(key, -1)
#             if score != -1:
#                 break
#         if score == -1:
#             score = extract_score(judgement)
#         if score >= 0 and score <= 10:
#             pass
#         else:
#             score = -1
#         rating = check_rating(rating, all_dimensions)
#     else:
#         score = -1
#     if rating == None or score == -1:
#         return None
#     else:
#         return {'rating': rating, 'score': score}


# def get_dimension_results(judged_answers, references):
#     dimension_ratings = defaultdict(int)
#     dimension_counts = defaultdict(int)
#     for ans, ref in zip(judged_answers, references):
#         for k, v in ans['rating'].items():
#             if k != '综合得分' or k != 'Overall Score':
#                 dimension_ratings[k] += v
#                 dimension_counts[k] += 1
#             else:
#                 if k == '综合得分':
#                     dimension_ratings['综合得分'] += ans['score']
#                     dimension_counts['综合得分'] += 1
#                 else:
#                     dimension_ratings['Overall Score'] += ans['score']
#                     dimension_counts['Overall Score'] += 1

#     dimension_avg_ratings = defaultdict(float)
#     for dimension, total_score in dimension_ratings.items():
#         s = total_score / dimension_counts[dimension]
#         s = round(s, 2)
#         dimension_avg_ratings[dimension] = s

#     scores = {'dimensional_scores': dimension_avg_ratings}
#     return scores


# def get_capability_results(judged_answers, references, categories=CATEGORIES):
#     capability_ratings = defaultdict(int)
#     capability_counts = defaultdict(int)
#     for ans, ref in zip(judged_answers, references):
#         capability_ratings[ref['capability']] += ans['score']
#         capability_counts[ref['capability']] += 1

#     capability_avg_ratings = defaultdict(float)

#     for capability, total_score in capability_ratings.items():
#         s = total_score / capability_counts[capability]
#         s = round(s, 2)
#         capability_avg_ratings[capability] = s

#     temp_list = []
#     total_column_num = 2
#     for category, sub_categories in categories.items():
#         total_column_num += 1 + len(sub_categories)
#         capability_avg_ratings[category + '总分'] = np.mean([
#             np.mean(capability_avg_ratings[cat])
#             for cat in categories[category]
#         ])
#         capability_avg_ratings[category + '总分'] = round(
#             capability_avg_ratings[category + '总分'], 2)
#         temp_list.append(category + '总分')
#     capability_avg_ratings['总分'] = 0
#     for temp in temp_list:
#         capability_avg_ratings['总分'] += capability_avg_ratings[temp]
#     capability_avg_ratings['总分'] /= len(temp_list)
#     capability_avg_ratings['总分'] = round(capability_avg_ratings['总分'], 2)
#     return capability_avg_ratings


# @DICT_POSTPROCESSORS.register_module('alignbench')
# def alignbench_postprocess(output: dict,
#                            output_path: str,
#                            judge_type: Optional[str] = 'general') -> dict:
#     judged_answers, references = get_judgeanswer_and_reference(
#         output, output_path, post_process_alignbench)

#     if len(judged_answers) == 0:
#         scores = None

#     results = get_capability_results(judged_answers, references)
#     results['details'] = output
#     return results




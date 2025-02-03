# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval. Copy this file and complete it with the info for your task.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

Author:
"""

import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import Metrics, SampleLevelMetric
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


#aquas
def aquas_prompt_pfn(line, task_name: str = None):

    return Doc(
        task_name=task_name,
        query=line["context"]+'\n\n'+line["question"],
        choices=line["answer"],
        gold_index=0,
        instruction="",
    )
aquas_task = LightevalTaskConfig(
    name="aquas",
    prompt_function=aquas_prompt_pfn,
    suite=["community"],
    hf_repo="IIC/AQuAS",
    hf_subset="default",
    hf_avail_splits=['test'],
    evaluation_splits=['test'],
    few_shots_split="test",
    few_shots_select="random_sampling",
    # metric=[Metrics.bert_score], 
    metric=[Metrics.loglikelihood_acc_norm],
    trust_dataset=True,
    version=0,
)


# copa_es
def copa_es_pfn(line, task_name: str = None):
    premise = line["premise"]
    choices = [line["choice1"], line["choice2"]]
    question_map = {"cause": "causa", "effect": "efecto"}
    question = question_map[line["question"]]
    answer = line["label"]

    query = "{}، {} :\n0) {}\n1) {}\nla respuesta:".format(premise, question, choices[0], choices[1])

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=answer,
        instruction="",
    )


copa_es_task = LightevalTaskConfig(
    name="copa_es",
    prompt_function=copa_es_pfn,
    suite=["community"],
    hf_repo="BSC-LT/COPA-es",
    hf_subset="default",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc_norm],
    trust_dataset=True,
    version=0,
)

# paws_es
def paws_es_pfn(line, task_name: str = None):
    sentence1 = line["sentence1"]
    sentence2 = line["sentence2"]
    choices = ['si','no']

    query = "Determina si las siguientes dos frases son parafraseadas:\n\nFrase 1: {}\nFrase 2: {}\n\nResponde 'si' o 'no'.".format(sentence1, sentence2)

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=1 if line["label"] else 0,
        instruction="",
    )


paws_es_task = LightevalTaskConfig(
    name="paws_es",
    prompt_function=paws_es_pfn,
    suite=["community"],
    hf_repo="google-research-datasets/paws-x",
    hf_subset="es",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc_norm],
    trust_dataset=True,
    version=0,
)


# xnli_es
def xnli_es_pfn(line, task_name: str = None):
    premise = line["premise"]
    hypothesis = line["hypothesis"]
    choices = ['entailment','no entailment']

    query = "Dado el siguiente par de premisa y suposición, por favor, determina la relación entre la suposición y la premisa:\n\nPremisa: {}\nSuposición: {}\n\nResponde 'entailment' o 'no entailment'.".format(premise, hypothesis)

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=0 if line["label_name"] == 'entailment' else 0,
        instruction="",
    )


xnli_es_task = LightevalTaskConfig(
    name="xnli_es",
    prompt_function=xnli_es_pfn,
    suite=["community"],
    hf_repo="AntoineBlanot/xnli-es",
    hf_subset="default",
    hf_avail_splits=["test","train"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc_norm],
    trust_dataset=True,
    version=0,
)


# wnli_es
def wnli_es_pfn(line, task_name: str = None):
    sentence1 = line["sentence1"]
    sentence2 = line["sentence2"]
    choices = ['sí','no']

    query = "La tarea consiste en determinar si estas dos frases son coherentes. Las dos frases son:\n\n1: {}\n2: {}\n\n¿Son coherentes estas dos frases? Si son coherentes, elija 'sí'. Si no son coherentes, elija 'no'.".format(sentence1, sentence2)

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=0 if line["label"] else 1,
        instruction="",
    )


wnli_es_task = LightevalTaskConfig(
    name="wnli_es",
    prompt_function=wnli_es_pfn,
    suite=["community"],
    hf_repo="avacaondata/wnli-es",
    hf_subset="default",
    hf_avail_splits=["test", "validation"],
    evaluation_splits=["test"],
    few_shots_split="validation",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc_norm],
    trust_dataset=True,
    version=0,
)


# xstorycloze_es
def xstorycloze_es_pfn(line, task_name: str = None):
    input_sentence_1 = line["input_sentence_1"]
    input_sentence_2 = line["input_sentence_2"]
    input_sentence_3 = line["input_sentence_3"]
    input_sentence_4 = line["input_sentence_4"]
    context = input_sentence_1 + input_sentence_2 + input_sentence_3 + input_sentence_4
    options = [line["sentence_quiz1"],line["sentence_quiz2"]]
    choices = ['A','B']

    query = "Se te proporciona el inicio de una historia. Tu tarea es predecir el siguiente evento más probable que se ajuste al contexto de la historia. Elige la opción que tenga más sentido según la información proporcionada.\n\nHistoria: {}\nPregunta:\nA.{}\nB.{}\nSolo responde A o B.".format(context, options[0],options[1])

    return Doc(
        task_name=task_name,
        query=query,
        choices=choices,
        gold_index=0 if line["answer_right_ending"] else 1,
        instruction="",
    )


xstorycloze_es_task = LightevalTaskConfig(
    name="xstorycloze_es",
    prompt_function=xstorycloze_es_pfn,
    suite=["community"],
    hf_repo="juletxara/xstory_cloze",
    hf_subset="es",
    hf_avail_splits=["eval", "train"],
    evaluation_splits=["eval"],
    few_shots_split="train",
    few_shots_select="sequential",
    metric=[Metrics.loglikelihood_acc_norm],
    trust_dataset=True,
    version=0,
)


# STORE YOUR EVALS
TASKS_TABLE = [aquas_task,copa_es_task,paws_es_task,xnli_es_task,wnli_es_task,xstorycloze_es_task]

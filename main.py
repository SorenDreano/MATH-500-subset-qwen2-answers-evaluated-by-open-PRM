#!/usr/bin/env python
# coding: utf-8

from grader import grade_answer
from utils import prepare_input, prepare_batch_input_for_model, derive_step_rewards, tokenizer_fn, extract_responses, extract_boxed_solution
from utils import base_model_name, prm_model_name, dataset_name, device, seed, max_new_tokens, n_diverse
import tqdm
import random
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from trl import AutoModelForCausalLMWithValueHead

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(device).eval()
base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
dataset = load_dataset(dataset_name)
dataset = dataset.filter(lambda sample: sample["level"] <= 3)["test"].shuffle(seed=seed).select(range(20))

greedy_results = []
greedy_responses = []
for sample in tqdm.tqdm(dataset):    
    tokenized = tokenizer_fn(sample["problem"], base_tokenizer, device)
    
    output = base_model.generate(**tokenized, do_sample=False, temperature=None, top_p=None, top_k=None, num_beams=1, max_new_tokens=max_new_tokens)
    response = extract_responses(output, base_tokenizer)[0]
    
    greedy_results.append(extract_boxed_solution(response))
    greedy_responses.append(response)

dataset = dataset.add_column("greedy_result", greedy_results)
dataset = dataset.add_column("greedy_response", greedy_responses)

diverse_results = []
diverse_responses = []

for sample in tqdm.tqdm(dataset):
    tokenized = tokenizer_fn(sample["problem"], base_tokenizer, device)

    output = base_model.generate(**tokenized, do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=max_new_tokens, num_return_sequences=n_diverse)
    responses = extract_responses(output, base_tokenizer)
    
    diverse_results.append([extract_boxed_solution(response) for response in responses])
    diverse_responses.append(responses)

dataset = dataset.add_column("diverse_result", diverse_results)
dataset = dataset.add_column("diverse_response", diverse_responses)
dataset.save_to_disk("mt500withresponses")

prm_model = AutoModelForCausalLMWithValueHead.from_pretrained(prm_model_name).to(device).eval()
prm_tokenizer = AutoTokenizer.from_pretrained(prm_model_name)

rewards_list = []
for sample in tqdm.tqdm(dataset):
    datas = [{"problem": sample["problem"], "response": sample["greedy_response"]}]
    processed_data = [prepare_input(d["problem"], d["response"], prm_tokenizer, "\n") for d in datas]
    input_ids, steps, reward_flags = zip(*processed_data)
    input_ids, attention_mask, reward_flags = prepare_batch_input_for_model(input_ids, reward_flags, prm_tokenizer.pad_token_id, device)
    _, _, rewards = prm_model(input_ids=input_ids, attention_mask=attention_mask, return_probs=True)
    step_rewards = derive_step_rewards(rewards, reward_flags)
    rewards_list.extend(step_rewards)

dataset = dataset.add_column("greedy_last_reward", [reward_list[:][-1] for reward_list in rewards_list])

rewards_diverse_list = []
with torch.no_grad():
    for sample in tqdm.tqdm(dataset):
        current_diverse_list = []
        for response in sample["diverse_response"]:
            datas = [{"problem": sample["problem"], "response": response}]
            processed_data = [prepare_input(d["problem"], d["response"], prm_tokenizer, "\n") for d in datas]
            input_ids, steps, reward_flags = zip(*processed_data)
            input_ids, attention_mask, reward_flags = prepare_batch_input_for_model(input_ids, reward_flags, prm_tokenizer.pad_token_id, device)
            _, _, rewards = prm_model(input_ids=input_ids, attention_mask=attention_mask, return_probs=True)
            step_rewards = derive_step_rewards(rewards, reward_flags)
            current_diverse_list.extend(step_rewards)
        rewards_diverse_list.append(current_diverse_list)

diverse_last_rewards = []
for scores_diverse_reward in rewards_diverse_list:
    diverse_last_rewards.append([s[-1] for s in scores_diverse_reward])
dataset = dataset.add_column("diverse_last_rewards", diverse_last_rewards)

greedy_accuracy = []
for sample in dataset:
    greedy_accuracy.append(grade_answer(sample["greedy_result"], sample["answer"]))
dataset = dataset.add_column("greedy_accuracy", greedy_accuracy)

best_of_n_result = []
for sample in dataset:
    best_of_n_result.append(sample["diverse_result"][np.argmax(sample["diverse_last_rewards"])])
dataset = dataset.add_column("best_of_n_result", best_of_n_result)

best_of_n_accuracy = []
for sample in dataset:
    best_of_n_accuracy.append(grade_answer(sample["best_of_n_result"], sample["answer"]))
dataset = dataset.add_column("best_of_n_accuracy", best_of_n_accuracy)

dataset.to_csv("test.csv")
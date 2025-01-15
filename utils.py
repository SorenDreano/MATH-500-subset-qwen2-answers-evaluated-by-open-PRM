import torch
import numpy as np
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
prm_model_name = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
dataset_name = "HuggingFaceH4/MATH-500"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
system_prompt = """You are a highly knowledgeable mathematics assistant.
Solve the given math problems step by step, explaining your reasoning clearly and concisely.
Ensure the final answer is provided in the requested format and is accurate.
Use the format \\boxed{answer} to present the final result for easy identification.
It is very important that you only use the \\boxed{answer} format."""
user_prompt = """Use the format \\boxed{answer}, where "answer" is the result of the problem."""
seed = 1
max_new_tokens = 1024
n_diverse = 16

def extract_boxed_solution(text: str) -> str | None:
    """
    Extracts the content of the last `\boxed{}` in a given LaTeX-style text.

    Args:
        text (str): The input string containing LaTeX-style content.

    Returns:
        Optional[str]: The extracted content inside the last `\boxed{}` if found 
        and properly matched, otherwise `None`.

    Example:
        >>> extract_boxed_solution("The result is \\boxed{42}.")
        '42'
        >>> extract_boxed_solution("Unmatched \\boxed{42")
        None
    """
    try:
        start_index = text.rindex("\\boxed{")
        content_start = start_index + 7
        bracket_count = 1
        current_pos = content_start

        while bracket_count > 0 and current_pos < len(text):
            if text[current_pos] == "{":
                bracket_count += 1
            elif text[current_pos] == "}":
                bracket_count -= 1
            current_pos += 1

        if bracket_count == 0:
            content = text[content_start : current_pos - 1].strip()
            return content
        else:
            print("Error: Unmatched brackets in the text")
            return None

    except ValueError:
        print("No boxed solution found in the text")
        return None
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return None

def tokenizer_fn(input_text: str, tokenizer: Qwen2TokenizerFast, device: str) -> BatchEncoding:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{input_text}\n{user_prompt}"},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt").to(device)

def extract_responses(output: torch.Tensor, tokenizer: Qwen2TokenizerFast) -> list[str]:
    detokenized = tokenizer.batch_decode(output, skip_special_tokens=True)
    return [sentence.split("\nassistant\n")[1] for sentence in detokenized]

# The functions below are obtained from https://github.com/SkyworkAI/skywork-o1-prm-inference/blob/main/model_utils/io_utils.py
def prepare_input(problem: str, response: str, tokenizer: Qwen2TokenizerFast, step_token: str) -> tuple[list[str], list[str], list[int]]:
    prompt_ids = tokenizer.encode(tokenizer.bos_token + problem + "\n")
    response_ids = []
    steps = []
    reward_flags = [0] * len(prompt_ids)
    step_token_id = tokenizer.encode(step_token)[-1]
    for idx, step in enumerate(response.split(step_token)):
        if step != "":
            step_ids = tokenizer.encode(step)
        else:
            step_ids = []
        step_ids += [step_token_id]
        step = step + step_token
        flag = [0] * len(step_ids)
        flag[-1] = 1
        response_ids.extend(step_ids)
        reward_flags.extend(flag)
        steps.append(step)
    input_ids = prompt_ids + response_ids
    return input_ids, steps, reward_flags

def prepare_batch_input_for_model(input_ids: tuple[list[str]], reward_flags: tuple[list[int]], pad_token_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor(ids) for ids in input_ids], 
        batch_first=True,
        padding_value=pad_token_id
    )
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor([1] * len(ids)) for ids in input_ids], 
        batch_first=True,
        padding_value=0
    )
    padded_reward_flags = torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor(reward_flag) for reward_flag in reward_flags], 
        batch_first=True,
        padding_value=0
    )
    return padded_input_ids.to(device), padded_attention_mask.to(device), padded_reward_flags.to(device)

def sigmoid(x: float) -> float:
    return 1/(np.exp(-x) + 1)

# This function has be modified to apply the sigmoid function to the logits
def derive_step_rewards(rewards: torch.Tensor, reward_flags: torch.Tensor) -> list[list[float]]:
    batch_size = rewards.shape[0]
    batch_step_rewards = []
    for i in range(batch_size):
        rewards_indices = torch.nonzero(reward_flags[i] == 1).view(-1)
        step_rewards = [sigmoid(rewards[i][rewards_indices[j]].item()) for j in range(len(rewards_indices))]
        batch_step_rewards.append(step_rewards)
    return batch_step_rewards
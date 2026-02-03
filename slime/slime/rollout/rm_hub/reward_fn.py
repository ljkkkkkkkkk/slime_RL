import sys
import os
import re
import asyncio

from verl.utils.reward_score import math_dapo, prime_math
from recipe.r1.tasks import gpqa
from recipe.knapsack_rl import math_utils


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    boxed_pred = math_dapo.last_boxed_only_string(solution)
    extracted_pred = math_dapo.remove_boxed(boxed_pred) if boxed_pred is not None else None

    return extracted_pred


def rllm_math_reward_fn(solution_str: str, ground_truth: str):
    """Reward function for math problems using RLLM's math utils.
    
    Copy from: https://github.com/agentica-project/rllm/blob/7b47687f6a9ef1bf5cbd56dd1af61fff08c4b0e4/rllm/rewards/math_reward.py
    """
    model_response = solution_str
    
    # Extract solution.
    if "</think>" in model_response:
        model_solution = model_response.split("</think>")[1]
    elif "\\boxed" in model_response:
        model_solution = model_response
    else:
        return 0.0, False, "[INVALID]"
    
    model_answer = math_utils.extract_answer(model_solution)
    if model_answer is None:
        return 0.0, False, "[INVALID]"

    # Process the ground truth(s)
    ground_truths = ground_truth
    if ground_truths is None:
        return 0.0, False, "[INVALID]"
    
    # Convert single answer to list for uniform processing
    if isinstance(ground_truths, (str, float, int)):
        ground_truths = [ground_truths]
        
    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = math_utils.extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)
    
    if not processed_ground_truths:
        return 0.0, False, "[INVALID]"

    # Check against all possible correct answers
    for ground_truth in processed_ground_truths:
        is_correct = math_utils.grade_answer_mathd(model_answer, ground_truth) or math_utils.grade_answer_sympy(model_answer, ground_truth)
        if is_correct:
            return 1.0, True, model_answer
    
    return 0.0, False, model_answer


# ---------------------------------------------------------
# 核心入口: compute_score
# ---------------------------------------------------------
async def compute_score(
    args,
    sample,
    **kwargs,
):
    # 1. 字段对齐提取 (核心修复：改用 sample.response)
    data_source = getattr(args, 'data_source', 'math')
    
    # 优先取 response，如果由于某种原因框架以后改名了，再尝试 res 作为备选
    solution_str = getattr(sample, 'response', getattr(sample, 'res', ""))
    
    # 获取标准答案
    ground_truth = getattr(sample, 'label', getattr(sample, 'ground_truth', ""))

    # --- 调试日志（可选，确认没问题后可以删掉） ---
    # print(f"[DEBUG] Processing sample: source={data_source}, len={len(solution_str)}")

    # 2. 路由判分逻辑 (保持原样)
    res_val = 0.0
    
    if data_source == "math_dapo" or str(data_source).startswith("aime"):  
        res_val = math_dapo.compute_score(solution_str, ground_truth)
    
    elif data_source in ["AIME", "AIME2025", "AMC", "MATH", "MINERVA", "OLYMPIAD_BENCH", "deepscaler", "DigitalLearningGmbH/MATH-lighteval"]:
        score, _, _ = rllm_math_reward_fn(solution_str, ground_truth)
        res_val = score

    elif data_source in ["Idavidrein/gpqa"]:
        score_val = gpqa.compute_score(solution_str, ground_truth)
        if score_val == 0:
            # 这里调用你之前写的提取函数
            ext = extract_boxed_answer(solution_str)
            if ext == ground_truth:
                score_val = 1.0
        res_val = score_val

    else:
        score, _, _ = rllm_math_reward_fn(solution_str, ground_truth)
        res_val = score

    # 3. 强制转换返回数字
    try:
        final_score = float(res_val) if not isinstance(res_val, dict) else float(res_val.get('score', 0.0))
    except:
        final_score = 0.0

    return final_score
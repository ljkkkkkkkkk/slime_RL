import sglang as sgl
import json
import time
import os

def run_test():
    prompt_path = "/lianjiakun1/data/dataset/deepscaler/test_prompts_200.json"
    model_path = "/lianjiakun1/models/Qwen2.5-Math-7B-FP8"
    
    if not os.path.exists(prompt_path):
        print(f"错误: 找不到文件 {prompt_path}")
        return

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
        prompts=prompts*10

    print(f"正在初始化引擎，加载模型: {model_path} ...")
    # 强制启用 FP8 KV Cache 以测试 H20 的极限性能
    llm = sgl.Engine(
        model_path=model_path,
        max_running_requests=2048,
    )

    sampling_params = {
        "temperature": 0.8, 
        "top_p": 0.95, 
        "max_new_tokens": 2048
    }

    print(f"开始推理 {len(prompts)} 条数据...")
    
    # 预热 (Warmup): 排除初次加载算子的干扰
    llm.generate(prompts[:1], sampling_params)

    # 正式计时
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    total_duration = end_time - start_time
    # 计算实际生成的 Token 数量
    total_tokens = sum(len(output["text"].split()) for output in outputs)

    print("\n" + "="*40)
    print(f"【推理性能报告】")
    print(f"模型路径: {model_path}")
    print(f"总计耗时: {total_duration:.2f} 秒")
    print(f"吞吐量 (TPS): {total_tokens / total_duration:.2f} tokens/s")
    print(f"平均每条延迟: {total_duration / len(prompts):.2f} 秒")
    print("="*40 + "\n")

    llm.shutdown()

if __name__ == "__main__":
    run_test()

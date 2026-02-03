import asyncio
from dataclasses import dataclass
import sys
import os

# 确保能找到当前目录下的函数
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reward_fn import compute_score

@dataclass
class MockSample:
    res: str
    label: str

async def test_reward_model():
    test_cases = [
        {
            "name": "1. 标准提取",
            "res": "Therefore, the answer is \\boxed{12}.",
            "label": "12",
            "expected": 1.0
        },
        {
            "name": "2. 乱码+多Boxed (测最后一个)",
            "res": "Is it \\boxed{10}? No. Smithsonian 抓紧... it is \\boxed{12}. <|endoftext|>",
            "label": "12",
            "expected": 1.0
        },
        {
            "name": "3. 答案包含 LaTeX 格式",
            "res": "The solution is \\boxed{2x-3y=0}.",
            "label": "2x-3y=0",
            "expected": 1.0
        },
        {
            "name": "4. 纯文本兜底提取 (无boxed)",
            "res": "The final answer is 12",
            "label": "12",
            "expected": 1.0
        },
        {
            "name": "5. 错误答案",
            "res": "I think it is \\boxed{5}.",
            "label": "12",
            "expected": 0.0
        }
    ]

    print(f"\n{'Test Case':<35} | {'Score':<8} | {'Status'}")
    print("-" * 60)

    for case in test_cases:
        sample = MockSample(res=case["res"], label=case["label"])
        
        class MockArgs:
            data_source = "MATH"
        
        try:
            score = await compute_score(MockArgs(), sample)
            # 因为我们在 compute_score 里强制返回了 float，这里直接比对
            pass_test = (float(score) == case["expected"])
            status = "✅ PASS" if pass_test else f"❌ FAIL (Got {score})"
        except Exception as e:
            status = f"💥 ERROR: {str(e)}"
            score = "N/A"
        
        print(f"{case['name']:<35} | {score:<8} | {status}")

if __name__ == "__main__":
    asyncio.run(test_reward_model())
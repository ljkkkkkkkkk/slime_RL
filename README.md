# slime_RL

This repository is used to run experiments for **low-precision training and rollout**.

## Usage

### 1. Prepare model files
Create your own `models/` directory and download the model you want to use into it.

### 2. Scripts overview
Go to the `scripts/` directory. There are three example scripts:

- `Qwen2.5-fp8`  
  Uses **FP8 low-precision** settings (custom args and HuggingFace model path).

- `Qwen2.5-bf16`  
  Uses **BF16 precision** with different low-precision arguments and model path.

- The remaining script is a **reference example**.

The main differences between the scripts are:
- low-precision related arguments
- HuggingFace model paths

### 3. Run experiments
Before running any script, make sure all arguments match your requirements.

Detailed instructions can be found here:  
👉 https://github.com/THUDM/slime/blob/main/docs/en/get_started/quick_start.md


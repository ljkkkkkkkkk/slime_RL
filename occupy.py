import torch
import torch.multiprocessing as mp
import time
import os


def high_gpu_load_task(device_id, tensor_size, stop_event):
    """
    在指定的GPU上执行持续的高强度计算任务。
    参数:
    device_id (int): 要使用的GPU设备ID。
    tensor_size (int): 用于计算的方阵维度，值越大，负载越高。
    stop_event (mp.Event): 用于通知进程停止的多进程事件。
    """
    # 将此进程绑定到指定的GPU
    torch.cuda.set_device(device_id)
    device_name = torch.cuda.get_device_name(device_id)
    print(f"进程 {os.getpid()} 已启动，目标 GPU: {device_id} ({device_name})")

    # 创建一个巨大的张量并将其移至目标GPU
    # 对于 A100，可以从一个较大的尺寸开始，例如 20000
    try:
        tensor = torch.randn(tensor_size, tensor_size, device=f'cuda:{device_id}')
        print(f"GPU {device_id}: 已成功分配大小为 ({tensor_size}, {tensor_size}) 的张量。")
    except torch.cuda.OutOfMemoryError:
        print(
            f"错误：GPU {device_id} 内存不足，无法分配大小为 ({tensor_size}, {tensor_size}) 的张量。请尝试减小 tensor_size。")
        return

    # 无限循环执行计算，直到收到停止信号
    while not stop_event.is_set():
        try:
            # 执行一系列复杂的矩阵乘法以增加计算密度
            a = torch.matmul(tensor, tensor)
            b = torch.matmul(a, tensor)
            c = torch.matmul(b, a)
            # 你可以在这里添加更多计算来微调负载

            # 短暂休眠以避免100%满载并允许监控工具刷新
            # 对于 >30% 的负载，休眠时间可以很短或为零
            time.sleep(0.01)
        except torch.cuda.OutOfMemoryError:
            print(f"错误：在 GPU {device_id} 的计算过程中发生内存不足。")
            stop_event.set()  # 通知其他进程也停止
            break
        except Exception as e:
            print(f"GPU {device_id} 发生错误: {e}")
            stop_event.set()  # 通知其他进程也停止
            break

    print(f"进程 {os.getpid()} 在 GPU {device_id} 上已停止。")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires NVIDIA GPUs.")
        exit()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("未检测到任何 GPU。")
        exit()

    print(f"检测到 {num_gpus} 个 GPU。将在所有 GPU 上启动高负载任务。")

    # --- 参数调整 ---
    # 对于 A100 GPU，可以尝试 15000 到 30000 之间的值。
    # 这个值对 GPU 使用率和内存占用影响巨大。
    # 建议从 20000 开始，然后根据 `nvidia-smi` 的输出来微调。
    TENSOR_SIZE = 8000

    try:
        # 设置多进程启动方法
        mp.set_start_method('spawn', force=True)

        # 创建一个事件，用于优雅地停止所有子进程
        stop_event = mp.Event()

        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=high_gpu_load_task, args=(i, TENSOR_SIZE, stop_event))
            processes.append(p)
            p.start()

        print("所有 GPU 负载进程已启动。按 Ctrl+C 停止。")

        # 保持主进程运行，并监听键盘中断信号
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n收到停止信号... 正在优雅地关闭所有 GPU 进程...")
        stop_event.set()
        for p in processes:
            p.join()  # 等待所有进程结束
        print("所有进程已成功关闭。")

    except Exception as e:
        print(f"主进程发生错误: {e}")
        stop_event.set()
        for p in processes:
            p.join()
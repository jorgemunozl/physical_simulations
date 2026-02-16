"""
I want to train a neural network 
for 10 epochs while count to 3.
"""
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

def decay_loss(epoch: int) -> float:
    return 1 / (1 + epoch)

def task_block():
    print("Training Neural Network...")
    for i in range(10):
        print("------")
        print(f"Epoch {i} completed")
        print(f"Loss: {decay_loss(i)}")
        time.sleep(1)
        print("----")
    return "Task completed"

def count_to_3():
    for i in range(3):
        print(i)
        time.sleep(1)
    return "Count to 3 completed"

async def task_async():
    print("Task started")
    for i in range(3):
        print(i)
        await asyncio.sleep(1)
    return "Task completed"

async def run_tasks():
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        await asyncio.gather(
            loop.run_in_executor(pool, task_block),
            task_async(),
        )

asyncio.run(run_tasks())
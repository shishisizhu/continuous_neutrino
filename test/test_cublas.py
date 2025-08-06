import setproctitle
import multiprocessing as mp
import torch
import os
import sys

def run_cublas():
    a = torch.randn(4, 4, device='cuda')
    b = torch.randn(4, 4, device='cuda')
    try:
        c = a @ b
        print(f"[pid] {os.getpid()}")
        print(c)
        sys.stdout.flush()
        return True
    except Exception as e:
        print(f"[PID]: {os.getpid()} Error: {e}")
        sys.stdout.flush()
        return False

def process_run(queue):
    #run_cublas() //It would be right if you run cublas before setproctitle. 
    setproctitle.setproctitle(f"test_process_{os.getpid()}")
    print(f"[pid] {os.getpid()} before run cublas")
    success = run_cublas()
    print(f"[pid] {os.getpid()} after run cublas")
    queue.put({'pid': os.getpid(), 'success': success})

if __name__ == "__main__":
    mp.set_start_method('spawn')
    result_queue = mp.Queue()
    processes = []
    run_cublas()
    for i in range(2):
        p = mp.Process(target=process_run, args=(result_queue,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("\n" + "="*50)
    print("Exit information:")
    for i, p in enumerate(processes):
        print(f"Process {i+1} (PID: {p.pid}) exited with code {p.exitcode}")
    print("\nExecution results:")
    while not result_queue.empty():
        res = result_queue.get()
        status = "Success" if res['success'] else "Failed"
        print(f"PID {res['pid']}: {status}")


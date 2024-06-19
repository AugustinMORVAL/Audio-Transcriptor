from tqdm import tqdm
from time import sleep

with  tqdm(total=100, desc="Loading", unit="t", ncols=100, colour="green") as pbar:
    for i in range(10):
        sleep(1)
        print("Loading...")
        pbar.update(10)
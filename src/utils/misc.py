from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def starmap_with_progress(pool, func, args, total):
    """Wrapper for Pool.starmap with tqdm progress bar."""
    results = []
    with tqdm(total=total, desc="Processing") as pbar:
        for result in pool.starmap(func, args):
            results.append(result)
            pbar.update()
    return results
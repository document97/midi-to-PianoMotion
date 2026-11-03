"""Live progress of the long parallel stages: the log shows a slow stage while it runs, not only in the report."""
import time

EVERY_SHARE = .1                 # a line at every tenth of the jobs ...
EVERY_SECONDS = 20.              # ... and at least this often


def _clock(seconds):
    m, s = divmod(int(round(seconds)), 60)
    return f'{m}m{s:02d}s'


def progress_map(pool, fn, jobs, label, chunksize=1):
    """``pool.map(fn, jobs)`` as a list, logging done/total, elapsed and the time left."""
    jobs = list(jobs)
    total, out = len(jobs), []
    if not total:
        return out
    start = last = time.perf_counter()
    mark = EVERY_SHARE
    print(f'[{label}] {total} jobs', flush=True)
    for result in pool.map(fn, jobs, chunksize=chunksize):
        out.append(result)
        now = time.perf_counter()
        done = len(out)
        if done == total or done / total >= mark or now - last >= EVERY_SECONDS:
            mark = (int(done / total / EVERY_SHARE) + 1) * EVERY_SHARE
            last = now
            left = (now - start) / done * (total - done)
            print(f'[{label}] {done}/{total} ({100 * done // total}%) {_clock(now - start)}, ~{_clock(left)} left', flush=True)
    return out

import numpy as np
from mp_utils import JobRunner, Worker

def fun(num):
    x = np.random.rand(num)
    print(x)

def testrun():
    workers = [Worker(method=fun) for _ in range(4)]

    jobrunner = JobRunner(workers, max_run_calls=1)
    jobs = [((i,), {},) for i in range(5)]
    jobrunner.run(jobs)
    print('second run')
    jobrunner.run(jobs)
    print('third run')
    jobrunner.run(jobs)

if __name__=='__main__':

    testrun()
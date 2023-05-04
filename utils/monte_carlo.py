import numpy as np
from tqdm import tqdm


def estimate_distance(
    inv_repartition_function1, inv_repartition_function2, nb_sample, seed=10
):
    """
    repartition_function1 : callable that give
                            repartition function of first law
    repartition_function2 : same but for law 2
    seed : number to initialize randomness
    """
    sampler = np.random.default_rng(seed=seed)
    estim = 0
    for iter in tqdm(range(nb_sample)):
        sample = sampler.uniform()
        estim += np.abs(
            inv_repartition_function1(sample) - inv_repartition_function2(sample)
        )
    return estim / nb_sample

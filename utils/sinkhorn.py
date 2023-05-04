import numpy as np
import time

seed = 10
nb_samples = 1000
eps = 1e-1
tau = 1e-4
nb_steps = 5


def calculate_distance_matrix(a1: np.array, a2: np.array) -> np.array:
    """
    Calculate the pairwise distance between two samples

    Args:
        sample1 (np.array): Array of values
        sample2 (np.array): Array of values

    Returns:
        distance d (np.array): Matrix containing the pairwise distance between sample1 and sample2
    """
    d = np.zeros((len(a1), len(a1)))
    for i, sample1 in enumerate(a1):
        for j, sample2 in enumerate(a2):
            d[i][j] = np.sqrt(np.sum((sample1 - sample2) ** 2))
    return d


def sinkhorn(
    first_distribution,
    second_distribution,
    nb_samples: int,
    eps: float,
    tau: float,
    nb_steps: int,
):
    """

    Args:
        first_distribution (_type_): First distribution from whom we sample
        second_distribution (_type_): Second distribution from whom we sample
        nb_samples (int): Number of samples taken for each distribution
        eps (float): Relative value taken to normalize the distance_matrix
        tau (float): Threshold for which we pass has_converged as True
        nb_steps (int): Number of steps taken by the algorithm before returning a value
    """
    # Creating the sample
    uniform1, uniform2 = np.random.uniform(size=nb_samples), np.random.uniform(
        size=nb_samples
    )
    sample1 = first_distribution.ppf(uniform1)
    sample2 = second_distribution.ppf(uniform2)

    # Assign the necessary parameters values for the algorithm
    distance_matrix = calculate_distance_matrix(a1=sample1, a2=sample2)
    relative_distance: float = eps * np.mean(distance_matrix)
    K: np.array = np.exp(-distance_matrix / relative_distance)
    has_converged: bool = False

    v_t = np.ones(nb_samples)

    t1 = time.time()
    if nb_steps:
        for step in range(nb_steps):
            u_t = 1 / nb_samples * 1 / (K @ v_t)
            v_t = 1 / nb_samples * 1 / (K.T @ u_t)
            has_converged = (
                np.linalg.norm(u_t * (K @ v_t) - 1 / nb_samples, ord=1)
                + np.linalg.norm(v_t * (K.T @ u_t) - 1 / nb_samples, ord=1)
            ) < tau
            if has_converged:
                break

    else:
        c = 1
        u_t = 1 / nb_samples * 1 / (K @ v_t)
        v_t = 1 / nb_samples * 1 / (K.T @ u_t)
        has_converged = (
            np.linalg.norm(u_t * (K @ v_t) - 1 / nb_samples, ord=1)
            + np.linalg.norm(v_t * (K.T @ u_t) - 1 / nb_samples, ord=1)
        ) < tau
        while not has_converged:
            u_t = 1 / nb_samples * 1 / (K @ v_t)
            v_t = 1 / nb_samples * 1 / (K.T @ u_t)
            has_converged = (
                np.linalg.norm(u_t * (K @ v_t) - 1 / nb_samples, ord=1)
                + np.linalg.norm(v_t * (K.T @ u_t) - 1 / nb_samples, ord=1)
            ) < tau
            c += 1
        has_converged = c
    t2 = time.time()
    P = np.diag(u_t) @ K @ np.diag(v_t)
    value = np.sum(P)
    return has_converged, value, t2 - t1

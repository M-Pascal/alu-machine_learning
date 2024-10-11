#!/usr/bin/env python3
""" Defines Binomial class that represents binomial distribution """


class Binomial:
    """
    A class to represent the Binomial distribution.

    Constructor:
    def __init__(self, data=None, n=1, p=0.5):
        Initializes the class with number of trials (n) and success probability (p).

    Attributes:
    n (int): Number of Bernoulli trials.
    p (float): Probability of success for each trial.

    Methods:
    pmf(k): Computes the Probability Mass Function (PMF) for a given number of successes (k).
    cdf(k): Computes the Cumulative Distribution Function (CDF) for a given number of successes (k).
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Constructor for the class.

        Args:
            data (list): Dataset to estimate the distribution (optional).
            n (int): Number of Bernoulli trials.
            p (float): Probability of a success in a trial.

        Behavior:
        - If `data` is provided:
            - Compute `n` and `p` based on the data (round `n` to the nearest integer).
            - Raise a `TypeError` if `data` is not a list.
            - Raise a `ValueError` if `data` has fewer than two elements.
        - If `data` is not provided:
            - Use the specified values for `n` and `p`.
            - Raise a `ValueError` if `n` is not a positive integer.
            - Raise a `ValueError` if `p` is not between 0 and 1 (inclusive).
            
        The attributes `n` and `p` are set according to the above logic.
        """
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            else:
                self.n = n
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.p = p
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                mean = float(sum(data) / len(data))
                summation = 0
                for x in data:
                    summation += ((x - mean) ** 2)
                variance = (summation / len(data))
                q = variance / mean
                p = (1 - q)
                n = round(mean / p)
                p = float(mean / n)
                self.n = n
                self.p = p

    def pmf(self, k):
        """
        calculates the value of the PMF for a given number of successes

        parameters:
            k [int]: number of successes
                If k is not an int, convert it to int
                If k is out of range, return 0

        return:
            the PMF value for k
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        p = self.p
        n = self.n
        q = (1 - p)
        n_factorial = 1
        for i in range(n):
            n_factorial *= (i + 1)
        k_factorial = 1
        for i in range(k):
            k_factorial *= (i + 1)
        nk_factorial = 1
        for i in range(n - k):
            nk_factorial *= (i + 1)
        binomial_co = n_factorial / (k_factorial * nk_factorial)
        pmf = binomial_co * (p ** k) * (q ** (n - k))
        return pmf

    def cdf(self, k):
        """
        calculates the value of the CDF for a given number of successes

        parameters:
            k [int]: number of successes
                If k is not an int, convert it to int
                If k is out of range, return 0

        return:
            the CDF value for k
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf

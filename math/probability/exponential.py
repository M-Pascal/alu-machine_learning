#!/usr/bin/env python3
""" Defines Exponential class that represents exponential distribution """


class Exponential:
    """
    This class models an Exponential distribution.

    Constructor:
        def __init__(self, data=None, lambtha=1.0):
            Initializes the class with an optional dataset
            and lambda (rate parameter).

    Attributes:
        lambtha (float): The expected rate of occurrences
        within a given time period.

    Methods:
        pdf(x): Computes the Probability Density Function
        (PDF) for a specified time period.
        cdf(x): Computes the Cumulative Distribution Function
        (CDF) for a specified time period.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Constructor for the class.

        Parameters:
            - data (list): A list of data points used to estimate
            the distribution.
            - lambtha (float): The expected rate of occurrences
            over a given time period.

        Behavior:
        - The instance attribute `lambtha` is set as a float.
        - If `data` is not provided:
            - The provided `lambtha` is used.
            - Raises ValueError if `lambtha` is not positive.
        - If `data` is provided:
            - The `lambtha` is calculated based on the data.
            - Raises TypeError if `data` is not a list.
            - Raises ValueError if `data` contains fewer than two data points.
        """
        if data is None:
            if lambtha < 1:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                lambtha = float(len(data) / sum(data))
                self.lambtha = lambtha

    def pdf(self, x):
        """
        calculates the value of the PDF for a given time period

        parameters:
            x [int]: time period
                If x is out of range, return 0

        return:
            the PDF value for x
        """
        if x < 0:
            return 0
        e = 2.7182818285
        lambtha = self.lambtha
        pdf = lambtha * (e ** (-lambtha * x))
        return pdf

    def cdf(self, x):
        """
        calculates the value of the CDF for a given time period

        parameters:
            x [int]: time period
                If x is out of range, return 0

        return:
            the CDF value for x
        """
        if x < 0:
            return 0
        e = 2.7182818285
        lambtha = self.lambtha
        cdf = 1 - (e ** (-lambtha * x))
        return cdf

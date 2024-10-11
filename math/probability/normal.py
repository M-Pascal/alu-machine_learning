#!/usr/bin/env python3
""" Defines Normal class that represents normal distribution """


class Normal:
    """
    Class to represent a normal distribution.

    Constructor:
        __init__(self, data=None, mean=0.0, stddev=1.0):
            Initializes the distribution with an optional dataset, mean, and standard deviation.

    Attributes:
        mean (float): Represents the mean (average) of the normal distribution.
        stddev (float): Represents the standard deviation, a measure of how spread out the distribution is.

    Methods:
        z_score(self, x):
            Computes the z-score for a given x value (how far x is from the mean in standard deviations).
        
        x_value(self, z):
            Computes the x value for a given z-score (reverse of z_score calculation).

        pdf(self, x):
            Computes the Probability Density Function (PDF) value at a given x (how likely x is within the distribution).

        cdf(self, x):
            Computes the Cumulative Distribution Function (CDF) value at a given x (the probability that a random variable is less than or equal to x).
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Constructor for the class.

        Args:
            data (list): The data used for estimating the distribution.
            mean (float): The mean value of the distribution.
            stddev (float): The standard deviation of the distribution.

        Functionality:
            - Initializes the mean and stddev instance attributes as float values.
            - If no data is provided:
                - Uses the supplied mean and stddev values.
                - Raises a ValueError if stddev is not a positive number.
            - If data is provided:
                - Computes the mean and standard deviation based on the data.
                - Raises a TypeError if data is not a list.
                - Raises a ValueError if the list has fewer than two elements.
        """

        if data is None:
            if stddev < 1:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = float(stddev)
                self.mean = float(mean)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                mean = float(sum(data) / len(data))
                self.mean = mean
                summation = 0
                for x in data:
                    summation += ((x - mean) ** 2)
                stddev = (summation / len(data)) ** (1 / 2)
                self.stddev = stddev

    def z_score(self, x):
        """
        calculates the z-score of a given x-value

        parameters:
            x: x-value

        return:
            z-score of x
        """
        mean = self.mean
        stddev = self.stddev
        z = (x - mean) / stddev
        return z

    def x_value(self, z):
        """
        calculates the x-value of a given z-score

        parameters:
            z: z-score

        return:
            x-value of z
        """
        mean = self.mean
        stddev = self.stddev
        x = (z * stddev) + mean
        return x

    def pdf(self, x):
        """
        calculates the value of the PDF for a given x-value

        parameters:
            x: x-value

        return:
            the PDF value for x
        """
        mean = self.mean
        stddev = self.stddev
        e = 2.7182818285
        pi = 3.1415926536
        power = -0.5 * (self.z_score(x) ** 2)
        coefficient = 1 / (stddev * ((2 * pi) ** (1 / 2)))
        pdf = coefficient * (e ** power)
        return pdf

    def cdf(self, x):
        """
        calculates the value of the CDF for a given x-value

        parameters:
            x: x-value

        return:
            the CDF value for x
        """
        mean = self.mean
        stddev = self.stddev
        pi = 3.1415926536
        value = (x - mean) / (stddev * (2 ** (1 / 2)))
        erf = value - ((value ** 3) / 3) + ((value ** 5) / 10)
        erf = erf - ((value ** 7) / 42) + ((value ** 9) / 216)
        erf *= (2 / (pi ** (1 / 2)))
        cdf = (1 / 2) * (1 + erf)
        return cdf

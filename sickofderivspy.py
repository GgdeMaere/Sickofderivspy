import numpy as np


def errorize(array, pos_error=0, neg_error=0):
    """
    Turns a list or array into a numpy array of ErrorFloats.
    :param array: Array to errorify
    :param pos_error: Upper error
    :param neg_error: Lower error
    """
    new_array = []
    for element in array:
        if type(element) is not ErrorFloat:
            element = ErrorFloat(element, pos_error, neg_error)
        new_array.append(element)
    return np.array(new_array)


def values(array):
    """
    Returns a numpy array containing the values of the ErrorFloats in the given array
    """
    new_array = []
    for element in array:
        if type(element) is not ErrorFloat:
            element = ErrorFloat(element)
        new_array.append(element.value)
    return np.array(new_array)


def pos_errors(array):
    """
    Returns a numpy array containing the upper errors of the ErrorFloats in the given array
    """
    new_array = []
    for element in array:
        if type(element) is not ErrorFloat:
            element = ErrorFloat(element)
        new_array.append(element.pos_e)
    return np.array(new_array)


def neg_errors(array):
    """
    Returns a numpy array containing the lower errors of the ErrorFloats in the given array
    """
    new_array = []
    for element in array:
        if type(element) is not ErrorFloat:
            element = ErrorFloat(element)
        new_array.append(element.neg_e)
    return np.array(new_array)


class ErrorFloat:
    def __init__(self, value, pos_error=0, neg_error=0):
        """
        Creates an ErrorFloat
        :param pos_error: Upper error on the value
        :param neg_error: Lower error on the value
        """
        self.value = value
        self.pos_e = pos_error
        self.neg_e = neg_error

    def __repr__(self):
        return f"ErrorFloat: {self.value} + {self.pos_e} - {self.neg_e}"

    def __str__(self):
        string = "%f + %f - %f" % (self.value, self.pos_e, self.neg_e)
        return string

    def __eq__(self, other):
        return self.value == other.value and self.pos_e == other.pos_e and self.neg_e == other.neg_e

    def __propagate_error(self, other=None, dfdx=0, dfdy=0):
        if other is None:
            new_pos_e = np.abs(dfdx * self.pos_e)
            new_neg_e = np.abs(dfdx * self.neg_e)
        else:
            new_pos_e = np.sqrt((dfdx * self.pos_e)**2 + (dfdy * other.pos_e)**2)
            new_neg_e = np.sqrt((dfdx * self.neg_e)**2 + (dfdy * other.neg_e)**2)
        return new_pos_e, new_neg_e

    def __add__(self, other):
        if type(other) != ErrorFloat:
            other = ErrorFloat(other)
        new_value = self.value + other.value
        dfdx = 1
        dfdy = 1
        new_pos_e, new_neg_e = self.__propagate_error(other, dfdx, dfdy)
        return ErrorFloat(new_value, new_pos_e, new_neg_e)

    def __mul__(self, other):
        if type(other) != ErrorFloat:
            other = ErrorFloat(other)
        new_value = self.value * other.value
        dfdx = other.value
        dfdy = self.value
        new_pos_e, new_neg_e = self.__propagate_error(other, dfdx, dfdy)
        return ErrorFloat(new_value, new_pos_e, new_neg_e)

    def __pow__(self, power, modulo=None):
        if type(power) != ErrorFloat:
            power = ErrorFloat(power)
        new_value = self.value**power.value
        dfdx = power.value*self.value**(power.value - 1)
        dfdy = (self.value**power.value) * np.log(self.value)
        new_pos_e, new_neg_e = self.__propagate_error(power, dfdx, dfdy)
        return ErrorFloat(new_value, new_pos_e, new_neg_e)

    def __pos__(self):
        return self

    def __neg__(self):
        other = ErrorFloat(-1)
        return self.__mul__(other)

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __truediv__(self, other):
        if type(other) != ErrorFloat:
            other = ErrorFloat(other)
        return self.__mul__(other.__pow__(ErrorFloat(-1)))

    def __rtruediv__(self, other):
        if type(other) != ErrorFloat:
            other = ErrorFloat(other)
        return other.__mul__(self.__pow__(ErrorFloat(-1)))

    def __radd__(self, other):
        if type(other) != ErrorFloat:
            other = ErrorFloat(other)
        return other.__add__(self)

    def __rsub__(self, other):
        if type(other) != ErrorFloat:
            other = ErrorFloat(other)
        return other.__sub__(self)

    def __rmul__(self, other):
        if type(other) != ErrorFloat:
            other = ErrorFloat(other)
        return other.__mul__(self)

    def __rpow__(self, other):
        if type(other) != ErrorFloat:
            other = ErrorFloat(other)
        return other.__pow__(self)

    def exp(self):
        new_value = np.exp(self.value)
        dfdx = np.exp(self.value)
        new_pos_e, new_neg_e = self.__propagate_error(dfdx=dfdx)
        return ErrorFloat(new_value, new_pos_e, new_neg_e)

    def log(self):
        new_value = np.log(self.value)
        dfdx = 1/self.value
        new_pos_e, new_neg_e = self.__propagate_error(dfdx=dfdx)
        return ErrorFloat(new_value, new_pos_e, new_neg_e)

    def log2(self):
        new_value = np.log2(self.value)
        dfdx = 1/(self.value*np.log(2))
        new_pos_e, new_neg_e = self.__propagate_error(dfdx=dfdx)
        return ErrorFloat(new_value, new_pos_e, new_neg_e)

    def log10(self):
        new_value = np.log10(self.value)
        dfdx = 1/(self.value*np.log(10))
        new_pos_e, new_neg_e = self.__propagate_error(dfdx=dfdx)
        return ErrorFloat(new_value, new_pos_e, new_neg_e)

    def sin(self):
        new_value = np.sin(self.value)
        dfdx = np.cos(self.value)
        new_pos_e, new_neg_e = self.__propagate_error(dfdx=dfdx)
        return ErrorFloat(new_value, new_pos_e, new_neg_e)

    def cos(self):
        new_value = np.cos(self.value)
        dfdx = -np.sin(self.value)
        new_pos_e, new_neg_e = self.__propagate_error(dfdx=dfdx)
        return ErrorFloat(new_value, new_pos_e, new_neg_e)

    def tan(self):
        new_value = np.tan(self.value)
        dfdx = 1/(np.cos(self.value)**2)
        new_pos_e, new_neg_e = self.__propagate_error(dfdx=dfdx)
        return ErrorFloat(new_value, new_pos_e, new_neg_e)

    def arcsin(self):
        new_value = np.arcsin(self.value)
        dfdx = 1/np.sqrt(1 - self.value**2)
        new_pos_e, new_neg_e = self.__propagate_error(dfdx=dfdx)
        return ErrorFloat(new_value, new_pos_e, new_neg_e)

    def arccos(self):
        new_value = np.arccos(self.value)
        dfdx = -1/np.sqrt(1 - self.value**2)
        new_pos_e, new_neg_e = self.__propagate_error(dfdx=dfdx)
        return ErrorFloat(new_value, new_pos_e, new_neg_e)

    def arctan(self):
        new_value = np.arctan(self.value)
        dfdx = 1/(1 + self.value**2)
        new_pos_e, new_neg_e = self.__propagate_error(dfdx=dfdx)
        return ErrorFloat(new_value, new_pos_e, new_neg_e)

    def arctan2(self, other):
        print('Still haven\'t implemented this? Get to work')
        return NotImplemented

    def sinh(self):
        new_value = np.sinh(self.value)
        dfdx = np.cosh(self.value)
        new_pos_e, new_neg_e = self.__propagate_error(dfdx=dfdx)
        return ErrorFloat(new_value, new_pos_e, new_neg_e)

    def cosh(self):
        new_value = np.cosh(self.value)
        dfdx = np.sinh(self.value)
        new_pos_e, new_neg_e = self.__propagate_error(dfdx=dfdx)
        return ErrorFloat(new_value, new_pos_e, new_neg_e)

    def tanh(self):
        return self.sinh()/self.cosh()

    def arcsinh(self):
        new_value = np.arcsinh(self.value)
        dfdx = 1/np.sqrt(1 + self.value**2)
        new_pos_e, new_neg_e = self.__propagate_error(dfdx=dfdx)
        return ErrorFloat(new_value, new_pos_e, new_neg_e)

    def arccosh(self):
        new_value = np.arccosh(self.value)
        dfdx = 1/(np.sqrt(self.value - 1)*np.sqrt(self.value + 1))
        new_pos_e, new_neg_e = self.__propagate_error(dfdx=dfdx)
        return ErrorFloat(new_value, new_pos_e, new_neg_e)

    def arctanh(self):
        new_value = np.arctanh(self.value)
        dfdx = 1/(1 - self.value**2)
        new_pos_e, new_neg_e = self.__propagate_error(dfdx=dfdx)
        return ErrorFloat(new_value, new_pos_e, new_neg_e)

    def sqrt(self):
        new_value = np.sqrt(self.value)
        dfdx = 1/(2*np.sqrt(self.value))
        new_pos_e, new_neg_e = self.__propagate_error(dfdx=dfdx)
        return ErrorFloat(new_value, new_pos_e, new_neg_e)

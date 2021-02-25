import numpy as np
import sickofderivspy as sod

# Create ErrorFloats
float_0 = sod.ErrorFloat(0.5)  # float_0 = 0.5 \pm 0
float_1 = sod.ErrorFloat(1, 0.5, 0.5)  # float_1 = 1 \pm 0.5
float_2 = sod.ErrorFloat(1.5, pos_error=0.8, neg_error=0.5)  # float_2 = 1.5^{+0.8}_{-0.5}

# Turn the elements in a numpy array into ErrorFloats, potentially with an error
array = np.array([3.14, 1.59, 2.65])
array_with_error = sod.errorize(array, pos_error=3.58, neg_error=9.79)

# Get the values of the elements in the array
array_with_error_values = sod.values(array_with_error)
# Analogously, get positive and negative errors.

import numpy as np
import sickofderivspy as sod
import scipy.constants as cst

float1 = sod.ErrorFloat(3)
float2 = sod.ErrorFloat(1.9, 3, 4)

nparr1 = np.array([float1, float2, 3])
nparr2 = np.array([float2, float1, 0.1])

float1.pos_e = 1
float1.neg_e = 3
float2.value += 0.1

# errorize
assert type(nparr1[0]) == sod.ErrorFloat
nparr1, nparr2 = sod.errorize(nparr1), sod.errorize(nparr2)
assert type(nparr1[0]) == sod.ErrorFloat
assert type(nparr1[2]) == sod.ErrorFloat
assert type(nparr2[2]) == sod.ErrorFloat


# equality
float1_copy = sod.ErrorFloat(3, 1, 3)
assert float1 == float1_copy


# addition
float3 = float1 + float2
nparr3 = nparr1 + nparr2
nparr3_bis = nparr1 + 2
nparr3_ter = 2 + nparr1

assert float3.value == 5
assert float3.pos_e == np.sqrt(10)
assert float3.neg_e == 5
assert (float1 + 2.).value == 5
assert (float1 + 2.).pos_e == float1.pos_e
assert (float1 + 2.).neg_e == float1.neg_e
assert (2. + float1).value == 5
assert (2. + float1).pos_e == float1.pos_e
assert (2. + float1).neg_e == float1.neg_e

assert (abs(sod.values(nparr3) - np.array([5, 5, 3.1])) <= 1e-15).all()
assert (abs(sod.pos_errors(nparr3) - np.array([np.sqrt(10), np.sqrt(10), 0])) <= 1e-15).all()
assert (abs(sod.neg_errors(nparr3) - np.array([5, 5, 0])) <= 1e-15).all()

assert (abs(sod.values(nparr3_bis) - np.array([5, 4, 5])) <= 1e-15).all()
assert (abs(sod.pos_errors(nparr3_bis) - np.array([1, 3, 0])) <= 1e-15).all()
assert (abs(sod.neg_errors(nparr3_bis) - np.array([3, 4, 0])) <= 1e-15).all()

assert (abs(sod.values(nparr3_ter) - np.array([5, 4, 5])) <= 1e-15).all()
assert (abs(sod.pos_errors(nparr3_ter) - np.array([1, 3, 0])) <= 1e-15).all()
assert (abs(sod.neg_errors(nparr3_ter) - np.array([3, 4, 0])) <= 1e-15).all()

# subtraction
float4 = float1 - float2
nparr4 = nparr1 - nparr2
nparr4_bis = nparr1 - 2
nparr4_ter = 2 - nparr1

assert float4.value == 1
assert float4.pos_e == np.sqrt(10)
assert float4.neg_e == 5
assert (float1 - 2.).value == 1
assert (float1 - 2.).pos_e == float1.pos_e
assert (float1 - 2.).neg_e == float1.neg_e
assert (2. - float1).value == -1
assert (2. - float1).pos_e == float1.pos_e
assert (2. - float1).neg_e == float1.neg_e


# multiplication
float5 = float1*float2
nparr5 = nparr1*nparr2
nparr5_bis = nparr1*2
nparr5_ter = 2*nparr1

right_pos_e5 = np.sqrt(4*float1.pos_e**2 + 9*float2.pos_e**2)
right_neg_e5 = np.sqrt(4*float1.neg_e**2 + 9*float2.neg_e**2)

assert float5.value == 6
assert float5.pos_e == right_pos_e5
assert float5.neg_e == right_neg_e5
assert (float1 * 2.).value == 6
assert (float1 * 2.).pos_e == 2*float1.pos_e
assert (float1 * 2.).neg_e == 2*float1.neg_e
assert (2. * float1).value == 6
assert (2. * float1).pos_e == 2*float1.pos_e
assert (2. * float1).neg_e == 2*float1.neg_e

assert (abs(sod.values(nparr5) - np.array([6, 6, 0.3])) <= 1e-15).all()
assert (abs(sod.pos_errors(nparr5) - np.array([right_pos_e5, right_pos_e5, 0])) <= 1e-15).all()
assert (abs(sod.neg_errors(nparr5) - np.array([right_neg_e5, right_neg_e5, 0])) <= 1e-15).all()

assert (abs(sod.values(nparr5_bis) - np.array([6, 4, 6])) <= 1e-15).all()
assert (abs(sod.pos_errors(nparr5_bis) - np.array([2, 6, 0])) <= 1e-15).all()
assert (abs(sod.neg_errors(nparr5_bis) - np.array([6, 8, 0])) <= 1e-15).all()

assert (abs(sod.values(nparr5_bis) - np.array([6, 4, 6])) <= 1e-15).all()
assert (abs(sod.pos_errors(nparr5_bis) - np.array([2, 6, 0])) <= 1e-15).all()
assert (abs(sod.neg_errors(nparr5_bis) - np.array([6, 8, 0])) <= 1e-15).all()


# division
float6 = float1/float2
nparr6 = nparr1/nparr2
nparr6_bis = nparr1/2
nparr6_ter = 2/nparr1

right_pos_e6 = np.sqrt((1/4)*float1.pos_e**2 + (9/16)*float2.pos_e**2)
right_neg_e6 = np.sqrt((1/4)*float1.neg_e**2 + (9/16)*float2.neg_e**2)

assert float6.value == 1.5
assert float6.pos_e == right_pos_e6
assert float6.neg_e == right_neg_e6
assert (float1/2).value == 1.5
assert (float1/2).pos_e == float1.pos_e/2
assert (float1/2).neg_e == float1.neg_e/2
assert (2/float1).value == 2/3
assert (2/float1).pos_e == (2/9)*float1.pos_e
assert (2/float1).neg_e == (2/9)*float1.neg_e

# power
float7 = float1**float2
nparr7 = nparr1**nparr2
nparr7_bis = nparr1**2
nparr7_ter = 2**nparr1

assert float7.value == 9
assert float7.pos_e == np.sqrt(36*float1.pos_e**2 + (9*np.log(3))**2 * float2.pos_e**2)
assert float7.neg_e == np.sqrt(36*float1.neg_e**2 + (9*np.log(3))**2 * float2.neg_e**2)
assert (float1 ** 2.).value == 9
assert (float1 ** 2.).pos_e == 6*float1.pos_e
assert (float1 ** 2.).neg_e == 6*float1.neg_e
assert (2. ** float1).value == 8
assert (2. ** float1).pos_e == 8*np.log(2)*float1.pos_e
assert (2. ** float1).neg_e == 8*np.log(2)*float1.neg_e

# To check:
np.add(float1, float2)
np.subtract(float1, float2)
np.multiply(float1, float2)
np.divide(float1, float2)
np.power(float1, float2)
np.exp(float1)
np.log(float1)
np.log2(float1)
np.log10(float1)
np.positive(float1)
np.negative(float1)
np.sin(float1)
np.cos(float1)
np.tan(float1)
np.arcsin(sod.ErrorFloat(np.pi/4))
np.arccos(sod.ErrorFloat(np.pi/4))
np.arctan(sod.ErrorFloat(np.pi/4))
np.sinh(float1)
np.cosh(float1)
np.tanh(float1)
np.arcsinh(sod.ErrorFloat(np.pi/4))
np.arccosh(sod.ErrorFloat(np.pi/3))
np.arctanh(sod.ErrorFloat(np.pi/4))


# Test a couple of more complicated formulae
def planck(wavelength, temp):
    a = (2*cst.h*cst.c**2) / (wavelength**5)
    b = np.exp(cst.h*cst.c/(wavelength*cst.k*temp)) - 1
    return a * (1/b)


wavelengths = sod.errorize(np.linspace(400e-9, 700e-9, 10))
wavelengths = sod.errorize(wavelengths, 1e-9, 1e-9)
test_temp = sod.ErrorFloat(1000, 150, 400)
intensities = planck(wavelengths, test_temp)

right_int = np.array([2.78163940982688, 29.657245341766828, 219.3739087616884, 1213.4338139410238, 5308.096465112669,
                      19163.299550436448, 59013.777409874434, 159097.97762335374, 383343.1766368364, 839392.0672319535])

right_int_e_p = np.array([15.009644358448508, 147.71705599027874, 1014.5983467678053, 5237.901094050564,
                          21480.651435930256, 72987.22578059217, 212277.42457697232, 542165.4021081055,
                          1241013.9614746433, 2587992.1350367134])

right_int_e_n = np.array([40.02217734998033, 393.8832490937349, 2705.4290670932937, 13967.007919966874,
                          57279.18502754496, 194625.13741089965, 566054.3215324652, 1445732.539062289,
                          3309286.6248313445, 6901158.339407138])


assert (abs(sod.values(intensities) / right_int - 1) <= 1e-14).all()  # Not bad
assert (abs(sod.pos_errors(intensities) / right_int_e_p - 1) <= 1e-4).all()  # This really should be better
assert (abs(sod.neg_errors(intensities) / right_int_e_n - 1) <= 1e-4).all()

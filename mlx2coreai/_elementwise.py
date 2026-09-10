"""Small dtype-preserving decompositions for elementwise CoreAI operations."""
from coreai._compiler.dialects import coreai
from coreai._compiler.ir import ComplexType, IntegerType


def abs_(x):
    if not ComplexType.isinstance(x.type.element_type):
        return coreai.abs_(x)
    real, imag = coreai.real_part(x), coreai.imaginary_part(x)
    # Match MLX's primitive: magnitude in the real component, then a separate
    # cast to float. Preserve its sqrt(r*r + i*i) overflow/NaN behavior.
    magnitude = coreai.sqrt(coreai.broadcasting_add(coreai.broadcasting_mul(real, real), coreai.broadcasting_mul(imag, imag)))
    zero = coreai.broadcast_to(coreai.constant(0, dtype=real.type.element_type), coreai.get_shape(real))
    return coreai.create_complex(magnitude, zero)


def floor(x):
    if IntegerType.isinstance(x.type.element_type):
        return x
    return coreai.broadcasting_floor_divide(x, coreai.constant(1, dtype=x.type.element_type))


def ceil(x):
    if IntegerType.isinstance(x.type.element_type):
        return x
    negative_one = coreai.constant(-1, dtype=x.type.element_type)
    return coreai.broadcasting_mul(floor(coreai.broadcasting_mul(x, negative_one)), negative_one)


def round_(x):
    return x if IntegerType.isinstance(x.type.element_type) else coreai.round_(x)


def sign(x):
    zero = coreai.constant(0, dtype=x.type.element_type)
    positive = coreai.cast(coreai.broadcasting_greater(x, zero), x.type.element_type)
    negative = coreai.cast(coreai.broadcasting_greater(zero, x), x.type.element_type)
    return coreai.broadcasting_sub(positive, negative)


def trunc(x):
    if IntegerType.isinstance(x.type.element_type):
        return x
    zero = coreai.constant(0, dtype=x.type.element_type)
    return coreai.broadcasting_where(coreai.broadcasting_greater(x, zero), floor(x), ceil(x))


def atan2(y, x):
    dtype = x.type.element_type
    zero, one, pi = (coreai.constant(v, dtype=dtype) for v in (0, 1, 3.141592653589793))
    def neg(value):
        return coreai.broadcasting_mul(value, coreai.constant(-1, dtype=dtype))
    choose = coreai.broadcasting_where
    eq, gt = coreai.broadcasting_equal, coreai.broadcasting_greater
    either, both = coreai.broadcasting_or, coreai.broadcasting_and
    # Reciprocal preserves the sign of zero, unlike an ordinary comparison.
    def negative(value):
        return either(gt(zero, value), both(eq(value, zero), gt(zero, coreai.broadcasting_divide(one, value))))
    xn, yn = negative(x), negative(y)
    angle = coreai.atan(coreai.broadcasting_divide(y, choose(eq(x, zero), one, x)))
    angle = choose(xn, coreai.broadcasting_add(angle, choose(yn, neg(pi), pi)), angle)
    half_pi = coreai.broadcasting_mul(pi, coreai.constant(0.5, dtype=dtype))
    zero_angle = choose(xn, choose(yn, neg(pi), pi), y)
    vertical_angle = choose(yn, neg(half_pi), half_pi)
    angle = choose(eq(x, zero), choose(eq(y, zero), zero_angle, vertical_angle), angle)
    infinity = coreai.constant(float('inf'), dtype=dtype)
    both_inf = both(eq(coreai.abs_(x), infinity), eq(coreai.abs_(y), infinity))
    quarter_pi = coreai.broadcasting_mul(pi, coreai.constant(0.25, dtype=dtype))
    inf_angle = choose(xn, coreai.broadcasting_mul(quarter_pi, coreai.constant(3, dtype=dtype)), quarter_pi)
    angle = choose(both_inf, choose(yn, neg(inf_angle), inf_angle), angle)
    nan = either(coreai.broadcasting_not_equal(x, x), coreai.broadcasting_not_equal(y, y))
    return choose(nan, coreai.constant(float('nan'), dtype=dtype), angle)

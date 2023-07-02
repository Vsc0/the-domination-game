from functools import reduce


def composite_dilation(dilation_function):

    def compose(a, b):
        return lambda x: b(a(x))

    return reduce(compose, dilation_function, lambda x: x)

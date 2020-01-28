"""
Pure Python implementation of rANS, by Jamie Townsend, to accompany the
tutorial paper on rANS at https://arxiv.org/abs/2001.09186. The same variable
names are used in this file as are used in the tutorial.

We use the names `push` and `pop` for encoding and decoding respectively. The
compressed state is a pair `m = (s, t)`, where `s` is an int in the range
`[2 ** (s_prec - t_prec), 2 ** s_prec)` and `t` is an immutable stack,
implemented using a cons list, containing ints in the range `[0, 2 ** t_prec)`
(`prec` is short for 'precision'). The precisions must satisfy

  t_prec < s_prec.

For convenient compatibility with C/Numpy types we use the settings
s_prec = 64 and t_prec = 32.

Both the `push` method and the `pop` method assume access to a probability
distribution over symbols. We use the name `x` for a symbol. To describe the
probability distribution we model the real interval [0, 1] with the range of
integers [0, 1, 2, ..., 2 ** p_prec]. Each symbol is represented by a
sub-interval within that range. This can be visualized for a probability
distribution over the set of symbols {'a', 'b', 'c', 'd'}:

    0                                                             1
    |                           |---- P(x)----|                   |
    |                                                             |
    |   'a'           'b'           x == 'c'           'd'        |
    |----------|----------------|-------------|-------------------|
    |                              ^                              |
    |------------ c ------------|--|--- p ----|                   |
    0                            s_bar                        2 ** p_prec

Each sub-interval can be represented by a pair of non-negative integers: `c`
and `p`. As shown in the above diagram, the number `p` represents the width
of the interval corresponding to `x`, so that

  P(x) = p / 2 ** p_prec

where P is the probability mass function of our distribution.

The number `c` represents the beginning of the interval corresponding to `x`,
and is analagous to the cumulative distribution function evaluated on `x`.

The model over symbols, which must be provided by the user, is specified by the
triple (f, g, p_prec). The function g does the mapping

  g: x |-> c, p

and the function f does the mapping

  f: s_bar |-> x, (c, p)

where s_bar is in {0, 1, ..., 2 ** p_prec - 1}. The values returned by f should
be the x, p and c corresponding to the sub-interval containing s_bar, as shown
in the diagram above.
"""
s_prec = 64
t_prec = 32
t_mask = (1 << t_prec) - 1
s_min  = 1 << s_prec - t_prec

#        head , tail
m_init = s_min, ()

def rans(model):
    f, g, p_prec = model
    def push(m, x):
        s, t = m
        c, p = g(x)
        while s >= p << s_prec - p_prec:
            s, t = s >> t_prec, (t, s & t_mask)
        return (s // p << p_prec) + s % p + c, t

    def pop(m):
        s, t = m
        s_bar = s & ((1 << p_prec) - 1)
        x, (c, p) = f(s_bar)
        s = p * (s >> p_prec) + s_bar - c
        while s < s_min:
            t, t_top = t
            s = (s << t_prec) + t_top
        return (s, t), x
    return push, pop

def flatten_stack(t):
    flat = []
    while t:
        t, t_top = t
        flat.append(t_top)
    return flat

def unflatten_stack(flat):
    t = ()
    for t_top in reversed(flat):
        t = t, t_top
    return t


if __name__ == "__main__":
    import math

    log = math.log2

    # We encode some data using the example model in the paper and verify the
    # inequality in equation (7).

    # First setup the model
    p_prec = 3

    # Cumulative probabilities
    cs = {'a': 0,
          'b': 1,
          'c': 3,
          'd': 6}

    # Probability masses
    ps = {'a': 1,
          'b': 2,
          'c': 3,
          'd': 2}

    # Backwards mapping
    s_bar_to_x = {0: 'a',
                  1: 'b', 2: 'b',
                  3: 'c', 4: 'c', 5: 'c',
                  6: 'd', 7: 'd'}

    def f(s_bar):
        x = s_bar_to_x[s_bar]
        c, p = cs[x], ps[x]
        return x, (c, p)

    def g(x):
        return cs[x], ps[x]

    model = f, g, p_prec

    push, pop = rans(model)

    # Some data to compress
    xs = ['a', 'b', 'b', 'c', 'b', 'c', 'd', 'c', 'c']

    # Compute h(xs):
    h = sum(map(lambda x: log(2 ** p_prec / ps[x]), xs))
    print('Information content of sequence: h(xs) = {:.2f} bits.'.format(h))
    print()

    # Initialize the message
    m = m_init

    # Encode the data
    for x in xs:
        m = push(m, x)

    # Verify the inequality in eq (7)
    eps = 2 ** (p_prec + t_prec - s_prec)
    s, t = m
    lhs = log(s_min)
    rhs = log(s) - h + t_prec * len(flatten_stack(t)) + len(xs) * log(1 - eps)
    print('Eq (7) inequality, lhs - rhs: {:.2e}'.format(lhs - rhs))
    print()

    # Decode the message, check that the decoded data matches original
    xs_decoded = []
    for _ in range(len(xs)):
        m, x = pop(m)
        xs_decoded.append(x)

    xs_decoded = reversed(xs_decoded)

    for x_orig, x_new in zip(xs, xs_decoded):
        assert x_orig == x_new

    # Check that the message has been returned to its original state
    assert m == m_init
    print('Decode successful!')
    print()

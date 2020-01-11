import operator
from functools import reduce
import lab as B
import numpy as np

__all__ = ['const', 'var', 'EQ']


def is_number(x):
    """Check whether `x` is a number.

    Args:
        x (object): Object to check.

    Returns:
        bool: `True` if `x` is a number and `False` otherwise.
    """
    return isinstance(x, B.Number)


def is_inf(x):
    """Check whether `x` is infinite.

    Args:
        x (object): Object to check.

    Returns:
        bool: `True` if `x` is infinite and `False` otherwise.
    """
    return is_number(x) and np.isinf(x)


def const(x):
    """Constant polynomial.

    Args:
        x (tensor): Constant.

    Returns:
        :class:`.expq.Poly`: Resulting polynomial.
    """
    return Poly(Term(x))


def var(x, power=1):
    """Polynomial consisting of just a single variable.

    Args:
        x (tensor): Constant.
        power (int, optional): Power. Defaults to one.

    Returns:
        :class:`.expq.Poly`: Resulting polynomial.
    """
    return Poly(Term(1, Factor(x, power)))


class Factor:
    """Variable raised to some power.

    Args:
        var (str): Variable name.
        power (int): Power.
    """

    def __init__(self, var, power):
        self.var = var
        self.power = power

    def eval(self, **var_map):
        """Evaluate factor.

        Args:
            **var_map (tensor): Variable mapping.

        Returns:
            tensor: Evaluated factor.
        """
        return var_map[self.var]**self.power

    def __eq__(self, other):
        return self.var == other.var and self.power == other.power

    def __str__(self):
        return f'{self.var}^{self.power}'

    def __repr__(self):
        return str(self)

    def __mul__(self, other):
        if is_number(other) and other == 1:
            return self
        if not other.var == self.var:
            raise RuntimeError(f'Other factor {other} must be function of '
                               f'same variable "{self.var}".')
        return Factor(self.var, self.power + other.power)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __hash__(self):
        return hash(self.var) + hash(self.power)


class Term:
    """Product of a constant and multiple :class:`.expq.Factor` objects.

    Args:
        const (tensor): Constant.
        *factors (:class:`.expq.Factor`): Factors.
    """

    def __init__(self, const, *factors):
        self.const = const

        # Discard factors if constant is zero.
        if is_number(const) and const == 0:
            factors = []

        # Merge common factors.
        vars = set(x.var for x in factors)
        self.factors = \
            set(reduce(operator.mul,
                       filter(lambda y: y.var == var, factors),
                       1) for var in vars)

    def is_function_of(self, var):
        """Check if this term is a function of some variable.

        Args:
            var (str): Variable name.

        Returns:
            bool: `True` if this term is a function of `var` and `False`
                otherwise.
        """
        return any([x.var == var for x in self.factors])

    def collect(self, factor):
        """Create a new term consisting of the same constant and all factors
        except one.

        Args:
            factor (:class:`.expq`): Factor to exclude.

        Returns:
            :class:`.expq.Term`: Result.
        """
        if factor not in self.factors:
            raise RuntimeError(f'Factor {factor} must be contained in term'
                               f'{self}.')
        return Term(self.const, *(self.factors - {factor}))

    def eval(self, **var_map):
        """Evaluate term.

        Args:
            **var_map (tensor): Variable mapping.

        Returns:
            tensor: Evaluated term.
        """
        return reduce(operator.mul,
                      [x.eval(**var_map) for x in self.factors],
                      self.const)

    def is_constant(self):
        """Check whether this term is constant.

        Returns:
            bool: `True` if the term is constant and `False` otherwise.
        """
        return len(self.factors) == 0

    def substitute(self, var, poly):
        """Substitute a variable for a polynomial.

        Args:
            var (str): Name of variable to substitute.
            poly (:class:`.expq.Poly`): Polynomial to substitute.

        Returns:
            :class:`.expq.Poly`: Result of substitution.
        """
        factors = []
        power = 0
        for factor in self.factors:
            # Retain factor if its variable is different, otherwise save its
            # power to afterwards raise the polynomial to.
            if factor.var == var:
                power = factor.power
            else:
                factors.append(factor)
        return Poly(Term(self.const, *factors))*poly**power

    def __str__(self):
        if len(self.factors) > 0:
            return '{} * {}'.format(self.const,
                                    ' '.join(map(str, self.factors)))
        else:
            return str(self.const)

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        if is_number(other) and other == 0:
            return self
        if not self.factors == other.factors:
            raise RuntimeError(f'Factors of {self} and {other} must match.')
        return Term(self.const + other.const, *self.factors)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if is_number(other) and other == 1:
            return self
        return Term(self.const*other.const,
                    *(list(self.factors) + list(other.factors)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        return self.const == other.const and self.factors == other.factors

    def __neg__(self):
        return Term(-self.const, *self.factors)

    def __hash__(self):
        return hash(self.const) + \
               reduce(operator.add, map(hash, self.factors), 0)


class Poly:
    """Sum of several :class:`.expq.Term` objects.

    Args:
        *terms (:class:`.expq.Term`): Terms.
    """

    def __init__(self, *terms):
        # Merge common terms.
        factor_sets = set(frozenset(x.factors) for x in terms)
        self.terms = set(reduce(operator.add,
                                filter(lambda y: y.factors == x, terms),
                                0) for x in factor_sets)

    def collect(self, factor):
        """Create a new polynomial consisting of terms whose factors contain
        factor and subsequently collect `factor`.

        Args:
            factor (:class:`.expq.Factor`): Factor to collect.

        Returns:
            :class:`.expq.Poly`: Result of collection.
        """
        return Poly(*[x.collect(factor)
                      for x in self.terms if factor in x.factors])

    def reject(self, var):
        """Create a new polynomial excluding terms whose factors contain the
        variable `var`.

        Args:
            var (str): Variable name.

        Returns:
            :class:`.expq.Poly`: Result of rejection.
        """
        return Poly(*[x for x in self.terms if not x.is_function_of(var)])

    def eval(self, **var_map):
        """Evaluate polynomial.

        Args:
            **var_map (tensor): Variable mapping.

        Returns:
            tensor: Evaluted polynomial.
        """
        return reduce(operator.add,
                      [x.eval(**var_map) for x in self.terms],
                      0)

    def is_constant(self):
        """Check whether this polynomial is constant.

        Returns:
            bool: `True` if the polynomial is constant and `False` otherwise.
        """
        return len(self.terms) == 0 or \
               (len(self.terms) == 1 and list(self.terms)[0].is_constant())

    def substitute(self, var, poly):
        """Substitute a variable for a polynomial.

        Args:
            var (str): Name of variable to substitute.
            poly (:class:`.expq.Poly`): Polynomial to substitute.

        Returns:
            :class:`.expq.Poly`: Result of substitution.
        """
        return reduce(operator.add,
                      [x.substitute(var, poly) for x in self.terms],
                      0)

    def __str__(self):
        if len(self.terms) == 0:
            return '0'
        else:
            return ' + '.join([str(term) for term in self.terms])

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        if is_number(other) and other == 0:
            return self
        return Poly(*(list(self.terms) + list(other.terms)))

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return Poly(*[-term for term in self.terms])

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        if is_number(other) and other == 1:
            return self
        return Poly(*[x*y for x in self.terms for y in other.terms])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power, modulo=None):
        if type(power) != int or power < 0:
            raise RuntimeError('can only raise to nonnegative integers')
        return reduce(operator.mul, [self]*power, 1)


class EQ:
    """A constant multiplied by an exponentiated polynomial.

    Args:
        poly (:class:`.expq.Poly`): Polynomial to substitute.
        const (tensor, optional): Constant. Defaults to one.
    """

    def __init__(self, poly, const=1):
        self.const = const
        self.poly = poly

    def substitute(self, var, poly):
        """Substitute a variable for a polynomial.

        Args:
            var (str): Name of variable to substitute.
            poly (:class:`.expq.Poly`): Polynomial to substitute.

        Returns:
            :class:`.expq.EQ`: Result of substitution.
        """
        return EQ(self.poly.substitute(var, poly), self.const)

    def eval(self, **var_map):
        """Evaluate exponentiated quadratic form.

        Args:
            **var_map (tensor): Variable mapping.

        Returns:
            tensor: Evaluated exponentiated quadratic form.
        """
        return B.squeeze(self.const*B.exp(self.poly.eval(**var_map)))

    def integrate(self, *vars, **var_map):
        """Integrate over a subset of the variables from :math:`-\\infty` to
        :math:`\\infty` and evaluate the result.

        Args:
            *vars (str): Variable names.
            **var_map (tensor): Variable mapping.

        Returns:
            tensor: Result.
        """
        eq = self
        for var in vars:
            eq = eq._integrate(var)
        return eq.eval(**var_map)

    def integrate_half(self, *vars, **var_map):
        """Integrate over a subset of the variables from :math:`-\\infty` to
        :math:`0` and evaluate the result.

        Args:
            *vars (str): Variable names.
            **var_map (tensor): Variable mapping.

        Returns:
            tensor: Result.
        """
        if len(vars) == 1:
            return self._integrate_half1(vars[0], **var_map)
        elif len(vars) == 2:
            return self._integrate_half2(*vars, **var_map)
        else:
            raise NotImplementedError()

    def integrate_box(self, *vars_and_lims, **var_map):
        """Integrate over a subset of the variables from some lower limit to
        some upper limit and evaluate the result. Infinity can be specified
        using `np.inf`. Any infinite lower limit corresponds to negative
        infinity, and any infinite upper limit corresponds to positive infinity.

        Args:
            *vars_and_lims (tuple): Three-tuples containing the variable names,
                lower limits, and upper limits.
            **var_map (tensor): Variable mapping.

        Returns:
            tensor: Result.
        """
        # Filter doubly infinite limits.
        filtered_vars_and_lims = []
        expq = self
        for var, lower, upper in vars_and_lims:
            if is_inf(lower) and is_inf(upper):
                expq = expq._integrate(var)
            else:
                filtered_vars_and_lims.append((var, lower, upper))
        vars_and_lims = filtered_vars_and_lims

        # Return if all integration is done.
        if len(vars_and_lims) == 0:
            return expq.eval(**var_map)

        # Integrate over box.
        parts = [expq]
        for var, lower, upper in vars_and_lims:
            parts_new = []
            for part in parts:
                if not is_inf(upper):
                    parts_new.append(part.translate_var(var, upper))
                if not is_inf(lower):
                    parts_new.append(-part.translate_var(var, lower))
            parts = parts_new
        return reduce(operator.add,
                      [part.integrate_half(*list(zip(*vars_and_lims))[0],
                                           **var_map)
                       for part in parts],
                      0)

    def translate_var(self, var_name, poly):
        """Translate a variable by some polynomial; that is, substitute a
        variable for itself plus some polynomial.

        Args:
            var_name (str): Name of variable.
            poly (:class:`.expq.Poly`): Polynomial to shift by.

        Returns:
            :class:`.expq.EQ`: Resulting exponentiated quadratic form.
        """
        return EQ(self.poly.substitute(var_name, var(var_name) + poly),
                  self.const)

    def __mul__(self, other):
        return EQ(self.poly + other.poly,
                  self.const*other.const)

    def __neg__(self):
        return EQ(self.poly, -self.const)

    def _integrate(self, var):
        a = self.poly.collect(Factor(var, 2))
        b = self.poly.collect(Factor(var, 1))
        c = self.poly.reject(var)
        if not a.is_constant():
            raise ValueError('Quadratic coefficient must be constant.')
        a = a.eval()
        return EQ(Poly(Term(-.25/a))*b**2 + c,
                  self.const*(-B.pi/a)**.5)

    def _integrate_half1(self, var, **var_map):
        a = self.poly.collect(Factor(var, 2))
        b = self.poly.collect(Factor(var, 1))
        c = self.poly.reject(var)
        if not a.is_constant():
            raise ValueError('Quadratic coefficient must be constant.')
        a, b, c = [x.eval(**var_map) for x in [a, b, c]]
        return B.squeeze(.5*self.const*(-B.pi/a)**.5*
                         B.exp(-.25*b**2/a + c)*
                         (1 - B.erf(.5*b/(-a)**.5)))

    def _integrate_half2(self, var1, var2, **var_map):
        a11 = self.poly.collect(Factor(var1, 2))
        a22 = self.poly.collect(Factor(var2, 2))
        a12 = self.poly.collect(Factor(var1, 1)).collect(Factor(var2, 1))
        b1 = self.poly.collect(Factor(var1, 1)).reject(var2)
        b2 = self.poly.collect(Factor(var2, 1)).reject(var1)
        c = self.poly.reject(var1).reject(var2)

        # Evaluate.
        if not (a11.is_constant() and a22.is_constant() and a12.is_constant()):
            raise ValueError('Quadratic coefficients must be constant.')
        a11, a22, a12 = [coef*x.eval()
                         for coef, x in zip([-2, -2, -1], [a11, a22, a12])]
        b1, b2 = [x.eval(**var_map) for x in [b1, b2]]
        c = c.eval(**var_map)

        # Determinant of A.
        a_det = a11*a22 - a12**2

        # Inverse of A, corresponds to variance of distribution after
        # completing the square.
        ia11 = a22/a_det
        ia12 = -a12/a_det
        ia22 = a11/a_det

        # Mean of distribution after completing the square.
        mu1 = ia11*b1 + ia12*b2
        mu2 = ia12*b1 + ia22*b2

        # Normalise and compute CDF part.
        x1 = -mu1/ia11**.5
        x2 = -mu2/ia22**.5
        rho = ia12/(ia11*ia22)**.5

        # Evaluate CDF for all `x1` and `x2`.
        orig_shape = B.shape(mu1)
        num = reduce(operator.mul, orig_shape, 1)
        x1 = B.reshape(x1, num)
        x2 = B.reshape(x2, num)
        rho *= B.ones(x1)
        cdf_part = B.reshape(B.bvn_cdf(x1, x2, rho), *orig_shape)

        # Compute exponentiated part.
        quad_form = .5*(ia11*b1**2 + ia22*b2**2 + 2*ia12*b1*b2)
        det_part = 2*B.pi/a_det**.5
        exp_part = B.exp(quad_form + c)*det_part

        return B.squeeze(self.const*cdf_part*exp_part)

    def __str__(self):
        if len(self.poly.terms) == 0:
            return str(self.const)
        else:
            return '{} exp({})'.format(self.const, str(self.poly))

    def __repr__(self):
        return str(self)

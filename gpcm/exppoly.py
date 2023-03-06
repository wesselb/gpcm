import operator
from functools import reduce

import lab as B
import numpy as np
from plum import Dispatcher, PromisedType, isinstance

__all__ = ["var", "const", "ExpPoly"]

_dispatch = Dispatcher()


def safe_sqrt(x):
    """Perform a square root that is safe to use in AD.

    Args:
        x (tensor): Tensor to take square root of.

    Returns:
        tensor: Square root of `x`.
    """
    return B.sqrt(B.maximum(x, B.cast(B.dtype(x), 1e-30)))


def is_inf(x):
    """Check whether `x` is infinite.

    Args:
        x (object): Object to check.

    Returns:
        bool: `True` if `x` is infinite and `False` otherwise.
    """
    return isinstance(x, B.Number) and np.isinf(x)


def const(x):
    """Constant polynomial.

    Args:
        x (tensor): Constant.

    Returns:
        :class:`.exppoly.Poly`: Resulting polynomial.
    """
    return Poly(Term(x))


def var(x, power=1):
    """Polynomial consisting of just a single variable.

    Args:
        x (tensor): Constant.
        power (int, optional): Power. Defaults to one.

    Returns:
        :class:`.exppoly.Poly`: Resulting polynomial.
    """
    return Poly(Term(1, Factor(x, power)))


PromisedPoly = PromisedType()


class Factor:
    """Variable raised to some power.

    Args:
        name (str): Variable name.
        power (int, optional): Power. Defaults to one.
    """

    @_dispatch
    def __init__(self, name: str):
        Factor.__init__(self, name, 1)

    @_dispatch
    def __init__(self, name: str, power: int):
        if power < 1:
            raise ValueError(
                f'Power for variable "{name}" is {power}, but must be at least 1.'
            )
        self.name = name
        self.power = power

    def eval(self, **var_map):
        """Evaluate factor.

        Args:
            **var_map (tensor): Variable mapping.

        Returns:
            tensor: Evaluated factor.
        """
        return var_map[self.name] ** self.power

    def __str__(self):
        return f"{self.name}^{self.power}"

    def __repr__(self):
        return str(self)

    @_dispatch
    def __eq__(self, other: "Factor"):
        return self.name == other.name and self.power == other.power

    @_dispatch
    def __mul__(self, other: "Factor"):
        if other.name == self.name:
            return Factor(self.name, self.power + other.power)
        else:
            raise RuntimeError("Can only multiply factors of the same variable.")

    @_dispatch
    def __mul__(self, other: B.Numeric):
        if other is 1:
            return self
        else:
            raise RuntimeError("Can only multiply factors by one identically.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __hash__(self):
        return hash(self.name) + hash(self.power)


def _merge_common_factors(*factors):
    merged_factors = []
    for name in set(x.name for x in factors):
        associated_factors = filter(lambda x: x.name == name, factors)
        merged_factors.append(reduce(operator.mul, associated_factors, 1))
    return merged_factors


class Term:
    """Product of a constant and multiple :class:`.exppoly.Factor` objects.

    Args:
        const (tensor): Constant.
        *factors (:class:`.exppoly.Factor`): Factors.
    """

    @_dispatch
    def __init__(self, *factors: Factor):
        Term.__init__(self, 1, *factors)

    @_dispatch
    def __init__(self, const: B.Numeric, *factors: Factor):
        self.const = const

        # Discard factors if constant is equal to zero.
        if isinstance(const, B.Number) and const == 0:
            factors = []

        # Merge common factors. Store this as a set so that equality works as
        # expected.
        self.factors = set(_merge_common_factors(*factors))

    @_dispatch
    def is_function_of(self, name: str):
        """Check if this term is a function of some variable.

        Args:
            name (str): Variable name.

        Returns:
            bool: `True` if this term is a function of `var` and `False` otherwise.
        """
        return any([x.name == name for x in self.factors])

    @_dispatch
    def collect_for(self, factor: Factor):
        """Create a new term consisting of the same constant and all factors
        except one.

        Args:
            factor (:class:`.exppoly.Factor`): Factor to exclude.

        Returns:
            :class:`.exppoly.Term`: Result.
        """
        if factor not in self.factors:
            raise RuntimeError(f"Factor {factor} must be contained in term" f"{self}.")
        return Term(self.const, *(self.factors - {factor}))

    def eval(self, **var_map):
        """Evaluate term.

        Args:
            **var_map (tensor): Variable mapping.

        Returns:
            tensor: Evaluated term.
        """
        return reduce(
            operator.mul, [x.eval(**var_map) for x in self.factors], self.const
        )

    def is_constant(self):
        """Check whether this term is constant.

        Returns:
            bool: `True` if the term is constant and `False` otherwise.
        """
        return len(self.factors) == 0

    @_dispatch
    def highest_power(self, name: str):
        """Find the highest power of a variable.

        Args:
            name (str): Name of variable.

        Returns:
            int: Highest power of variable `name`. Returns `0` if it does not depend
                on `name`.
        """
        highest_power = 0
        for factor in self.factors:
            if factor.name == name:
                highest_power = max(highest_power, factor.power)
        return highest_power

    @_dispatch
    def substitute(self, name: str, poly: PromisedPoly):
        """Substitute a polynomial for a variable.

        Args:
            name (str): Name of variable.
            poly (:class:`.exppoly.Poly`): Polynomial to substitute for variable.

        Returns:
            :class:`.exppoly.Poly`: Result of substitution.
        """
        factors = []
        power = 0
        for factor in self.factors:
            # Retain factor if its variable is different, otherwise save its power to
            # afterwards raise the polynomial to. This check is safe because there
            # are not factors with the same name.
            if factor.name == name:
                power = factor.power
            else:
                factors.append(factor)
        return Poly(Term(self.const, *factors)) * poly**power

    def __str__(self):
        if len(self.factors) > 0:
            names = " ".join(sorted(map(str, self.factors)))
            return f"{self.const} * {names}"
        else:
            return str(self.const)

    def __repr__(self):
        return str(self)

    @_dispatch
    def __eq__(self, other: "Term"):
        return self.const == other.const and self.factors == other.factors

    @_dispatch
    def __add__(self, other: "Term"):
        if self.factors == other.factors:
            return Term(self.const + other.const, *self.factors)
        else:
            raise RuntimeError("Can only add terms of the same factors.")

    @_dispatch
    def __add__(self, other: B.Numeric):
        if other is 0:
            return self
        else:
            raise RuntimeError("Can only add to terms zero identically.")

    def __radd__(self, other):
        return self + other

    @_dispatch
    def __mul__(self, other: "Term"):
        return Term(
            self.const * other.const, *(list(self.factors) + list(other.factors))
        )

    @_dispatch
    def __mul__(self, other: B.Numeric):
        if other is 1:
            return self
        else:
            raise RuntimeError("Can only multiply terms with one identically.")

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return Term(-self.const, *self.factors)

    def __hash__(self):
        return hash(self.const) + hash(frozenset(self.factors))


def _merge_common_terms(*terms):
    merged_terms = []
    for factors in set(frozenset(x.factors) for x in terms):
        associated_terms = filter(lambda x: x.factors == factors, terms)
        merged_terms.append(reduce(operator.add, associated_terms, 0))
    return merged_terms


class Poly:
    """Sum of several :class:`.exppoly.Term` objects.

    Args:
        *terms (:class:`.exppoly.Term`): Terms.
    """

    @_dispatch
    def __init__(self, *terms: Term):
        # Merge common terms. Do _not_ store this as a set, even though we would like
        # to, because in certain AD frameworks the hash of the constant of a term
        # cannot be computed.
        self.terms = _merge_common_terms(*terms)

    @_dispatch
    def is_function_of(self, name: str):
        """Check if this polynomial is a function of some variable.

        Args:
            name (str): Variable name.

        Returns:
            bool: `True` if this term is a function of `var` and `False` otherwise.
        """
        return any([term.is_function_of(name) for term in self.terms])

    @_dispatch
    def collect_for(self, factor: Factor):
        """Create a new polynomial consisting of terms whose factors contain `factor`
        and subsequently collect `factor`, which means that it is excluded in those
        terms.

        Args:
            factor (:class:`.exppoly.Factor`): Factor to collect for.

        Returns:
            :class:`.exppoly.Poly`: Result of collection.
        """
        return Poly(*[x.collect_for(factor) for x in self.terms if factor in x.factors])

    @_dispatch
    def reject(self, name: str):
        """Create a new polynomial excluding terms whose factors contain a variable.

        Args:
            name (str): Variable name.

        Returns:
            :class:`.exppoly.Poly`: Result of rejection.
        """
        return Poly(*[x for x in self.terms if not x.is_function_of(name)])

    def eval(self, **var_map):
        """Evaluate polynomial.

        Args:
            **var_map (tensor): Variable mapping.

        Returns:
            tensor: Evaluated polynomial.
        """
        result = 0
        for term in self.terms:
            result = result + term.eval(**var_map)
        return result

    def is_constant(self):
        """Check whether this polynomial is constant.

        Returns:
            bool: `True` if the polynomial is constant and `False` otherwise.
        """
        return all([x.is_constant() for x in self.terms])

    @_dispatch
    def highest_power(self, name: str):
        """Find the highest power of a variable.

        Args:
            name (str): Name of variable.

        Returns:
            int: Highest power of variable `name`. Returns `0` if it does not depend
                on `name`.
        """
        return max([term.highest_power(name) for term in self.terms])

    @_dispatch
    def substitute(self, name: str, poly: "Poly"):
        """Substitute a polynomial for a variable.

        Args:
            name (str): Name of variable.
            poly (:class:`.exppoly.Poly`): Polynomial to substitute for variable.

        Returns:
            :class:`.exppoly.Poly`: Result of substitution.
        """
        result = 0
        for term in self.terms:
            result = result + term.substitute(name, poly)
        return result

    def __str__(self):
        if len(self.terms) == 0:
            return "0"
        else:
            return " + ".join([str(term) for term in self.terms])

    def __repr__(self):
        return str(self)

    @_dispatch
    def __eq__(self, other: "Poly"):
        return set(self.terms) == set(other.terms)

    @_dispatch
    def __add__(self, other: "Poly"):
        return Poly(*(list(self.terms) + list(other.terms)))

    @_dispatch
    def __add__(self, other: B.Numeric):
        if other is 0:
            return self
        else:
            return self + const(other)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return Poly(*[-term for term in self.terms])

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    @_dispatch
    def __mul__(self, other: "Poly"):
        return Poly(*[x * y for x in self.terms for y in other.terms])

    @_dispatch
    def __mul__(self, other: B.Numeric):
        if other is 1:
            return self
        else:
            return self * const(other)

    def __rmul__(self, other):
        return self * other

    @_dispatch
    def __pow__(self, power: int, modulo=None):
        assert modulo is None, 'Keyword "modulo" is not supported.'
        if power < 0:
            raise RuntimeError("Can only raise polynomials to non-negative integers.")
        return reduce(operator.mul, [self] * power, 1)


@_dispatch
def _as_poly(x: Poly):
    return x


@_dispatch
def _as_poly(x: B.Numeric):
    return const(x)


class ExpPoly:
    """A constant multiplied by an exponentiated polynomial.

    Args:
        const (tensor, optional): Constant.
        poly (:class:`.exppoly.Poly`): Polynomial.
    """

    @_dispatch
    def __init__(self, poly: Poly):
        ExpPoly.__init__(self, 1, poly)

    @_dispatch
    def __init__(self, const: B.Numeric, poly: Poly):
        self.const = const
        self.poly = poly

    @_dispatch
    def substitute(self, name: str, poly: Poly):
        """Substitute a polynomial for a variable.

        Args:
            name (str): Name of variable.
            poly (:class:`.exppoly.Poly`): Polynomial to substitute for variable.

        Returns:
            :class:`.exppoly.ExpPoly`: Result of substitution.
        """
        return ExpPoly(self.const, self.poly.substitute(name, poly))

    def eval(self, **var_map):
        """Evaluate the exponentiated polynomial.

        Args:
            **var_map (tensor): Variable mapping.

        Returns:
            tensor: Evaluated exponentiated polynomial.
        """
        return self.const * B.exp(self.poly.eval(**var_map))

    def integrate(self, *names, **var_map):
        """Integrate over a subset of the variables from :math:`-\\infty` to
        :math:`\\infty` and evaluate the result.

        Args:
            *names (str): Variable names.
            **var_map (tensor): Variable mapping.

        Returns:
            tensor: Result.
        """
        eq = self
        for name in names:
            eq = eq._integrate(name)
        return eq.eval(**var_map)

    def integrate_half(self, *names, **var_map):
        """Integrate over a subset of the variables from :math:`-\\infty` to
        :math:`0` and evaluate the result.

        Args:
            *names (str): Variable names.
            **var_map (tensor): Variable mapping.

        Returns:
            tensor: Result.
        """
        if len(names) == 1:
            return self._integrate_half1(names[0], **var_map)
        elif len(names) == 2:
            return self._integrate_half2(*names, **var_map)
        else:  # pragma: no cover
            raise NotImplementedError(
                "Cannot integrate from -inf to 0 over more than two variables."
            )

    def integrate_box(self, *name_and_lims, **var_map):
        """Integrate over a subset of the variables from some lower limit to some
        upper limit and evaluate the result. Infinity can be specified using
        `np.inf`. Any infinite lower limit corresponds to negative infinity, and any
        infinite upper limit corresponds to positive infinity.

        Args:
            *name_and_lims (tuple): Three-tuples containing the variable names,
                lower limits, and upper limits.
            **var_map (tensor): Variable mapping.

        Returns:
            tensor: Result.
        """
        # Filter doubly infinite limits.
        filtered_vars_and_lims = []
        ep = self
        for name, lower, upper in name_and_lims:
            if is_inf(lower) and is_inf(upper):
                ep = ep._integrate(name)
            else:
                filtered_vars_and_lims.append((name, lower, upper))
        name_and_lims = filtered_vars_and_lims

        # Return if all integration is done.
        if len(name_and_lims) == 0:
            return ep.eval(**var_map)

        # Integrate over box by breaking up the integrals.
        parts = [ep]
        for name, lower, upper in name_and_lims:
            parts_new = []
            for part in parts:
                if not is_inf(upper):
                    upper = _as_poly(upper)  # Ensure that it is a polynomial.
                    parts_new.append(part.translate_var(name, upper))
                if not is_inf(lower):
                    lower = _as_poly(lower)  # Ensure that it is a polynomial.
                    parts_new.append(-part.translate_var(name, lower))
            parts = parts_new

        # Perform integration.
        result = 0
        names = [name for name, _, _ in name_and_lims]
        for part in parts:
            result = result + part.integrate_half(*names, **var_map)
        return result

    @_dispatch
    def translate_var(self, name: str, poly: Poly):
        """Translate a variable by some polynomial; that is, substitute a variable
        for itself plus some polynomial.

        Args:
            name (str): Name of variable.
            poly (:class:`.exppoly.Poly`): Polynomial to shift by.

        Returns:
            :class:`.exppoly.ExpPoly`: Resulting exponentiated polynomial.
        """
        return ExpPoly(self.const, self.poly.substitute(name, var(name) + poly))

    def _integrate(self, name):
        if self.poly.highest_power(name) != 2:
            raise RuntimeError(f'Dependency on "{name}" must be quadratic.')

        a = self.poly.collect_for(Factor(name, 2))
        b = self.poly.collect_for(Factor(name, 1))
        c = self.poly.reject(name)

        if not a.is_constant():
            raise RuntimeError(f'Quadratic coefficient for "{name}" must be constant.')

        a = a.eval()

        return ExpPoly(
            self.const * safe_sqrt(-B.pi / a), Poly(Term(-0.25 / a)) * b**2 + c
        )

    def _integrate_half1(self, name, **var_map):
        if self.poly.highest_power(name) != 2:
            raise RuntimeError(f'Dependency on "{name}" must be quadratic.')

        a = self.poly.collect_for(Factor(name, 2))
        b = self.poly.collect_for(Factor(name, 1))
        c = self.poly.reject(name)

        a = a.eval(**var_map)
        b = b.eval(**var_map)
        c = c.eval(**var_map)

        return (
            0.5
            * self.const
            * safe_sqrt(-B.pi / a)
            * B.exp(-0.25 * b**2 / a + c)
            * (1 - B.erf(0.5 * b / safe_sqrt(-a)))
        )

    def _integrate_half2(self, name1, name2, **var_map):
        if self.poly.highest_power(name1) != 2 or self.poly.highest_power(name2) != 2:
            raise RuntimeError(
                f'Dependency on "{name1}" and {name2}" must ' f"be quadratic."
            )

        a11 = self.poly.collect_for(Factor(name1, 2))
        a22 = self.poly.collect_for(Factor(name2, 2))
        a12 = self.poly.collect_for(Factor(name1, 1)).collect_for(Factor(name2, 1))
        b1 = self.poly.collect_for(Factor(name1, 1)).reject(name2)
        b2 = self.poly.collect_for(Factor(name2, 1)).reject(name1)
        c = self.poly.reject(name1).reject(name2)

        # Evaluate and scale A.
        a11 = -2 * a11.eval(**var_map)
        a22 = -2 * a22.eval(**var_map)
        a12 = -1 * a12.eval(**var_map)
        b1 = b1.eval(**var_map)
        b2 = b2.eval(**var_map)
        c = c.eval(**var_map)

        # Determinant of A:
        a_det = a11 * a22 - a12**2

        # Inverse of A, which corresponds to variance of distribution after
        # completing the square:
        ia11 = a22 / a_det
        ia12 = -a12 / a_det
        ia22 = a11 / a_det

        # Mean of distribution after completing the square:
        mu1 = ia11 * b1 + ia12 * b2
        mu2 = ia12 * b1 + ia22 * b2

        # Normalise and compute CDF part.
        x1 = -mu1 / safe_sqrt(ia11)
        x2 = -mu2 / safe_sqrt(ia22)
        rho = ia12 / safe_sqrt(ia11 * ia22)

        # Evaluate CDF for all `x1` and `x2`.
        orig_shape = B.shape(mu1)
        num = reduce(operator.mul, orig_shape, 1)
        x1 = B.reshape(x1, num)
        x2 = B.reshape(x2, num)
        rho = rho * B.ones(x1)
        cdf_part = B.reshape(B.bvn_cdf(x1, x2, rho), *orig_shape)

        # Compute exponentiated part.
        quad_form = 0.5 * (ia11 * b1**2 + ia22 * b2**2 + 2 * ia12 * b1 * b2)
        det_part = 2 * B.pi / safe_sqrt(a_det)
        exp_part = det_part * B.exp(quad_form + c)

        return self.const * cdf_part * exp_part

    def __str__(self):
        return f"{self.const} * exp({self.poly})"

    def __repr__(self):
        return str(self)

    @_dispatch
    def __eq__(self, other: "ExpPoly"):
        return self.const == other.const and self.poly == other.poly

    @_dispatch
    def __mul__(self, other: "ExpPoly"):
        return ExpPoly(self.const * other.const, self.poly + other.poly)

    @_dispatch
    def __mul__(self, other: B.Numeric):
        return ExpPoly(self.const * other, self.poly)

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return ExpPoly(-self.const, self.poly)


PromisedPoly.deliver(Poly)

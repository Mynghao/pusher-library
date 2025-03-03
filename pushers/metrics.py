import numpy as np
from enum import Enum


class Idx(Enum):
    U = 0  # contravariant
    D = 1  # covariant
    T = 2  # tetrad


class Metric:
    """
    Base class for metrics. Should not be instantiated directly.
    All methods take (n x 2) or (n x 3) arrays of coordinates as inputs, where n is the number of particles.

    Static Methods
    --------------
    check_x(x)
        Checks if the input array is of the correct shape
    eLC_ijk(n)
        Returns the Levi-Civita psuedotensor

    Methods
    -------
    gamma(x, u)
        Returns the Lorentz factor of the particle
    d_i_alpha(x, eps)
        Returns the derivative of alpha with respect to x^i
    d_i_betaj(x, eps)
        Returns the derivative of beta^j with respect to x^i
    d_i_hjk(x, eps)
        Returns the derivative of h^ij with respect to x^k
    transform(v, x, frm, to)
        Converts a vector from one basis to another
    """

    def __init__(self): ...

    @staticmethod
    def check_x(x):
        """
        Checks if the input array is of the correct shape

        Parameters
        ----------
        x : np.array (n x D)
            contravariant coordinate x^i
        """
        assert len(x.shape) == 2
        # assert x.shape[1:] == (2,) or x.shape[1:] == (3,)
        assert x.shape[1:] == (2,)

    @staticmethod
    def eLC_ijk(n):
        """
        Levi-Civita psuedotensor

        Parameters
        ----------
        n : int
            number of particles

        Returns
        -------
        np.array (n x 3 x 3 x 3)
            Levi-Civita psuedotensor
        """
        return -np.array(
            [
                [
                    [np.zeros(n), np.zeros(n), np.zeros(n)],
                    [np.zeros(n), np.zeros(n), np.ones(n)],
                    [np.zeros(n), -np.ones(n), np.zeros(n)],
                ],
                [
                    [np.zeros(n), np.zeros(n), -np.ones(n)],
                    [np.zeros(n), np.zeros(n), np.zeros(n)],
                    [np.ones(n), np.zeros(n), np.zeros(n)],
                ],
                [
                    [np.zeros(n), np.ones(n), np.zeros(n)],
                    [-np.ones(n), np.zeros(n), np.zeros(n)],
                    [np.zeros(n), np.zeros(n), np.zeros(n)],
                ],
            ]
        ).T

    def hij(self, _):
        raise NotImplementedError

    def h_ij(self, _):
        raise NotImplementedError

    def alpha(self, _):
        raise NotImplementedError

    def betai(self, _):
        raise NotImplementedError

    def gamma(self, x, u):
        """
        Computes the Lorentz factor

        Parameters
        ----------
        x : np.array (n x D)
            contravariant coordinate x^i
        u : np.array (n x 3)
            covariant velocity u_i

        Returns
        -------
        np.array (n)
            Lorentz factor gamma
        """
        return np.sqrt(
            1
            + np.einsum("ni,ni->n", np.einsum("nij,ni->nj", self.hij(x), u), u)
        )

    def d_i_alpha(self, x, eps=1e-3):
        """
        Computes the derivative of alpha with respect to x^i

        Parameters
        ----------
        x : np.array (n x D)
            contravariant coordinate x^i
        eps : float, optional
            step size for numerical differentiation

        Returns
        -------
        np.array (n x 3 == n x i)
            covariant derivative of alpha with respect to x^i
        """
        Metric.check_x(x)
        n = x.shape[0]
        xr2 = x + np.array([eps, 0])
        xr1 = x - np.array([eps, 0])
        xth2 = x + np.array([0, eps])
        xth1 = x - np.array([0, eps])
        return np.array(
            [
                self.alpha(xr2) - self.alpha(xr1),
                self.alpha(xth2) - self.alpha(xth1),
                np.zeros(n),
            ]
        ).T / (2 * eps)

    def d_i_betaj(self, x, eps=1e-3):
        """
        Computes the derivative of beta^j with respect to x^i

        Parameters
        ----------
        x : np.array (n x D)
            contravariant coordinate x^i
        eps : float, optional
            step size for numerical differentiation

        Returns
        -------
        np.array (n x 3 x 3 == n x i x j)
            covariant derivative of beta^j with respect to x^i
        """
        Metric.check_x(x)
        n = x.shape[0]
        xr2 = x + np.array([eps, 0])
        xr1 = x - np.array([eps, 0])
        xth2 = x + np.array([0, eps])
        xth1 = x - np.array([0, eps])
        return np.array(
            [
                self.betai(xr2) - self.betai(xr1),
                self.betai(xth2) - self.betai(xth1),
                np.zeros((n, 3)),
            ]
        ).swapaxes(0, 1) / (2 * eps)

    def d_i_hjk(self, x, eps=1e-3):
        """
        Computes the derivative of h^ij with respect to x^k

        Parameters
        ----------
        x : np.array (n x D)
            contravariant coordinate x^i
        eps : float, optional
            step size for numerical differentiation

        Returns
        -------
        np.array (n x 3 x 3 x 3 == n x k x i x j)
            covariant derivative of h^ij with respect to x^k
        """
        Metric.check_x(x)
        n = x.shape[0]
        xr2 = x + np.array([eps, 0])
        xr1 = x - np.array([eps, 0])
        xth2 = x + np.array([0, eps])
        xth1 = x - np.array([0, eps])
        return np.array(
            [
                self.hij(xr2) - self.hij(xr1),
                self.hij(xth2) - self.hij(xth1),
                np.zeros((n, 3, 3)),
            ]
        ).swapaxes(0, 1) / (2 * eps)

    def transform(self, v, x, frm, to):
        """
        Converts a vector from one basis to another

        Parameters
        ----------
        v : np.array (n x 3)
            vector to be transformed
        x : np.array (n x D)
            contravariant coordinate x^i
        frm : Idx
            initial basis (Idx.U, Idx.D, Idx.T)
        to : Idx
            target basis (Idx.U, Idx.D, Idx.T)

        Returns
        -------
        np.array (n x 3)
            transformed vector

        Raises
        ------
        ValueError
            If the transformation is invalid
        """
        Metric.check_x(x)
        if frm == Idx.T or to == Idx.T:
            n = x.shape[0]

            # defining metric components
            hrr = self.hij(x)[:, 0, 0]
            h_tt = self.h_ij(x)[:, 1, 1]
            h_pp = self.h_ij(x)[:, 2, 2]
            h_rp = self.h_ij(x)[:, 0, 2]

        # tetrad matrices
        if (frm == Idx.D and to == Idx.T) or (frm == Idx.T and to == Idx.U):
            ei_ih = np.array(
                [
                    [np.sqrt(hrr), np.zeros(n), np.zeros(n)],
                    [np.zeros(n), 1 / np.sqrt(h_tt), np.zeros(n)],
                    [-np.sqrt(hrr) * h_rp / h_pp, np.zeros(n), 1 / np.sqrt(h_pp)],
                ]
            ).T
        elif (frm == Idx.T and to == Idx.D) or (frm == Idx.U and to == Idx.T):
            eih_i = np.array(
                [
                    [1 / np.sqrt(hrr), np.zeros(n), h_rp / np.sqrt(h_pp)],
                    [np.zeros(n), np.sqrt(h_tt), np.zeros(n)],
                    [np.zeros(n), np.zeros(n), np.sqrt(h_pp)],
                ]
            ).T

        #         try:
        #             print(ei_ih, ei_ih.shape)

        #         except:
        #             print(eih_i, eih_i.shape)

        #         print(v, v.shape)

        if frm == Idx.D and to == Idx.T:
            return np.einsum("nji,nj->ni", ei_ih, v)
        elif frm == Idx.T and to == Idx.D:
            return np.einsum("nij,nj->ni", eih_i, v)
        elif frm == Idx.U and to == Idx.T:
            return np.einsum("nji,nj->ni", eih_i, v)
        elif frm == Idx.T and to == Idx.U:
            # print(ei_ih, v)
            return np.einsum("nij,nj->ni", ei_ih, v)
        elif frm == Idx.D and to == Idx.U:
            return np.einsum("nij,nj->ni", self.hij(x), v)
        elif frm == Idx.U and to == Idx.D:
            return np.einsum("nij,nj->ni", self.h_ij(x), v)
        else:
            raise ValueError("Invalid transformation")


class Kerr(Metric):
    """
    Base class for Kerr metrics. Should not be instantiated directly. Inherits from Metric.

    Attributes
    ----------
    a : float
        Dimensionless spin of the black hole

    Methods
    -------
    Sigma(x)
        Computes Sigma = r^2 + a^2 cos^2 theta
    Delta(x)
        Computes Delta = r^2 - 2 r + a^2
    A(x)
        Computes A = (r^2 + a^2)^2 - a^2 Delta sin^2 theta
    """

    def __init__(self, a):
        super().__init__()
        self._a = a

    @property
    def a(self):
        return self._a

    def Sigma(self, x):
        """
        Parameters
        ----------
        x : np.array (n x D)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n)
            Sigma = r^2 + a^2 cos^2 theta
        """
        Metric.check_x(x)
        return x[:, 0] ** 2 + self.a**2 * np.cos(x[:, 1]) ** 2

    def Delta(self, x):
        """
        Parameters
        ----------
        x : np.array (n x D)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n)
            Delta = r^2 - 2 r + a^2
        """
        Metric.check_x(x)
        return x[:, 0] ** 2 - 2 * x[:, 0] + self.a**2

    def A(self, x):
        """
        Parameters
        ----------
        x : np.array (n x D)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n)
            A = (r^2 + a^2)^2 - a^2 Delta sin^2 theta
        """
        Metric.check_x(x)
        return (x[:, 0] ** 2 + self.a**2) ** 2 - self.a**2 * self.Delta(x) * np.sin(
            x[:, 1]
        ) ** 2


class Kerr_BL(Kerr):
    """
    Kerr metric in Boyer-Lindquist coordinates. Inherits from Kerr.

    Methods
    -------
    alpha(x)
        Returns the time-lag factor
    betai(x)
        Returns the shift vector
    hij(x)
        Returns the 3-metric (upper indices)
    h_ij(x)
        Returns the 3-metric (lower indices)
    """

    def __init__(self, a):
        super().__init__(a)

    def alpha(self, x):
        """
        Computes time-lag factor

        Parameters
        ----------
        x : np.array (n x D)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n)
            time-lag factor alpha
        """
        Metric.check_x(x)
        return np.sqrt(self.Delta(x) * self.Sigma(x) / self.A(x))

    def betai(self, x):
        """
        Computes the shift vector

        Parameters
        ----------
        x : np.array (n x D)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n x 3)
            shift vector beta^i
        """
        Metric.check_x(x)
        n = x.shape[0]
        return np.array([np.zeros(n), np.zeros(n), -2 * self.a * x[:, 0] / self.A(x)]).T

    def hij(self, x):
        """
        Computes 3-metric

        Parameters
        ----------
        x : np.array (n x D)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n x 3 x 3 == n x i x j)
            3-metric hij
        """
        Metric.check_x(x)
        n = x.shape[0]
        Sigma = self.Sigma(x)
        Delta = self.Delta(x)
        return np.array(
            [
                [self.A(x) / (Delta * Sigma), np.zeros(n), np.zeros(n)],
                [np.zeros(n), 1 / Sigma, np.zeros(n)],
                [
                    np.zeros(n),
                    np.zeros(n),
                    1 / (Sigma * np.sin(x[:, 1]) ** 2)
                    + (4 * x[:, 0] ** 2 - 1) * self.a**2 / (Delta * Sigma),
                ],
            ]
        ).T

    def h_ij(self, x):
        """
        Computes 3-metric

        Parameters
        ----------
        x : np.array (n x D)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n x 3 x 3 == n x i x j)
            3-metric h_ij (Lower indices)
        """
        Metric.check_x(x)
        n = x.shape[0]
        Sigma = self.Sigma(x)
        Delta = self.Delta(x)
        return np.array(
            [
                [Sigma / Delta, np.zeros(n), np.zeros(n)],
                [np.zeros(n), Sigma, np.zeros(n)],
                [np.zeros(n), np.zeros(n), self.A(x) * np.sin(x[:, 1]) ** 2 / Sigma],
            ]
        ).T


class Kerr_KS(Kerr):
    """
    Kerr metric in Kerr-Schild coordinates. Inherits from Kerr.

    Methods
    -------
    z(x)
        Returns z = 2 r / Sigma
    alpha(x)
        Returns the time-lag factor
    betai(x)
        Returns the shift vector
    hij(x)
        Returns the 3-metric (upper indices)
    h_ij(x)
        Returns the 3-metric (lower indices)
    """

    def __init__(self, a):
        super().__init__(a)

    def z(self, x):
        """
        Parameters
        ----------
        x : np.array (n x 2)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n)
            z = 2 r / Sigma
        """
        Metric.check_x(x)
        return 2 * x[:, 0] / self.Sigma(x)

    def alpha(self, x):
        """
        Computes time-lag factor

        Parameters
        ----------
        x : np.array (n x 2)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n)
            time-lag factor alpha
        """
        Metric.check_x(x)
        return np.sqrt(1 / (1 + self.z(x)))

    def betai(self, x):
        """
        Computes the shift vector

        Parameters
        ----------
        x : np.array (n x 2)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n x 3)
            shift vector beta^i
        """
        Metric.check_x(x)
        n = x.shape[0]
        return np.array([self.z(x) / (1 + self.z(x)), np.zeros(n), np.zeros(n)]).T

    def hij(self, x):
        """
        Computes 3-metric

        Parameters
        ----------
        x : np.array (n x 2)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n x 3 x 3 == n x i x j)
            3-metric hij
        """
        Metric.check_x(x)
        n = x.shape[0]
        A = self.A(x)
        Sigma = self.Sigma(x)
        return np.array(
            [
                [A / (Sigma * (Sigma + 2 * x[:, 0])), np.zeros(n), self.a / Sigma],
                [np.zeros(n), 1 / self.Sigma(x), np.zeros(n)],
                [self.a / Sigma, np.zeros(n), 1 / (Sigma * np.sin(x[:, 1]) ** 2)],
            ]
        ).T

    def h_ij(self, x):
        """
        Computes 3-metric

        Parameters
        ----------
        x : np.array (n x 2)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n x 3 x 3 == n x i x j)
            3-metric h_ij (Lower indices)
        """
        Metric.check_x(x)
        n = x.shape[0]
        A = self.A(x)
        z = self.z(x)
        Sigma = self.Sigma(x)
        a = self.a
        return np.array(
            [
                [1 + z, np.zeros(n), -a * (1 + z) * np.sin(x[:, 1]) ** 2],
                [np.zeros(n), Sigma, np.zeros(n)],
                [
                    -a * (1 + z) * np.sin(x[:, 1]) ** 2,
                    np.zeros(n),
                    A * np.sin(x[:, 1]) ** 2 / Sigma,
                ],
            ]
        ).T


class Minkowski(Metric):
    """
    Minkowski metric in Cartesian coordinates. Inherits from Metric.

    Methods
    -------
    alpha(x)
        Returns the time-lag factor
    betai(x)
        Returns the shift vector
    hij(x)
        Returns the 3-metric (upper indices)
    h_ij(x)
        Returns the 3-metric (lower indices)
    """

    def alpha(self, x):
        """
        Computes time-lag factor

        Parameters
        ----------
        x : np.array (n x 2)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n)
            time-lag factor alpha
        """
        Metric.check_x(x)
        return np.ones(x.shape[0])

    def betai(self, x):
        """
        Computes the shift vector

        Parameters
        ----------
        x : np.array (n x 2)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n x 3)
            shift vector beta^i
        """
        Metric.check_x(x)
        n = x.shape[0]
        return np.zeros((n, 3))

    def hij(self, x):
        """
        Computes 3-metric

        Parameters
        ----------
        x : np.array (n x 2)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n x 3 x 3 == n x i x j)
            3-metric hij
        """
        Metric.check_x(x)
        n = x.shape[0]
        return np.array(
            [
                [np.ones(n), np.zeros(n), np.zeros(n)],
                [np.zeros(n), 1 / x[:, 0] ** 2, np.zeros(n)],
                [np.zeros(n), np.zeros(n), 1 / (x[:, 0] ** 2 * np.sin(x[:, 1]) ** 2)],
            ]
        ).T

    def h_ij(self, x):
        """
        Computes 3-metric

        Parameters
        ----------
        x : np.array (n x 2)
            contravariant coordinate x^i

        Returns
        -------
        np.array (n x 3 x 3 == n x i x j)
            3-metric hij
        """
        Metric.check_x(x)
        n = x.shape[0]
        return np.array(
            [
                [np.ones(n), np.zeros(n), np.zeros(n)],
                [np.zeros(n), x[:, 0] ** 2, np.zeros(n)],
                [np.zeros(n), np.zeros(n), x[:, 0] ** 2 * np.sin(x[:, 1]) ** 2],
            ]
        ).T

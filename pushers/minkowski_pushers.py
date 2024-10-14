# This code assumes q/m = c = 1
import numpy as np
from typing import Dict, Tuple, Callable


class Drag:
    """
    Base class for drag-force pusher.

    Methods
    -------
    cross_section(e)
        Calculates the cross section for hadronic interactions.
    sync_update_u(u, u_prepush, dt, E, B)
        Updates the velocity of the particle with synchrotron radiation.
    had_update_u(u, u_prepush, dt)
        Updates the velocity of the particle with hadronic interactions.
    update_u(u, u_prepush, dt)
        Updates the velocity of the particle.
    """

    def __init__(self, process: str, **args) -> None:
        """
        Parameters
        ----------
        process : str
            Type of process. Can be 'sync', 'pp', 'pg', or 'bth'.
        args : dict
            Additional arguments for the process.

        Notes
        -----
        For non-sync drag, the scheme can be 'prob' or 'cont'.
        """
        assert process in ["sync", "pp", "pg", "bth"], "Invalid process type"
        self.process = process
        if process != "sync":
            self.scheme = args.pop("scheme", "cont")
        self.args = args

    def cross_section(self, e: float) -> float:
        """
        Parameters
        ----------
        e : float
            Energy of the particle in the background frame.

        Returns
        -------
        float
            Cross section for the hadronic interaction.

        Notes
        -----
        For Bethe-Heitler, it returns the cross section times the inelasticity.
        """

        def func(x, s, coefs_p, coefs_q):
            return s * np.polyval(coefs_p, x) / np.polyval(coefs_q, x)

        if self.process == "pp":
            coefs_pp_p = [-0.1341, 6.602, -1.404, 0.1783, 0.004166]
            coefs_pp_q = [-0.3566, 7.135, -2.043, 0.2875, 0.004166]
            s_pp = 1.693
            return 10 ** func(np.log10(e), s_pp, coefs_pp_p, coefs_pp_q)
        elif self.process == "pg":
            coefs_pg_p = [0.5046, -4.889, 8.251, 17.81, 10.45, 2.459, 0.2217]
            coefs_pg_q = [0.469, -5.062, -309.1, 21.06, 13.78, 3.634, 0.3646]
            s_pg = -1.155
            return 10 ** func(np.log10(e), s_pg, coefs_pg_p, coefs_pg_q)
        elif self.process == "bth":
            coefs_bth_p = [
                -0.007843,
                -0.035,
                0.06724,
                0.3862,
                -1.601,
                -1.556,
                -0.2164,
                3.6,
                -0.3493,
                -4.843,
                11.42,
                -20.23,
                15.48,
            ]
            coefs_bth_q = [1]
            s_bth = 1e5
            return func(np.log10(e), s_bth, coefs_bth_p, coefs_bth_q)
        else:
            raise ValueError(
                "Invalid process type for cross section. Must be 'pp', 'pg', or 'bth'"
            )

    def sync_update_u(
        self,
        u_prepush: np.ndarray[float],
        u: np.ndarray[float],
        dt: float,
        E: np.ndarray[float],
        B: np.ndarray[float],
    ) -> np.ndarray[float]:
        """
        Parameters
        ----------
        u_prepush : np.ndarray[float]
            u_i of the particle (size 3) before the EM update.
        u : np.ndarray[float]
            u_i of the particle (size 3) after the EM update.
        E : np.ndarray[float]
            Interpolated electric field at the particle's position (size 3).
        B : np.ndarray[float]
            Interpolated magnetic field at the particle's position (size 3).
        dt : float
            Time step.

        Returns
        -------
        np.ndarray[float]
            Updated u_i of the particle with synchrotron radiation.
        """
        assert self.process == "sync", "Invalid process type for synchrotron radiation"
        B0 = 1
        etarec = 0.1
        gamma_rad = self.args.get("gamma_rad")
        e = E / B0
        b = B / B0

        u_ave = 0.5 * (u + u_prepush)
        gam_ave = np.sqrt(1 + np.sum(u_ave**2))
        v_ave = u_ave / gam_ave

        kap = e + np.cross(np.cross(v_ave, b), b) + np.dot(v_ave, e) * e
        chi = np.sum((e + np.cross(v_ave, b)) ** 2) - np.dot(v_ave, e) ** 2
        const = B0 * etarec / gamma_rad**2

        return u_prepush + const * (kap - gam_ave**2 * chi * v_ave) * dt

    def had_update_u(
        self,
        u_prepush: np.ndarray[float],
        u: np.ndarray[float],
        dt: float,
    ):
        """
        Parameters
        ----------
        u_prepush : np.ndarray[float]
            u_i of the particle (size 3) before the EM update.
        u : np.ndarray[float]
            u_i of the particle (size 3) after the EM update.
        dt : float
            Time step.

        Returns
        -------
        np.ndarray[float]
            Updated u_i of the particle with hadronic interactions
        """
        assert self.process in [
            "pp",
            "pg",
            "bth",
        ], "Invalid process type for hadronic interactions"
        u_ave = 0.5 * (u + u_prepush)

        def transform_v(v_p, v_b):
            gamma = 1 / np.sqrt(1 - np.linalg.norm(v_b) ** 2)
            return (
                1
                / (1 - np.dot(v_p, v_b))
                * (v_p / gamma - v_b + (gamma / (1 + gamma)) * np.dot(v_p, v_b) * v_b)
            )

        if self.process == "pp":
            xi = 0.17
            sigma = self.cross_section(np.sqrt(np.linalg.norm(u_ave) ** 2 + 1))
            u_b = self.args.get("u_bg")
            nr = self.args.get("n_bg")

            gamma_ave = np.sqrt(1 + np.linalg.norm(u_ave) ** 2)
            gamma_b = np.sqrt(1 + np.linalg.norm(u_b) ** 2)

            v_rel = transform_v(u_ave / gamma_ave, u_b / gamma_b)
            gamma_rel = 1 / np.sqrt(1 - np.linalg.norm(v_rel) ** 2)
            n_rel = nr / gamma_b
            dt_rel = dt / gamma_b
            prob = np.linalg.norm(v_rel) * dt_rel * n_rel * sigma

            if self.scheme == "prob":
                if np.random.rand() < prob:
                    u_new_rel = v_rel * gamma_rel * (1 - xi)
                    gamma_new_rel = np.sqrt(1 + np.linalg.norm(u_new_rel) ** 2)
                    v_new = transform_v(u_new_rel / gamma_new_rel, -u_b / gamma_b)
                    gamma_new = 1 / np.sqrt(1 - np.linalg.norm(v_new) ** 2)
                    u_new = v_new * gamma_new
                    return u_new
                return u_prepush
            elif self.scheme == "cont":
                u_new_rel = v_rel * gamma_rel * (1 - prob * xi)
                gamma_new_rel = np.sqrt(1 + np.linalg.norm(u_new_rel) ** 2)
                v_new = transform_v(u_new_rel / gamma_new_rel, -u_b / gamma_b)
                gamma_new = 1 / np.sqrt(1 - np.linalg.norm(v_new) ** 2)
                u_new = v_new * gamma_new
                return u_new
        else:
            raise NotImplementedError("Only pp interactions are implemented")

    def update_u(
        self, u_prepush: np.ndarray[float], u: np.ndarray[float], dt: float, **args
    ) -> np.ndarray[float]:
        """
        Parameters
        ----------
        u_prepush : np.ndarray[float]
            u_i of the particle (size 3) before the EM update.
        u : np.ndarray[float]
            u_i of the particle (size 3) after the EM update.
        dt : float
            Time step.
        args : dict
            Additional arguments for the process.

        Returns
        -------
        np.ndarray[float]
            Updated u_i of the particle.
        """
        if self.process == "sync":
            return self.sync_update_u(u, u_prepush, dt, **args)
        else:
            return self.had_update_u(u, u_prepush, dt, **args)


class Boris:
    """
    Base class for the Boris pusher with radiative losses.

    Methods
    -------
    push(x, u, E_func, B_func, dt)
        Pushes the particle in the electric and magnetic fields with given hadronic interactions.
    update_u(x, u, E, B, dt)
        Updates the velocity of the particle.
    update_x(x, u, dt)
        Updates the position of the particle.
    """

    def __init__(self) -> None: ...

    def update_u(
        self,
        u: np.ndarray[float],
        E: np.ndarray[float],
        B: np.ndarray[float],
        dt: float,
    ) -> np.ndarray[float]:
        """
        Parameters
        ----------
        u : np.ndarray[float]
            u_i of the particle (size 3).
        E : np.ndarray[float]
            Interpolated electric field at the particle's position (size 3).
        B : np.ndarray[float]
            Interpolated magnetic field at the particle's position (size 3).
        dt : float
            Time step.

        Returns
        -------
        np.ndarray[float]
            Updated u_i of the particle.
        """
        k = 0.5 * dt

        u_neg = u + k * E
        t = k * B / np.sqrt(1 + (np.linalg.norm(u_neg)) ** 2)
        s = 2 * t / (1 + np.linalg.norm(t) ** 2)
        u_pos = u_neg + np.cross((u_neg + np.cross(u_neg, t)), s)

        return u_pos + k * E

    def update_x(
        self, x: np.ndarray[float], u: np.ndarray[float], dt: float
    ) -> np.ndarray[float]:
        """
        Parameters
        ----------
        x : np.ndarray[float]
            x_i of the particle (size 3).
        u : np.ndarray[float]
            u_i of the particle (size 3).
        dt : float
            Time step.

        Returns
        -------
        np.ndarray[float]
            Updated x_i of the particle.
        """
        return x + u * dt / np.sqrt(1 + np.linalg.norm(u) ** 2)


class GCA:
    """
    Base class for the GCA pusher.

    Methods
    -------
    push(x, u, E_func, B_func, dt)
        Pushes the particle in the electric and magnetic fields.
    update_u(x, u, E, B, dt)
        Updates the velocity of the particle.
    update_x(x, u, dt)
        Updates the position of the particle.
    """

    def __init__(self, **args) -> None:
        self.args = args

    @staticmethod
    def vE(E: np.ndarray[float], B: np.ndarray[float]) -> np.ndarray[float]:
        w_E = np.cross(E, B) / (np.dot(E, E) + np.dot(B, B))
        return w_E * (1 - np.sqrt(1 - 4 * np.dot(w_E, w_E)))

    @staticmethod
    def kappa(E: np.ndarray[float], B: np.ndarray[float]) -> float:
        ve = GCA.vE(E, B)
        return 1 / np.sqrt(1 - np.dot(ve, ve))

    @staticmethod
    def Gamma(
        upar: float,
        uperp: float,
        E: np.ndarray[float],
        B: np.ndarray[float],
    ) -> float:
        return GCA.kappa(E, B) * np.sqrt(1 + upar**2 + uperp**2)

    @staticmethod
    def ufull(
        upar: float,
        uperp: float,
        E: np.ndarray[float],
        B: np.ndarray[float],
    ) -> np.ndarray[float]:
        b = B / np.linalg.norm(B)
        # perp_v = np.cross(GCA.vE(E + np.array([1e-8, 1e-8, -1e-8]), B), B)
        return (
            upar * b
            + GCA.vE(E, B) * GCA.Gamma(upar, uperp, E, B)
            # + uperp * perp_v / np.linalg.norm(perp_v)
        )

    @staticmethod
    def uperp(
        mu: float,
        E: np.ndarray[float],
        B: np.ndarray[float],
    ) -> float:
        return np.sqrt(2 * np.dot(B, B) * GCA.kappa(E, B) * mu)

    def update_u(
        self,
        u: np.ndarray[float],
        E: np.ndarray[float],
        B: np.ndarray[float],
        dt: float,
    ) -> np.ndarray[float]:
        """
        Parameters
        ----------
        u : np.ndarray[float]
            full u_i of the particle (size 3).
        E : np.ndarray[float]
            Interpolated electric field at the particle's position (size 3).
        B : np.ndarray[float]
            Interpolated magnetic field at the particle's position (size 3).
        dt : float
            Time step.

        Returns
        -------
        np.ndarray[float]
            Updated u_i of the particle: u_new_|| = u_|| + E_|| * dt.

        Notes
        -----
        All extra drifts are neglected (except for ExB).
        """
        mu = self.args.get("mu", 0)
        b = B / np.linalg.norm(B)
        upar = np.dot(u, b)
        return GCA.ufull(upar + np.dot(E, b) * dt, GCA.uperp(mu, E, B), E, B)

    def update_x(
        self,
        x: np.ndarray[float],
        upar: float,
        E_func: Callable[[np.ndarray], np.ndarray],
        B_func: Callable[[np.ndarray], np.ndarray],
        dt: float,
    ) -> np.ndarray[float]:
        """
        Parameters
        ----------
        x : np.ndarray[float]
            x_i of the particle (size 3).
        upar : float
            u_|| of the particle.
        E_func : Callable[[np.ndarray], np.ndarray]
            Function that returns the electric field at a given position.
        B_func : Callable[[np.ndarray], np.ndarray]
            Function that returns the magnetic field at a given position.
        dt : float
            Time step.

        Returns
        -------
        np.ndarray[float]
            Updated x_i of the particle.
        """
        err_crit = self.args.get("err", 1e-6)
        mu = self.args.get("mu", 0)

        E0 = E_func(x)
        B0 = B_func(x)
        b0 = B0 / np.linalg.norm(B0)
        v_E0 = GCA.vE(E0, B0)
        gam0 = GCA.Gamma(upar, GCA.uperp(mu, E0, B0), E0, B0)

        xn = np.copy(x)
        xnew = np.copy(x)
        err = 1

        maxiter = 1000
        cntr = 0
        while err > err_crit:
            if cntr > maxiter:
                raise ValueError("GCA did not converge")
            cntr += 1
            B1 = B_func(xnew)
            E1 = E_func(xnew)
            b1 = B1 / np.linalg.norm(B1)
            v_E1 = GCA.vE(E1, B1)
            gam1 = GCA.Gamma(upar, GCA.uperp(mu, E1, B1), E1, B1)

            v_p_ave = 0.5 * upar * (b0 / gam0 + b1 / gam1)
            v_E_ave = 0.5 * (v_E0 + v_E1)
            dx = (v_p_ave + v_E_ave) * dt
            err = np.linalg.norm(xn + dx - xnew)
            xnew = xn + dx

        return xnew


def IntegrateTrajectory(
    x0: np.ndarray[float],
    u0: np.ndarray[float],
    dt: float,
    tmax: float,
    E_func: Callable[[np.ndarray], np.ndarray] = lambda _: np.array([0, 0, 0]),
    B_func: Callable[[np.ndarray], np.ndarray] = lambda _: np.array([0, 0, 0]),
    drags: Dict[str, Dict[str, float | np.ndarray[float]]] = None,
    use_hybrid: bool = False,
    gca_params: Dict[str, float] = {},
    progressbar=lambda _: _,
) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Integrate the trajectory of the particle in the given fields with specified drag forces.

    Parameters
    ----------
    x0 : np.ndarray[float]
        Initial position of the particle @ n.
    u0 : np.ndarray[float]
        Initial 4-velocity of the particle @ n - 1/2.
    dt : float
        Time step.
    tmax : float
        Maximum time for the integration.
    E_func : Callable[[np.ndarray], np.ndarray], optional
        Function that returns the electric field at a given position, by default lambda _: np.array([0, 0, 0]).
    B_func : Callable[[np.ndarray], np.ndarray], optional
        Function that returns the magnetic field at a given position, by default lambda _: np.array([0, 0, 0]).
    drags : Dict[str, Dict[str, float | np.ndarray[float]]], optional
        Dictionary with drag forces and their parameters, by default None.
    use_hybrid : bool, optional
        Use the hybrid Boris-GCA pusher, by default False.
    gca_params : Dict[str, float], optional
        Parameters for the GCA pusher, by default {}.
    progressbar : lambda, optional
        Progress bar function, by default lambda _: _; can be tqdm.tqdm, ProgIter, etc.

    Returns
    -------
    Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]
        Tuple with the time, position, and velocity of the particle.
        - time: np.ndarray[float]
            Time steps of the integration.
        - position: np.ndarray[float]
            Position of the particle at each time step.
        - velocity: np.ndarray[float]
            Velocity of the particle at each time step.
        - pusher: np.ndarray[str] : only if use_hybrid
            Pusher used at each time step.
    """

    nsteps = int(tmax / dt)

    pusher_boris = Boris()
    if use_hybrid:
        E_ovr_B_max = gca_params.pop("E_ovr_B_max", 0.9)
        larmor_max = gca_params.pop("larmor_max", 0.1)
        pusher_gca = GCA(**gca_params)

    pusher_drags = []
    if drags is not None:
        for drag, drag_params in drags.items():
            pusher_drags.append(Drag(drag, **drag_params))

    x, u = x0, u0
    t = 0
    times = [t]
    xs = [x]
    us = [u]

    if use_hybrid:
        pusher = ["Boris"]

    for it in progressbar(range(nsteps)):
        E, B = E_func(x), B_func(x)
        use_gca = False
        if use_hybrid:
            rho = np.sum(np.cross(u, B) ** 2) ** 0.5 / np.dot(B, B)
            use_gca = (np.linalg.norm(E) / E_ovr_B_max < np.linalg.norm(B)) and (
                rho < larmor_max
            )

        if use_gca:
            u_new = pusher_gca.update_u(u, E, B, dt)
        else:
            u_new = pusher_boris.update_u(u, E, B, dt)

        if np.any([np.isnan(ui) for ui in u_new]):
            if use_hybrid:
                print(
                    f"\nNaNs found in the U update @ t = {t:.2f} [{it} / {nsteps}]\n",
                    f"use_gca = {use_gca}; u = {u}; x = {x}; u_new = {u_new}; x_new = {x_new}",
                    f"rho = {rho}; |E| = {np.linalg.norm(E)}; |B| = {np.linalg.norm(B)}",
                )
            break

        for drag in pusher_drags:
            args = {}
            if drag.process == "sync":
                args["E"] = E
                args["B"] = B
            if use_gca and drag.process == "sync":
                continue
            u_new = drag.update_u(u, u_new, dt, **args)

        if use_gca:
            upar = np.dot(u_new, B / np.linalg.norm(B))
            x_new = pusher_gca.update_x(x, upar, E_func, B_func, dt)
        else:
            x_new = pusher_boris.update_x(x, u_new, dt)

        if np.any([np.isnan(ui) for ui in u_new]):
            if use_hybrid:
                print(
                    f"\nNaNs found in the X update @ t = {t:.2f} [{it} / {nsteps}]\n",
                    f"use_gca = {use_gca}; u = {u}; x = {x}; u_new = {u_new}; x_new = {x_new}",
                    f"rho = {rho}; |E| = {np.linalg.norm(E)}; |B| = {np.linalg.norm(B)}",
                )
            break

        x, u = x_new, u_new

        t += dt
        times.append(t)
        xs.append(x)
        us.append(u)
        if use_hybrid:
            pusher.append("Boris" if not use_gca else "GCA")
    if use_hybrid:
        return (
            np.array(times),
            np.array(xs),
            np.array(us),
            np.array(pusher),
        )
    else:
        return (
            np.array(times),
            np.array(xs),
            np.array(us),
        )

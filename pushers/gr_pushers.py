import numpy as np
from typing import Dict, Tuple, Callable
from pushers.metrics import Metric, Idx

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

class grBoris:
    """
    Base class for the general relativistic Boris pusher.

    Methods
    -------
    error_x(metric, q1, q2, pos)
        Calculates the error in the position update.
    error_u(metric, q1, q2, pos)
        Calculates the error in the 4-velocity update.
    update_u(metric, x, u, E, B, dt)
        Updates the 4-velocity of the particle due to EM forces.
    update_u_i(metric, x, u, dt)
        Updates the covariant 4-velocity of the particle.
    update_xi(metric, x, u, dt)
        Updates the contravariant position of the particle.
    """
    
    def __init__(self): ...
    
    @staticmethod
    def error_x(
        metric: Metric, 
        q1: np.ndarray[float], 
        q2: np.ndarray[float], 
        pos: np.ndarray[float]
    ) -> float:
        x2d = pos[:, :-1]
        dq = np.abs(q2 - q1)
        return (
            np.sqrt(
                np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.h_ij(x2d), dq), dq)
            )[0]
            / np.sqrt(
                np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.h_ij(x2d), q1), q1)
            )[0]
        )

    @staticmethod
    def error_u(
        metric: Metric, 
        q1: np.ndarray[float], 
        q2: np.ndarray[float], 
        pos: np.ndarray[float]
    ) -> float:
        x2d = pos[:, :-1]
        dq = np.abs(q2 - q1)
        return (
            np.sqrt(
                np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.hij(x2d), dq), dq)
            )[0]
            / np.sqrt(
                np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.hij(x2d), q1), q1)
            )[0]
        )

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
            Particle velocity at time n in tetrad basis
        E : np.ndarray[float]
            Electric field at time n in tetrad basis
        B : np.ndarray[float]
            Magnetic field at time n in tetrad basis
        dt : float
            Timestep

        Returns
        -------
        np.ndarray[float]
            Updated particle 4-velocity due to EM fields
        """

        k = 0.5 * dt
        u_neg = u + k * E
        t = k * B / np.sqrt(1 + (np.linalg.norm(u_neg)) ** 2)
        s = 2 * t / (1 + np.linalg.norm(t) ** 2)
        u_pos = u_neg + np.cross((u_neg + np.cross(u_neg, t)), s)
        return u_pos + k * E

    def update_u_i(
        self, 
        metric: Metric, 
        x: np.ndarray[float], 
        u: np.ndarray[float], 
        dt: float
    ) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
        """
        Update the covariant 4-velocity due to spacetime curvature
        u_i^(n-1/2) -> u_i^(n+1/2)

        Parameters
        ----------
        metric : Metric
        x : np.ndarray[float]
            Particle position at time n in tetrad basis
        u : np.ndarray[float]
            Particle 4-velocity at time n-1/2 in tetrad basis
        dt : float
            Timestep

        Returns
        -------
        Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]
            Updated particle 4-velocity, energy, and gamma
        """

        u_new_1 = np.copy(u)
        u_new_2 = np.copy(u)
        x2d = x[:, :-1]
        err = 1
        while err > 10e-8:
            u_mid = 0.5 * (u + u_new_2)
            gamma = metric.gamma(x2d, u_mid)
            gamma = metric.gamma(x2d, u_mid)
            first = np.einsum("n,ni->ni", -gamma, metric.d_i_alpha(x2d))
            second = np.einsum("nj,nij->ni", u_mid, metric.d_i_betaj(x2d))
            third = np.einsum("nj,nk,nijk->ni", u_mid, u_mid, metric.d_i_hjk(x2d))
            third = np.einsum("n,ni->ni", -metric.alpha(x2d) / (2 * gamma), third)
            u_new_2 = u + dt * (first + second + third)
            err = grBoris.error_u(metric, u_new_1, u_new_2, x)
            u_new_1 = np.copy(u_new_2)
        return (
            u_new_2,
            -(
                -metric.alpha(x2d) ** 2 * gamma / metric.alpha(x2d)
                + np.einsum("ij,ij->i", metric.betai(x2d), u)
            ),
            gamma,
        )

    def update_xi(
        self, 
        metric: Metric, 
        x: np.ndarray[float], 
        u_i: np.ndarray[float], 
        dt: float
    ) -> np.ndarray[float]:
        """
        Update the contravariant position of massive particles.
        x^(n) -> x^(n+1)

        Parameters
        ----------
        metric : Metric
        x : np.ndarray[float]
            Particle position at time n 
        u_i : np.ndarray[float]
            Particle 4-velocity at time n+1/2 
        dt : float
            Timestep

        Returns
        -------
        np.ndarray[float]
            Updated particle position
        """
        x_new_1 = np.copy(x)
        x_new_2 = np.copy(x)
        x2d = x[:, :-1]
        err = 1
        while err > 10e-8:
            xmid2d = 0.5 * (x2d + x_new_2[:, :-1])
            gamma = metric.gamma(xmid2d, u_i)
            gamma = metric.gamma(xmid2d, u_i)
            first = np.einsum("nij,nj->ni", metric.hij(xmid2d), u_i)
            first = np.einsum("n,ni->ni", metric.alpha(xmid2d) / gamma, first)
            second = -metric.betai(xmid2d)
            x_new_2 = x + dt * (first + second)
            err = grBoris.error_x(metric, x_new_1, x_new_2, (x_new_1 + x_new_2) / 2)
            x_new_1 = np.copy(x_new_2)

        return x_new_2

class grGCA:
    """
    Base class for the general relativistic GCA pusher.

    Methods
    -------
    error_x(metric, q1, q2, pos)
        Calculates the error in the position update.
    error_u(metric, q1, q2, pos)
        Calculates the error in the 4-velocity update.
    update_up_ih(up_ih, Dpih, dt)
        Updates the parallel 4-velocity due to electric field.
    update_up(metric, x, up, vD_i, bi, dt)
        Updates the parallel component of 4-velocity due to spacetime curvature.
    update_xi(metric, x, up, D_phys, B_phys, dt)
        Updates the contravariant position of massive particles.
    """

    def __init__(self): ...

    @staticmethod
    def error_x(
        metric: Metric, 
        q1: np.ndarray[float], 
        q2: np.ndarray[float], 
        pos: np.ndarray[float]
    ) -> float:
        x2d = pos[:, :-1]
        dq = np.abs(q2 - q1)
        return (
            np.sqrt(
                np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.h_ij(x2d), dq), dq)
            )[0]
            / np.sqrt(
                np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.h_ij(x2d), q1), q1)
            )[0]
        )

    def update_up_ih(
        self,
        up_ih: np.ndarray[float],
        Dpih: np.ndarray[float],
        dt: float,
    ) -> np.ndarray[float]:
        """
        Update the parallel 4-velocity due to electric field.

        Parameters
        ----------
        up_ih : np.ndarray[float]
            Parallel 4-velocity n tetrad basis
        Dpih : np.ndarray[float]
            Parallel electric field in tetrad basis
        dt : float
            Timestep

        Returns
        -------
        np.ndarray[float]
            Updated parallel 4-velocity in tetrad basis
        """
        q_ovr_m = 1
        return up_ih + q_ovr_m * Dpih * dt

    def update_up(
        self,
        metric: Metric,
        x: np.ndarray[float],
        up: np.ndarray[float],
        vD_i: np.ndarray[float],
        bi: np.ndarray[float],
        dt: float,
    ) -> np.ndarray[float]:
        """
        Update the parallel component of 4-velocity due to spacetime curvature.

        Parameters
        ----------
        metric : Metric
        x : np.ndarray[float]
            Particle position at time n in tetrad basis
        up : np.ndarray[float]
            Parallel parallel 4-velocity component
        vD_i : np.ndarray[float]
            DxB drift covariant velocity
        bi : np.ndarray[float]
            Magnetic field unit vector in tetrad basis
        dt : float
            Timestep

        Returns
        -------
        np.ndarray[float]
            Updated covariant 4-velocity
        """
        
        x2d = x[:, :-1]
        up_new_1 = np.copy(up)
        up_new_2 = np.copy(up)
        uper_i = 0
        err = 1
        while err > 10e-8:
            up_mid = 0.5 * (up + up_new_2)
            kappa = 1 / np.sqrt(
                1 - np.einsum("nij,ni,nj->n", metric.hij(x2d), vD_i, vD_i)
            )
            gamma = kappa * np.sqrt(1 + up_mid**2) #assuming mu = 0
            u_i_mid = (
                np.einsum("n,ni->ni", up_mid, metric.transform(bi, x2d, Idx.U, Idx.D))
                + np.einsum("ni,n->ni", vD_i, gamma)
                + uper_i
            )

            first = -np.einsum("n,ni->ni", gamma, metric.d_i_alpha(x2d))
            second = np.einsum("nj,nij->ni", u_i_mid, metric.d_i_betaj(x2d))
            third = np.einsum("nj,nk,nijk->ni", u_i_mid, u_i_mid, metric.d_i_hjk(x2d))
            second = np.einsum("nj,nij->ni", u_i_mid, metric.d_i_betaj(x2d))
            third = np.einsum("nj,nk,nijk->ni", u_i_mid, u_i_mid, metric.d_i_hjk(x2d))
            third = np.einsum("n,ni->ni", -metric.alpha(x2d) / (2 * gamma), third)
            up_new_2 = up + dt * np.einsum("ni,ni->n", (first + second + third), bi)

            err = 2 * (up_new_2 - up_new_1) / (up_new_2 + up_new_1)
            up_new_1 = np.copy(up_new_2)
        
        return up_new_2

    def update_xi(
        self,
        metric: Metric,
        x: np.ndarray[float],
        up: np.ndarray[float],
        D_phys,
        B_phys,
        dt: float,
    ) -> np.ndarray[float]:
        """
        Update the contravariant position of massive particles.
        x^(n) -> x^(n+1)

        Parameters
        ----------
        metric : Metric
        x : np.ndarray[float] (n x 3)
            Positions at time n
        up : np.ndarray[float] (n)
            Parallel 4-velocity component
        D_phys : function
            Electric field function in tetrad basis at time n
        B_phys : function
            Magnetic field function in tetrad basis at time n
        dt : float
            Timestep

        Returns
        -------
        np.ndarray[float] (n x 3)
            Updated particle position
        """
        x_new_1 = np.copy(x)
        x_new_2 = np.copy(x)
        n = x.shape[0]
        ep_ijk = Metric.eLC_ijk(n)
        err = 1
        while err > 10e-8:
            xmid = 0.5 * (x + x_new_2)
            xmid2d = xmid[:, :-1]
            Di = metric.transform(D_phys(x), xmid2d, Idx.T, Idx.U)
            Bi = metric.transform(B_phys(x), xmid2d, Idx.T, Idx.U)
            B_i = metric.transform(Bi, xmid2d, Idx.U, Idx.D)
            B_norm = np.sqrt(np.einsum("ni,ni->n", B_i, Bi))
            bi = Bi / B_norm
            vD_i = (
                np.sqrt(np.abs(np.linalg.det(metric.h_ij(xmid2d))))
                * np.einsum("nijk,nj,nk->ni", ep_ijk, Di, Bi)
                / B_norm ** 2
            )
            kappa = 1 / np.sqrt(
                1 - np.einsum("nij,ni,nj->n", metric.hij(xmid2d), vD_i, vD_i)
            )
            gamma = kappa * np.sqrt(1 + up ** 2)  # assuming mu = 0

            first = up * bi / gamma + np.einsum("nij,nj->ni", metric.hij(xmid2d), vD_i)
            first = np.einsum("n,ni->ni", metric.alpha(xmid2d), first)
            second = -metric.betai(xmid2d)
            x_new_2 = x + dt * (first + second)
            err = grBoris.error_x(metric, x_new_1, x_new_2, 0.5 * (x_new_1 + x_new_2))
            x_new_1 = np.copy(x_new_2)

        return x_new_2

def IntegrateTrajectory(
    metric: Metric,
    x0: np.ndarray[float],
    u0: np.ndarray[float],
    dt: float,
    tmax: float,
    D_func: Callable[[np.ndarray], np.ndarray] = lambda _: np.array([0, 0, 0]),
    B_func: Callable[[np.ndarray], np.ndarray] = lambda _: np.array([0, 0, 0]),
    drags: Dict[str, Dict[str, float | np.ndarray[float]]] = None,
    pusher_type: str = "boris",
    grgca_params: Dict[str, float] = {},
    progressbar=lambda _: _,
) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Integrate the trajectory of a particle in a spacetime metric.

    Parameters
    ----------
    metric : Metric
        Metric of the spacetime.
    x0 : np.ndarray[float]
        Initial position of the particle @ n.
    u0 : np.ndarray[float]
        Initial 4-velocity of the particle.
    dt : float
        Timestep.
    tmax : float
        Maximum time to integrate the trajectory.
    D_func : Callable[[np.ndarray], np.ndarray], optional
        Function that returns the electric field at a given position.
    B_func : Callable[[np.ndarray], np.ndarray], optional
        Function that returns the magnetic field at a given position.
    drags : Dict[str, Dict[str, float | np.ndarray[float]]], optional
        Dictionary of drag parameters.
    pusher_type : str, optional
        Type of pusher to use. "boris" or "gca" or "hybrid".
    grgca_params : Dict[str, float], optional
        Parameters for the GCA pusher.
    progressbar : Callable[[float], float], optional
        Progress bar function.

    Returns
    -------
    Tuple[np.ndarray[float], np.ndarray[float]]
        Trajectory and 4-velocity of the particle.
    """

    nsteps = int(tmax / dt)
    pusher_grboris = grBoris()
    if pusher_type == "gca":
        pusher_grgca = grGCA(**grgca_params)
    if pusher_type == "hybrid":
        E_ovr_B_max = grgca_params.pop("E_ovr_B_max", 0.9)
        larmor_max = grgca_params.pop("larmor_max", 0.1)
        pusher_grgca = grGCA(**grgca_params)
        pusher = ["Boris"]
    
    pusher_drags = []
    if drags is not None:
        for drag, params in drags.items():
            pusher_drags.append(Drag(drag, **params))

    xi, u_i = x0, u0
    t = 0
    times = [t]
    xs = [xi]
    us = [u_i]
    n = x0.shape[0]
    ep_ijk = Metric.eLC_ijk(n)
    for it in progressbar(range(nsteps)):
        D_phys, B_phys = D_func(xi), B_func(xi)
        use_grgca = False
        if pusher_type == "gca":
            u_iper = 0 #sets mu = 0
            use_grgca = True
        if pusher_type == "hybrid":
            u_iper = 0 #sets mu = 0
            u_ih = metric.transform(u_i, xi[:, :-1], Idx.D, Idx.T)
            rho = np.sum(np.cross(u_ih, B_phys) ** 2) ** 0.5 / np.dot(B_phys, B_phys)
            use_grgca = (np.linalg.norm(D_phys) / E_ovr_B_max < np.linalg.norm(B_phys)) and (
                rho < larmor_max
            )
        
        if use_grgca:
            x2d = xi[:, :-1]
            Di = metric.transform(D_phys, xi[:, :-1], Idx.T, Idx.U)
            Bi = metric.transform(B_phys, xi[:, :-1], Idx.T, Idx.U)
            B_i = metric.transform(Bi, xi[:, :-1], Idx.U, Idx.D)
            B_norm = np.sqrt(np.einsum("ni,ni->n", Bi, B_i))
            bi = Bi / B_norm
            b_i = B_i / B_norm
            u_ih = metric.transform(u_i, xi[:, :-1], Idx.D, Idx.T)

            up_i = np.einsum("nj,nj,ni->ni", u_i, bi, b_i)
            Dpi = np.einsum("nj,nj,ni->ni", Di, b_i, bi)
            up_ih = metric.transform(up_i, xi[:, :-1], Idx.D, Idx.T)
            Dpih = metric.transform(Dpi, xi[:, :-1], Idx.U, Idx.T)
            up_ih2 = pusher_grgca.update_up_ih(up_ih, Dpih, dt / 2)
            up = np.einsum(
                "ni,ni->n", metric.transform(up_ih2, xi[:, :-1], Idx.T, Idx.D), bi
            )
            vD_i = (
                np.sqrt(np.abs(np.linalg.det(metric.h_ij(x2d))))
                * np.einsum("nijk,nj,nk->ni", ep_ijk, Di, Bi)
                / B_norm ** 2
            )
            
            kappa = 1 / np.sqrt(
                1 - np.einsum("nij,ni,nj->n", metric.hij(xi[:, :-1]), vD_i, vD_i)
            )
            gamma = kappa * np.sqrt(1 + up ** 2)
            u_ih2 = (
                metric.transform(up_ih2, xi[:, :-1], Idx.T, Idx.D)
                + np.einsum("ni,n->ni", vD_i, gamma)
                + u_iper
            )
        else:
            x2d = xi[:, :-1]
            u_ih = metric.transform(u_i, x2d, Idx.D, Idx.T)
            Dh = D_phys
            Bh = B_phys
            u_ih2 = pusher_grboris.update_u(u_ih, Dh, Bh, dt / 2)
        
        if drags is not None:
            for drag in pusher_drags:
                args = {}
                if drag.process == "sync":
                    args["E"] = D_phys
                    args["B"] = B_phys
                if use_grgca and drag.process == "sync":
                    continue        
                u_ih = drag.update_u(u_ih, u_ih2, dt, **args)
        else:
            u_ih = u_ih2
            
        if use_grgca:
            up = pusher_grgca.update_up(metric, xi, up, vD_i, bi, dt)
            up_i = np.einsum("n,ni->ni", up, b_i)
            up_ih = metric.transform(up_i, x2d, Idx.D, Idx.T)
            up_ih2 = pusher_grgca.update_up_ih(up_ih, Dpih, dt / 2)
            up = np.einsum(
                "ni,ni->n", metric.transform(u_ih, xi[:, :-1], Idx.T, Idx.D), bi
            )
            up2 = np.einsum(
                "ni,ni->n", metric.transform(u_ih2, xi[:, :-1], Idx.T, Idx.D), bi
            )
            gamma = kappa * np.sqrt(1 + up ** 2)
            gamma2 = kappa * np.sqrt(1 + up2 ** 2)
            u_i = (
                metric.transform(up_ih, xi[:, :-1], Idx.T, Idx.D)
                + np.einsum("ni,n->ni", vD_i, gamma)
                + u_iper
            )
            u_i2 = (
                metric.transform(up_ih2, xi[:, :-1], Idx.T, Idx.D)
                + np.einsum("ni,n->ni", vD_i, gamma2)
                + u_iper
            )
            u_ih = metric.transform(u_i, x2d, Idx.D, Idx.T)
            u_ih2 = metric.transform(u_i2, x2d, Idx.D, Idx.T)

        else:
            u_i = metric.transform(u_ih, x2d, Idx.T, Idx.D)
            u_i, energy, gamma = pusher_grboris.update_u_i(metric, xi, u_i, dt)
            u_ih = metric.transform(u_i, x2d, Idx.D, Idx.T)
            u_ih2 = pusher_grboris.update_u(u_ih, Dh, Bh, dt / 2)
        
        if drags is not None:
            for drag in pusher_drags:
                if use_grgca and drag.process == "sync":
                    continue        
                u_ih = drag.update_u(u_ih, u_ih2, dt, **args)
        else:
            u_ih = u_ih2
        
        if use_grgca:
            u_i = metric.transform(u_ih, x2d, Idx.T, Idx.D)
            up = np.einsum("ni,ni->n", u_i, bi)
            xi = pusher_grgca.update_xi(metric, xi, up, D_func, B_func, dt)
        else:
            u_i = metric.transform(u_ih, x2d, Idx.T, Idx.D)
            xi = pusher_grboris.update_xi(metric, xi, u_i, dt)

        t += dt
        times.append(t)
        xs.append(xi)
        us.append(u_i)
        if pusher_type == "hybrid":
            pusher.append("GCA" if use_grgca else "Boris")
    if pusher_type == "hybrid":
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
    
            

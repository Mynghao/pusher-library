import numpy as np
from typing import Dict, Tuple, Callable
from metrics import Metric, Idx

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
    def __init__(self): ...

    def error_x(self, metric, q1, q2, pos):
        x2d = pos[:, :-1]
        dq = q2 - q1
        return (
            np.sqrt(
                np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.h_ij(x2d), dq), dq)
            )[0]
            / np.sqrt(
                np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.h_ij(x2d), q1), q1)
            )[0]
        )

    def error_u(self, metric, q1, q2, pos):
        x2d = pos[:, :-1]
        dq = q2 - q1
        return (
            np.sqrt(
                np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.hij(x2d), dq), dq)
            )[0]
            / np.sqrt(
                np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.hij(x2d), q1), q1)
            )[0]
        )

    def update_u(self, x, u, E, B, dt):

        k = 0.5 * dt

        u_neg = u + k * E
        t = k * B / np.sqrt(1 + (np.linalg.norm(u_neg)) ** 2)
        s = 2 * t / (1 + np.linalg.norm(t) ** 2)
        u_pos = u_neg + np.cross((u_neg + np.cross(u_neg, t)), s)

        return u_pos + k * E

    def update_u_i(self, metric, x, u, dt):
        u_new_1 = np.copy(u)
        u_new_2 = np.copy(u)
        x2d = x[:, :-1]
        err = 1

        while err > 10e-8:
            u_mid = 0.5 * (u + u_new_2)
            gamma = metric.gamma(x2d, u_mid)
            first = np.einsum("n,ni->ni", -gamma, metric.d_i_alpha(x2d))
            second = np.einsum("nj,nij->ni", u_mid, metric.d_i_betaj(x2d))
            third = np.einsum("nj,nk,nijk->ni", u_mid, u_mid, metric.d_i_hjk(x2d))
            third = np.einsum("n,ni->ni", -metric.alpha(x2d) / (2 * gamma), third)
            u_new_2 = u + dt * (first + second + third)
            err = self.error_u(metric, u_new_1, u_new_2, x)
            u_new_1 = np.copy(u_new_2)

        return (
            u_new_2,
            -(
                -metric.alpha(x2d) ** 2 * gamma / metric.alpha(x2d)
                + np.einsum("ij,ij->i", metric.betai(x2d), u)
            ),
            gamma,
        )

    def update_xi(self, metric, x, u_i, dt):
        x_new_1 = np.copy(x)
        x_new_2 = np.copy(x)
        x2d = x[:, :-1]
        err = 1
        while err > 10e-8:
            xmid2d = 0.5 * (x2d + x_new_2[:, :-1])
            gamma = metric.gamma(xmid2d, u_i)
            first = np.einsum("nij,nj->ni", metric.hij(xmid2d), u_i)
            first = np.einsum("n,ni->ni", metric.alpha(xmid2d) / gamma, first)
            second = -metric.betai(xmid2d)
            x_new_2 = x + dt * (first + second)
            err = self.error_x(metric, x_new_1, x_new_2, (x_new_1 + x_new_2) / 2)
            x_new_1 = np.copy(x_new_2)

        return x_new_2

    def push(self, metric, xold, x, u_i, E_phys, B_phys, dt):
        assert x.shape == u_i.shape
        assert len(x.shape) == 2
        assert x.shape[1:] == (3,)
        x2d = x[:, :-1]
        xmid2d = 0.5 * (xold[:, :-1] + x[:, :-1])
        n = x.shape[0]

        Ei = metric.transform(E_phys(x), x2d, Idx.T, Idx.U)
        Bi = metric.transform(B_phys(x), x2d, Idx.T, Idx.U)
        E_i = metric.transform(Ei, x2d, Idx.U, Idx.D)
        B_i = metric.transform(Bi, x2d, Idx.U, Idx.D)
        B_norm = np.sqrt(np.einsum("ni,ni->n", Bi, B_i))
        E_norm = np.sqrt(np.einsum("ni,ni->n", Ei, E_i))
        # print(B_norm, E_norm)
#         if B_norm[0] == 0 and E_norm[0] == 0:
#             u_new, energy, gamma = self.update_u_i(metric, x, u_i, dt)
#             uh_new = metric.transform(u_new, xmid2d, Idx.D, Idx.T)
#             x_new, u = self.update_xi(metric, x, uh_new, dt)
#             return x, x_new, u_new, energy, gamma

        Eh = metric.transform(Ei, x2d, Idx.U, Idx.T)
        Bh = metric.transform(Bi, x2d, Idx.U, Idx.T)
        uh = metric.transform(u_i, x2d, Idx.D, Idx.T)

        uh_1 = self.update_u(x, uh, Eh, Bh, dt / 2)
        u_i_1 = metric.transform(uh_1, x2d, Idx.T, Idx.D)
        u_i_2, energy, gamma = self.update_u_i(metric, x, u_i_1, dt)
        uh_2 = metric.transform(u_i_2, x2d, Idx.D, Idx.T)
        uh_2 = self.update_u(x, uh_2, Eh, Bh, dt / 2)
        u_new = metric.transform(uh_2, x2d, Idx.T, Idx.D)
        x_new = self.update_xi(metric, x, uh_2, dt)

        return x, x_new, u_new, energy, gamma


class grGCA:

    def __init__(self): ...

    def error_x(self, metric, q1, q2, pos):
        x2d = pos[:, :-1]
        dq = q2 - q1
        return (
            np.sqrt(
                np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.h_ij(x2d), dq), dq)
            )[0]
            / np.sqrt(
                np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.h_ij(x2d), q1), q1)
            )[0]
        )

    def update_u_i(self, metric, x, u_i, D_phys, B_phys, dt):
        """
        Update the covariant velocity of massive particles.
        u_i^(n-1/2) -> u_i^(n+1/2)

        Parameters
        ----------
        metric : Metric
        x : np.array (n x D)
            Positions at time n
        u_i : np.array (n x 3)
            Covariant velocities at time n-1/2
        D_phys : np.array (n x 3)
            Electric field in tetrad basis at time n
        B_phys : np.array (n x 3)
            Magnetic field in tetrad basis at time n
        dt : float
            Timestep
        q_ovr_m : float, optional
            Charge-to-mass ratio (default is 1)

        Returns
        -------
        tuple : (np.array (n x 3), np.array (n))
            (u_i^(n+1/2), gamma)
        """
        n = x.shape[0]  # number of particles
        x2d = x[:, :-1]
        #xmid2d = 0.5 * (xold[:, :-1] + x[:, :-1])
        # u_i^(n-1/2) -> u^i^(n-1/2)
        ui = metric.transform(u_i, x2d, Idx.D, Idx.U)

        # D^ih^(n) & B^ih^(n) -> D^i^(n) & B^i^(n)
        Di = metric.transform(D_phys(x), x2d, Idx.T, Idx.U)
        Bi = metric.transform(B_phys(x), x2d, Idx.T, Idx.U)

        # D^i^(n) & B^i^(n) -> D_i^(n) & B_i^(n)
        D_i = metric.transform(Di, x2d, Idx.U, Idx.D)
        B_i = metric.transform(Bi, x2d, Idx.U, Idx.D)

        # |B|^(n)
        B_norm = np.sqrt(np.einsum("ni,ni->n", B_i, Bi))
        # b^i^(n) & b_i^(n)
        b_i = B_i / B_norm
        bi = Bi / B_norm

        # u_i^(n-1/2) & D_i^(n) -> u||^i^(n-1/2) & D||^i^(n) aligned with b^i^(n)
        u_ip = np.einsum("nj,nj,ni->ni", u_i, bi, b_i)
        Dip = np.einsum("nj,nj,ni->ni", Di, b_i, bi)

        # Levi-Civita psuedotensor
        ep_ijk = Metric.eLC_ijk(n)

        # (D x B)_i^(n) drift velocity
        vE_i = (
            np.sqrt(np.abs(np.linalg.det(metric.h_ij(x2d))))
            * np.einsum("nijk,nj,nk->ni", ep_ijk, Di, Bi)
            / B_norm ** 2
        )

        # Setting mu = 0
        u_iper = 0

        # u||^i^(n-1/2) & D||^i^(n) -> u||^ih^(n-1/2) & D||^ih^(n)
        u_ihp = metric.transform(u_ip, x2d, Idx.D, Idx.T)
        Dihp = metric.transform(Dip, x2d, Idx.U, Idx.T)

        # u||^ih^(n-1/2) -> u||^ih^(*) (electric push)
        q_ovr_m = 1
        u_ihp_2 = u_ihp + q_ovr_m * Dihp * dt / 2

        # u||^ih^(*) -> u||^i^(*)
        u_ip_2 = metric.transform(u_ihp_2, x2d, Idx.T, Idx.D)

        # u||^i^(*) -> u||^(*) (computing the projection)
        up = np.einsum("ni,ni->n", u_ip_2, bi)
        up_new_1 = np.copy(up)
        up_new_2 = np.copy(up)

        err = 1

        # u^(n-1/2) -> u^(n+1/2)
        # all right-hand side components defined at timestep n
        while err > 10e-8:
            # evaluate midpoint scalar value
            # u||^(mid) = (u||^(*) + u||^(**)) / 2
            up_mid = 0.5 * (up + up_new_2)

            # kappa = 1 / sqrt(1 - |vE|^2) @ n
            kappa = 1 / np.sqrt(
                1 - np.einsum("nij,ni,nj->n", metric.hij(x2d), vE_i, vE_i)
            )
            gamma = kappa * np.sqrt(1 + up_mid**2)  # assuming mu = 0

            # midpoint full velocity vector (u_iper = perpenducular component = 0)
            # u_i^(mid) = u||^(mid) * b_i + gamma * vE_i + (u_iper = 0)
            u_i_mid = (
                np.einsum("n,ni->ni", up_mid, b_i)
                + np.einsum("ni,n->ni", vE_i, gamma)
                + u_iper
            )

            # implicit push equation components
            first = -np.einsum("n,ni->ni", gamma, metric.d_i_alpha(x2d))
            second = np.einsum("nj,nij->ni", u_i_mid, metric.d_i_betaj(x2d))
            third = np.einsum("nj,nk,nijk->ni", u_i_mid, u_i_mid, metric.d_i_hjk(x2d))
            third = np.einsum("n,ni->ni", -metric.alpha(x2d) / (2 * gamma), third)

            # u||^(**) = u||^(*) + dt * (first + second + third)
            up_new_2 = up + dt * np.einsum("ni,ni->n", (first + second + third), bi)

            # error calculation
            err = 2 * (up_new_2 - up_new_1) / (up_new_2 + up_new_1)
            up_new_1 = np.copy(up_new_2)

        # reconstruct new velocity
        #         u_new = (
        #             np.einsum("n,ni->ni", ups_new_2, b_i)
        #             + np.einsum("n,ni->ni", gamma, vE_i)
        #             + u_iper
        #         )

        #         u2h = metric.transform(u2_i, xmid2d, Idx.U, Idx.T)
        #         uh_new = uihp + q_ovr_m * Dihp * dt / 2
        up_ih = metric.transform(
            np.einsum("n,ni->ni", up_new_2, b_i), x2d, Idx.D, Idx.T
        )
        up_ih_new = up_ih + q_ovr_m * Dihp * dt / 2
        up_i_new = metric.transform(up_ih_new, x2d, Idx.T, Idx.D)
        up_new = np.einsum('ni,ni->n', up_i_new, bi)
        u_new = (
                    up_i_new
                    + np.einsum("n,ni->ni", gamma, vE_i)
                    + u_iper
                )

        return (
            u_new,
            up_new,
            kappa,
            gamma,
        )

    def update_xi(self, metric, x, u_i, ups, D_phys, B_phys, dt):
        """
        Update the contravariant position of massive particles.
        x^(n) -> x^(n+1)

        Parameters
        ----------
        metric : Metric
        x : np.array (n x D)
            Positions at time n
        u_i : np.array (n x 3)
            Covariant velocities at time n+1/2
        D_phys : np.array (n x 3)
            Electric field in tetrad basis at time n
        B_phys : np.array (n x 3)
            Magnetic field in tetrad basis at time n
        dt : float
            Timestep

        Returns
        -------
        np.array (n x D)
            x^(n+1)
        """
        n = x.shape[0]  # number of particles

        x_new_1 = np.copy(x)
        x_new_2 = np.copy(x)
        err = 1

        # Levi-Civita psuedotensor
        ep_ijk = Metric.eLC_ijk(n)

        #         uh = metric.transform(u_i, xmid2d, Idx.D, Idx.T)
        #         uh_new = uihp + q_ovr_m * Dihp * dt / 2
                        
        # x^i^(n) -> x^i^(n+1)
        # all right-hand side components defined at timestp n+1/2
        while err > 10 ** (-8):
            # x^i^(mid) = (x^i^(n) + x^i^(*)) / 2
            xmid = 0.5 * (x + x_new_2)
            xmid2d = xmid[:, :-1]

            # Lorentz factor
            gamma = metric.gamma(xmid2d, u_i)

            # D^ih^(n) & B^ih^(n) -> E^i^(n) & B^i^(n)
            Di = metric.transform(D_phys(xmid2d), xmid2d, Idx.T, Idx.U)
            Bi = metric.transform(B_phys(xmid2d), xmid2d, Idx.T, Idx.U)
            # D^i^(n) & B^i^(n) -> E_i^(n) & B_i^(n)
            D_i = metric.transform(Di, xmid2d, Idx.U, Idx.D)
            B_i = metric.transform(Bi, xmid2d, Idx.U, Idx.D)
            # @TODO: THESE ARE FIELDS AT ^(n) not ^(n+1/2)

            # u_i^(n+1/2) -> u^i^(n+1/2)
            # ui = metric.transform(u_i, xmid2d, Idx.D, Idx.U)

            # |B|^(n)
            B_norm = np.sqrt(np.einsum("ni,ni->n", B_i, Bi))
            # b^i^(n) & b_i^(n)
            b_i = B_i / B_norm
            bi = Bi / B_norm

            # u_i^(n+1/2) & D_i^(n) -> u||^i^(n+1/2) & D||^i^(n) aligned with b^i^(n)
            

            # (D x B)_i^(n) drift velocity
            vE_i = (
                np.sqrt(np.abs(np.linalg.det(metric.h_ij(xmid2d))))
                * np.einsum("nijk,nj,nk->ni", ep_ijk, Di, Bi)
                / B_norm**2
            )
                            
            # implicit push equation components
            first = ups * bi / gamma + np.einsum("nij,nj->ni", metric.hij(xmid2d), vE_i)
            first = np.einsum("n,ni->ni", metric.alpha(xmid2d), first)
            second = -metric.betai(xmid2d)

            # x^i^(*) = x^i^(n) + dt * (first + second)
            x_new_2 = x + dt * (first + second)

            # error calculation
            err = self.error_x(metric, x_new_1, x_new_2, 0.5 * (x_new_1 + x_new_2))
            # x^i^(**) = x^i^(*)
            x_new_1 = np.copy(x_new_2)

        return x_new_2

    def push(self, metric, x, u_i, D_phys, B_phys, dt, q_ovr_m=1):
        """
        Parameters
        ----------
        metric : function
        x : np.array (n x 3)
            contravariant coordinates (at n)
        u_i: np.array (n x 3)
            covariant velocities (at n-1/2)
        D_phys: np.array (n x 3)
            electric field in tetrad basis (at n)
        B_phys: np.array (n x 3)
            magnetic field in tetrad basis (at n)

        Returns
        -------
        x_new : np.array (n x 3)
            updated contravariant coordinates over 1 timestep
        u_new : np.array (n x 3)
            updated covariant velocity over 1 timestep
        energy : np.array (n)
            total particle energy
        gamma : np.array (n)
            particle Lorentz factor
        """

        assert x.shape == u_i.shape
        assert len(x.shape) == 2
        assert x.shape[1:] == (2,) or x.shape[1:] == (3,)

        u_new, ups, kappa, gamma = self.update_u_i(metric, x, u_i, D_phys, B_phys, dt)
        x_new = self.update_xi(metric, x, u_new, ups, D_phys, B_phys, dt)

        return x_new, u_new, kappa, gamma

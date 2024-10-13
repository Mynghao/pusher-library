import numpy as np
from metrics import Metric, Idx

class grBoris:

    def __init__(self): ...
        
    def error_x(self, metric, q1, q2, pos):
        x2d = pos[:, :-1]
        dq = q2 - q1
        return np.sqrt(np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.h_ij(x2d), dq), dq))[0] / np.sqrt(np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.h_ij(x2d), q1), q1))[0]
    
    def error_u(self, metric, q1, q2, pos):
        x2d = pos[:, :-1]
        dq = q2 - q1
        return np.sqrt(np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.hij(x2d), dq), dq))[0] / np.sqrt(np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.hij(x2d), q1), q1))[0]
    
    def update_u(self, x, u, E, B, dt):
        
        k = 0.5 * dt
        
        u_neg = u + k * E
        t = k * B / np.sqrt(1 + (np.linalg.norm(u_neg)) ** 2)
        s = 2 * t / (1 + np.linalg.norm(t) ** 2)
        u_pos = u_neg + np.cross((u_neg + np.cross(u_neg, t)), s)
        
        return u_pos + k * E
    
    def update_u_i(self, metric, x, u, phot, dt):
        u_new_1 = np.copy(u)
        u_new_2 = np.copy(u)
        x2d = x[:, :-1]
        err = 1
        
        while err > 10e-8:
            u_mid = 0.5 * (u + u_new_2)
            gamma = metric.gamma(x2d, u_mid, phot)
            first = np.einsum("n,ni->ni", -gamma, metric.d_i_alpha(x2d))
            second = np.einsum("nj,nij->ni", u_mid, metric.d_i_betaj(x2d))
            third = np.einsum("nj,nk,nijk->ni", u_mid, u_mid, metric.d_i_hjk(x2d))
            third = np.einsum("n,ni->ni", -metric.alpha(x2d) / (2 * gamma), third)
            u_new_2 = u + dt * (first + second + third)
            err = self.error_u(metric, u_new_1, u_new_2, x)
            u_new_1 = np.copy(u_new_2)
            
        return u_new_2, - (- metric.alpha(x2d)**2 * gamma / metric.alpha(x2d) + np.einsum('ij,ij->i', metric.betai(x2d), u)), gamma
    
    
    def update_xi(self, metric, x, uh, phot, dt):
        x_new_1 = np.copy(x)
        x_new_2 = np.copy(x)
        x2d = x[:, :-1]
        err = 1
        while err > 10e-8:
            xmid2d = 0.5 * (x2d + x_new_2[:, :-1])
            u_i = metric.transform(uh, xmid2d, Idx.T, Idx.D)
            gamma = metric.gamma(xmid2d, u_i, phot)
            first = np.einsum("nij,nj->ni", metric.hij(xmid2d), u_i)
            first = np.einsum("n,ni->ni", metric.alpha(xmid2d) / gamma, first)
            second = - metric.betai(xmid2d)
            x_new_2 = x + dt * (first + second)
            err = self.error_x(metric, x_new_1, x_new_2, (x_new_1 + x_new_2) / 2)
            x_new_1 = np.copy(x_new_2)
            
        return x_new_2, u_i
        
    
    def grBorisPush(self, metric, xold, x, u_i, E_phys, B_phys, dt, phot = False):
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
        B_norm = np.sqrt(np.einsum('ni,ni->n', Bi, B_i))
        E_norm = np.sqrt(np.einsum('ni,ni->n', Ei, E_i))
        #print(B_norm, E_norm)
        if B_norm[0] == 0 and E_norm[0] == 0:
            u_new, energy, gamma = self.update_u_i(metric, x, u_i, phot, dt)
            uh_new = metric.transform(u_new, xmid2d, Idx.D, Idx.T)
            x_new, u = self.update_xi(metric, x, uh_new, phot, dt)
            return x, x_new, u_new, energy, gamma
    
        bi = Bi / B_norm
        Eh = metric.transform(Ei, x2d, Idx.U, Idx.T)
        Bh = metric.transform(Bi, x2d, Idx.U, Idx.T)
        uh = metric.transform(u_i, xmid2d, Idx.D, Idx.T)
        
        uh_1 = self.update_u(x, uh, Eh, Bh, dt / 2)
        u_i_1 = metric.transform(uh_1, xmid2d, Idx.T, Idx.D)
        u_i_2, energy, gamma = self.update_u_i(metric, x, u_i_1, phot, dt)
        uh_2 = metric.transform(u_i_2, xmid2d, Idx.D, Idx.T)
        uh_3 = self.update_u(x, uh_2, Eh, Bh, dt/2)
        x_new, u_new = self.update_xi(metric, x, uh_3, phot, dt)
        
        return x, x_new, u_new, energy, gamma

class grGCA:
    
    def __init__(self): ...
        
    def error_x(self, metric, q1, q2, pos):
        x2d = pos[:, :-1]
        dq = q2 - q1
        return np.sqrt(np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.h_ij(x2d), dq), dq))[0] / np.sqrt(np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.h_ij(x2d), q1), q1))[0]
    
    def error_u(self, metric, q1, q2, pos):
        x2d = pos[:, :-1]
        dq = q2 - q1
        return np.sqrt(np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.hij(x2d), dq), dq))[0] / np.sqrt(np.einsum("ni,ni->n", np.einsum("nij,ni->nj", metric.hij(x2d), q1), q1))[0]
    
    
    def update_u_i(self, metric, xold, x, u_i, D_phys, B_phys, dt):
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
        xmid2d = 0.5 * (xold[:, :-1] + x[:, :-1])
        # u_i^(n-1/2) -> u^i^(n-1/2)
        ui = metric.transform(u_i, xmid2d, Idx.D, Idx.U)
    
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
        uip = np.einsum("nj,ni,nj->ni", u_i, bi, bi)
        Dip = np.einsum("nj,ni,nj->ni", D_i, bi, bi)
    
        # Levi-Civita psuedotensor
        ep_ijk = Metric.eLC_ijk(n)
    
        # (D x B)_i^(n) drift velocity
        vE_i = - (
            np.sqrt(np.abs(np.linalg.det(metric.h_ij(x2d))))
            * np.einsum("nijk,nj,nk->ni", ep_ijk, Di, Bi)
            / B_norm ** 2
        )
    
        # Setting mu = 0
        u_iper = 0
    
        # u||^i^(n-1/2) & D||^i^(n) -> u||^ih^(n-1/2) & D||^ih^(n)
        uihp = metric.transform(uip, xmid2d, Idx.U, Idx.T)
        Dihp = metric.transform(Dip, x2d, Idx.U, Idx.T)
    
        # u||^ih^(n-1/2) -> u||^ih^(*) (electric push)
        q_ovr_m = 1
        uihp_2 = uihp + q_ovr_m * Dihp * dt / 2
    
        # u||^ih^(*) -> u||^i^(*)
        uip_2 = metric.transform(uihp_2, xmid2d, Idx.T, Idx.U)
    
        # u||^i^(*) -> u||^(*) (computing the projection)
        ups = np.einsum("ni,nj->n", uip_2, b_i)
        ups_new_1 = np.copy(ups)
        ups_new_2 = np.copy(ups)
    
        err = 1
    
        # u^(n-1/2) -> u^(n+1/2)
        # all right-hand side components defined at timestep n
        while err > 10e-8:
            # evaluate midpoint scalar value
            # u||^(mid) = (u||^(*) + u||^(**)) / 2
            ups_mid = 0.5 * (ups + ups_new_2)
    
            # kappa = 1 / sqrt(1 - |vE|^2) @ n
            kappa = 1 / np.sqrt(1 - np.einsum("nij,ni,nj->n", metric.hij(x2d), vE_i, vE_i))
            gamma = kappa * np.sqrt(1 + ups_mid**2) #assuming mu = 0
    
            # midpoint full velocity vector (u_iper = perpenducular component = 0)
            # u_i^(mid) = u||^(mid) * b_i + gamma * vE_i + (u_iper = 0)
            u_mid = (
                np.einsum("n,ni->ni", ups_mid, b_i)
                + np.einsum("ni,n->ni", vE_i, gamma)
                + u_iper
            )
    
            # implicit push equation components
            first = -np.einsum("n,ni->ni", gamma, metric.d_i_alpha(x2d))
            second = np.einsum("nj,nij->ni", u_mid, metric.d_i_betaj(x2d))
            third = np.einsum("nj,nk,nijk->ni", u_mid, u_mid, metric.d_i_hjk(x2d))
            third = np.einsum("n,ni->ni", -metric.alpha(x2d) / (2 * gamma), third)
    
            # u||^(**) = u||^(*) + dt * (first + second + third)
            ups_new_2 = ups + dt * np.einsum("ni,ni->n", (first + second + third), bi)
    
            # error calculation
            err = 2 * (ups_new_2 - ups_new_1) / (ups_new_2 + ups_new_1)
            ups_new_1 = np.copy(ups_new_2)
    
        # reconstruct new velocity
#         u_new = (
#             np.einsum("n,ni->ni", ups_new_2, b_i)
#             + np.einsum("n,ni->ni", gamma, vE_i)
#             + u_iper
#         )
        
#         u2h = metric.transform(u2_i, xmid2d, Idx.U, Idx.T)
#         uh_new = uihp + q_ovr_m * Dihp * dt / 2
        up2h = metric.transform(np.einsum("n,ni->ni", ups_new_2, b_i), xmid2d, Idx.D, Idx.T)
        up2h_new = up2h + q_ovr_m * Dihp * dt / 2
        
        #u_new = (
#             np.einsum("n,ni->ni", ups_new_2, b_i)
#             + np.einsum("n,ni->ni", gamma, vE_i)
#             + u_iper
#         )
            
        return (
            up2h_new,
            gamma,
        )
    
    
    def update_xi(self, metric, x, uph, D_phys, B_phys, dt):
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
        phot : bool
            True: massless (e.g. photon), False: massive (e.g. proton)
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
            gamma = metric.gamma(xmid2d, u_i, 0)
    
            # D^ih^(n) & B^ih^(n) -> E^i^(n) & B^i^(n)
            Di = metric.transform(D_phys(xmid2d), xmid2d, Idx.T, Idx.U)
            Bi = metric.transform(B_phys(xmid2d), xmid2d, Idx.T, Idx.U)
            # D^i^(n) & B^i^(n) -> E_i^(n) & B_i^(n)
            D_i = metric.transform(Di, xmid2d, Idx.U, Idx.D)
            B_i = metric.transform(Bi, xmid2d, Idx.U, Idx.D)
            # @TODO: THESE ARE FIELDS AT ^(n) not ^(n+1/2)
    
            # u_i^(n+1/2) -> u^i^(n+1/2)
            #ui = metric.transform(u_i, xmid2d, Idx.D, Idx.U)
    
            # |B|^(n)
            B_norm = np.sqrt(np.einsum("ni,ni->n", B_i, Bi))
            # b^i^(n) & b_i^(n)
            b_i = B_i / B_norm
            bi = Bi / B_norm
    
            # u_i^(n+1/2) & D_i^(n) -> u||^i^(n+1/2) & D||^i^(n) aligned with b^i^(n)
            uip = np.einsum("nj,ni,nj->ni", u_i, bi, bi)
            Dip = np.einsum("nj,ni,nj->ni", D_i, bi, bi)
    
            # scalar parallel velocity
            ups = np.einsum("ni,nj->n", uip, b_i)
    
            # (D x B)_i^(n) drift velocity
            vE_i = - (
                np.sqrt(np.abs(np.linalg.det(metric.h_ij(xmid2d))))
                * np.einsum("nijk,nj,nk->ni", ep_ijk, Di, Bi)
                / B_norm**2
            )
            #print(Di, D_i)
            #print(Bi, B_i)
            #print(vE_i)
            # implicit push equation components
            up_i = metric.transform(uph, xmid2d, Idx.T, Idx.D)
            ups = np.einsum("ni,nj->n", up_i, b_i)
            first = ups * np.einsum("nij,nj->ni", metric.hij(xmid2d), b_i) / gamma + np.einsum(
                "nij,nj->ni", metric.hij(xmid2d), vE_i
            )
            first = np.einsum("n,ni->ni", metric.alpha(xmid2d), first)
            second = -metric.betai(xmid2d)
    
            # x^i^(*) = x^i^(n) + dt * (first + second)
            x_new_2 = x + dt * (first + second)
    
            # error calculation
            err = error(metric, x_new_1, x_new_2, 0.5 * (x_new_1 + x_new_2))
            # x^i^(**) = x^i^(*)
            x_new_1 = np.copy(x_new_2)
        
        u_iper = 0
        kappa = 1 / np.sqrt(1 - np.einsum("nij,ni,nj->n", metric.hij(xmid2d), vE_i, vE_i))
        gamma = kappa * np.sqrt(1 + ups ** 2)
        u_new = (
             np.einsum("n,ni->ni", ups, b_i)
             + np.einsum("n,ni->ni", gamma, vE_i)
             + u_iper
         )
        return x_new_2, u_new
    
    
    def grGCAPush(self, metric, xold, x, u_i, D_phys, B_phys, dt, q_ovr_m=1):
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
    
        uh_new, gamma = self.update_u_i(metric, xold, x, u_i, D_phys, B_phys, dt)
        x_new, u_new = self.update_xi(metric, x, uh_new, D_phys, B_phys, dt)
    
        return x, x_new, u_new, gamma
    
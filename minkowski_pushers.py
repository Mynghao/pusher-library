#This code assumes q/m = c = 1
import numpy as np

class Boris:
    def __init__(self, drag, had_scheme, field_params):
        assert drag in ['none', 'sync', 'pp', 'pg', 'bth']
        assert had_scheme in ['none', 'cont', 'prob'], 'Invalid scheme type. Must be \'none\', \'cont\', or \'prob\''
        self.drag_type = drag
        self.had_scheme_type = had_scheme
        self.field_params = field_params
        
    def update_u(self, x, u, E, B, dt):
        
        k = 0.5 * dt
        
        u_neg = u + k * E
        t = k * B / np.sqrt(1 + (np.linalg.norm(u_neg)) ** 2)
        s = 2 * t / (1 + np.linalg.norm(t) ** 2)
        u_pos = u_neg + np.cross((u_neg + np.cross(u_neg, t)), s)
        
        return u_pos + k * E
    
    def update_x(self, x, u, dt):
        
        return x + u * dt / np.sqrt(1 + np.linalg.norm(u) ** 2)
    
    def cross_section(self, x, had_interaction_type):
        coefs_pp_p = [-0.1341, 6.602, -1.404, 0.1783, 0.004166]
        coefs_pp_q = [-0.3566, 7.135, -2.043, 0.2875, 0.004166]
        s_pp = 1.693
        inel_pp = 0.17
        
        coefs_pg_p = [0.5046, -4.889, 8.251, 17.81, 10.45, 2.459, 0.2217]
        coefs_pg_q = [0.469, -5.062, -309.1, 21.06, 13.78, 3.634, 0.3646]
        s_pg = -1.155
        inel_pg = 0.2
        
        coefs_bth_p = [-0.007843, -0.035, 0.06724, 0.3862, -1.601, -1.556, -0.2164, 3.6, -0.3493, -4.843, 11.42, -20.23, 15.48]
        coefs_bth_q = [1]
        s_bth = 1e5
        
        def func(x, s, coefs_p, coefs_q):
            return s * np.polyval(coefs_p, x) / np.polyval(coefs_q, x)
        
        if had_interaction_type == 'pp':
            return 10 ** func(np.log10(x), s_pp, coefs_pp_p, coefs_pp_q)
        elif had_interaction_type == 'pg':
            return 10 ** func(np.log10(x), s_pg, coefs_pg_p, coefs_pg_q)
        elif had_interaction_type == 'bth':
            return func(np.log10(x), s_bth, coefs_bth_p, coef_bth_q)
    
    def sync_update_u(self, x, u, u_boris, E, B, dt):
        B0 = 1
        rec = 0.1
        gam_rad = np.sqrt(rec)
        e = E / B0
        b = B / B0
        
        u_ave = 0.5 * (u + u_boris)
        gam_ave = np.sqrt(1 + np.linalg.norm(u_ave) ** 2)
        v_ave = u_ave / gam_ave
        
        kap = (e + np.cross(np.cross(v_ave, b), b) + np.dot(v_ave, e) * e)
        chi = (e + np.cross(v_ave, b)) ** 2 - np.dot(v_ave, e) ** 2
        const = B0 * rec / gam_rad ** 2
        
        return u_boris + const * (kap - gam_ave ** 2 * chi * v_ave) * dt
    
    def had_update_u(self, x, u, u_boris, E, B, dt):
        assert self.drag_type in ['pp', 'pg', 'bth'], 'Invalid hadronic interaction. Must be \'pp\', \'pg\', or \'bth\''
        assert self.had_scheme_type in ['cont', 'prob']
        
        u_ave = 0.5 * (u + u_boris)
        def transform_v(v_p, v_b):
            gamma = 1 / np.sqrt(1 - np.linalg.norm(v_b) ** 2)
            return 1 / (1 - np.dot(v_p, v_b)) * (v_p / gamma - v_b + (gamma / (1 + gamma)) * np.dot(v_p, v_b) * v_b)    
        
        if self.drag_type == 'pp':
            assert len(self.field_params) == 2
            xi = 0.17
            sigma = self.cross_section(np.sqrt(np.linalg.norm(u_ave) ** 2 + 1), 'pp')
#             lambda_0 = 50
#             u_b = np.array([])
            u_b = self.field_params[0]
            nr = self.field_params[1]
            #lambda_0 = self.field_params[2]
            
            gamma_ave = np.sqrt(1 + np.linalg.norm(u_ave) ** 2)
            gamma_b = np.sqrt(1 + np.linalg.norm(u_b) ** 2)
            
            v_rel = transform_v(u_ave / gamma_ave, u_b / gamma_b)
            #print(v_rel)
            gamma_rel = 1 / np.sqrt(1 - np.linalg.norm(v_rel) ** 2)
            n_rel = nr / gamma_b
            dt_rel = dt / gamma_b
            prob = np.linalg.norm(v_rel) * dt_rel * n_rel * sigma
               
            if self.had_scheme_type == 'prob':
                if np.random.rand() < prob:
                    u_new_rel = v_rel * gamma_rel * (1 - xi)
                    gamma_new_rel = np.sqrt(1 + np.linalg.norm(u_new_rel) ** 2)
                    v_new = transform_v(u_new_rel / gamma_new_rel, - u_b / gamma_b)
                    gamma_new = 1 / np.sqrt(1 - np.linalg.norm(v_new) ** 2)
                    u_new = v_new * gamma_new
                    return u_new
                return u_boris
                
            elif self.had_scheme_type == 'cont':
                u_new_rel = v_rel * gamma_rel * (1 - prob * xi)
                gamma_new_rel = np.sqrt(1 + np.linalg.norm(u_new_rel) ** 2)
                v_new = transform_v(u_new_rel / gamma_new_rel, - u_b / gamma_b)
                gamma_new = 1 / np.sqrt(1 - np.linalg.norm(v_new) ** 2)
                u_new = v_new * gamma_new
                return u_new
        
    def push(self, x, u, E_func, B_func, dt):
        
        E = E_func(x)
        B = B_func(x)
        
        u_new = self.update_u(x, u, E, B, dt)
        if self.drag_type == "sync":
            u_new = self.sync_update_u(x, u, u_new, E, B, dt)
        if self.drag_type == 'pp' or self.drag_type == 'pg' or self.drag_type == 'bth':
            u_new = self.had_update_u(x, u, u_new, E, B, dt)
            
        x_new = self.update_x(x, u_new, dt)
        
        return x_new, u_new
    
class GCA:
    def __init__(self): ...
        
    def update_u(self, x, u, E_func, B_func, dt):
        
        E = E_func(x)
        B = B_func(x)
        b = B / np.linalg.norm(B)
        E_p = np.dot(E, b)

        return u + E_p * dt
    
    def update_x(self, x, u, E_func, B_func, dt, err_crit = 1e-6, mu = 0):
        E0 = E_func(x)
        B0 = B_func(x)
        b0 = B / np.linalg.norm(B)
        w_E0 = np.cross(E0, B0) / (np.linalg.norm(E0) ** 2 + np.linalg.norm(B0) ** 2)
        v_E0 = w_E0 * (1 - np.sqrt(1 - 4 * np.linalg.norm(w_E0) ** 2))
        kap0 = 1 / np.sqrt(1 - np.linalg.norm(v_E0))
        gam0 = kap0 * np.sqrt(1 + u ** 2 + 2 * mu * np.linalg.norm(B0) * kap0)
        
        x1 = np.copy(x)
        x2 = np.copy(x)
        err = 1
        
        while err > err_crit:
            B1 = B_func(x1)
            E1 = E_func(x1)
            b1 = B1 / np.linalg.norm(B1)
            w_E1 = np.cross(E1, B1) / (np.linalg.norm(E1) ** 2 + np.linalg.norm(B1) ** 2)
            v_E1 = w_E1 * (1 - np.sqrt(1 - 4 * np.linalg.norm(w_E1) ** 2))
            kap1 = 1 / np.sqrt(1 - np.linalg.norm(v_E1))
            gam1 = kap1 * np.sqrt(1 + u ** 2 + 2 * mu * np.linalg.norm(B1) * kap1)
            
            v_p_ave = 0.5 * u * (b0 / gam0 + b1 / gam1)
            v_E_ave = 0.5 * (vE0 + vE1)
            x2 = (v_p_ave + v_E_ave) * dt
            err = np.linalg.norm(x2 - x1) / np.linalg.norm(x1)
            x1 = np.copy(x2)
            
        return x2
    
    def push(self, x, u, E_func, B_func, dt, err_crit = 1e-6, mu = 0):
        
        u_new = self.update_u(x, u, E_func, B_func, dt)
        x_new = self.update_x(x, u_new, E_func, B_func, dt, err_crit, mu)
        
        return x_new, u_new


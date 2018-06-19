# LGPEKF Attitude Filter (Lie Group Pseudo-measurement EKF)

import sys
import os
import pandas as pd
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# this library
sys.path.insert(0, '..')
from pyecca.so3.mrp import Mrp
from pyecca.so3.quat import Quat
from pyecca.so3.dcm import Dcm
from pyecca.utils.integrators import rk4
from pyecca.utils.sqrt import sqrt_covariance_predict, sqrt_correct
from pyecca.sim.kinematic import sim_params, analyze_hist, plot_hist, sim


# Filter Derivation


class SO3xR3(ca.SX):
    """
    Direct product of SO3 and R3
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.shape == (6, 1)
        self.r = Mrp(self[0:3])
        self.b = self[3:6]

    def inv(self):
        return self.__class__(ca.vertcat(self.r.inv(), -self.b))

    def __mul__(self, other):
        r = self.r * other.r
        b = self.b + other.b
        return self.__class__(ca.vertcat(r, b))

    @classmethod
    def exp(cls, xi):
        return cls(ca.vertcat(Mrp.exp(xi[0:3]), xi[3:6]))

    def log(self):
        return ca.vertcat(self.r.log(), self.b)

    def shadow_if_needed(self):
        r = ca.if_else(ca.norm_2(self.r) > 1, self.r.shadow(), self.r)
        return self.__class__(ca.vertcat(r, self.b))


# Prediction
n_x = 6
t = ca.SX.sym('t')
dt = ca.SX.sym('dt')
x = SO3xR3(ca.SX.sym('x', 6, 1))
x_h = SO3xR3(ca.SX.sym('x_h', 6, 1))
q = Quat(ca.SX.sym('q', 4, 1))
omega_b = ca.SX.sym('omega_b', 3, 1)

w_gyro_rw = ca.SX.sym('w_gyro_rw', 3, 1)
x_dot = ca.vertcat(x.r.derivative(omega_b - x.b), w_gyro_rw)
f_dynamics = ca.Function(
    'dynamics', [x, omega_b, w_gyro_rw],
    [x_dot],
    ['x', 'omega_b', 'w_gyro_rw'], ['x_dot'])

# can use lie gropu for integration due ot state coupling
x1 = rk4(lambda t, y: f_dynamics(y, omega_b, w_gyro_rw), t, x_h, dt)

f_predict_x = ca.Function(
    'x_predict',
    [x_h, omega_b, w_gyro_rw, dt],
    [x1],
    ['x0', 'omega_b', 'w_gyro_rw', 'dt'],
    ['x1']
)


# #### Square Root Covariance Prediction

eta = SO3xR3(ca.SX.sym('eta', 6, 1))  # (right)
f = ca.Function('f', [eta, x_h, w_gyro_rw], [ca.vertcat(-ca.mtimes(x_h.r.to_dcm(), eta.b), w_gyro_rw)])
f_J = ca.jacobian(f(eta, x_h, w_gyro_rw), eta)

# note, the estimated error is always zero when propagating the
# covariance, we might want the F without zero eta_R, when doing
# the LGEKF covariance correction term
F = ca.sparsify(ca.substitute(f_J, eta, ca.SX.zeros(n_x)))
f_F = ca.Function('F', [x_h], [F], ['x_h'], ['F'])

std_gyro = ca.SX.sym('std_gyro')
sn_gyro_rw = ca.SX.sym('sn_gyro_rw')
std_gyro_rw = sn_gyro_rw / ca.sqrt(dt)
Q = ca.diag(ca.vertcat(std_gyro, std_gyro, std_gyro, std_gyro_rw, std_gyro_rw, std_gyro_rw) ** 2)

# f_cond = ca.Function('cond', [x, u, PU, R], [condition_number(Si),],
#                       ['x', 'u', 'PU', 'R'], ['cond'])

W = ca.SX.sym('W', ca.Sparsity_lower(n_x))
W_dot_sol = sqrt_covariance_predict(W, F, Q)
f_W_dot = ca.Function('W_dot', [x_h, W, std_gyro, sn_gyro_rw, omega_b, dt], [W_dot_sol])
f_W_dot_lt = ca.Function('W_dot_lt', [x_h, W, std_gyro, sn_gyro_rw, omega_b, dt], [ca.tril(W_dot_sol)])
W1 = rk4(lambda t, y: f_W_dot_lt(x_h, y, std_gyro, sn_gyro_rw, omega_b, dt), t, W, dt)

f_predict_W = ca.Function(
    'predict_W',
    [x_h, W, std_gyro, sn_gyro_rw, omega_b, dt],
    [W1],
    ['x_h', 'W0', 'std_gyro', 'sn_gyro_rw', 'omega_b', 'dt'], ['W1'])

f_predict_x_W = ca.Function(
    'predict_x_W',
    [x_h, W, omega_b, std_gyro, sn_gyro_rw, dt],
    [ca.substitute(x1, w_gyro_rw, ca.SX.zeros(3)), W1],
    ['x0', 'W0', 'omega_b', 'std_gyro', 'sn_gyro_rw', 'dt'],
    ['x1', 'W1']
)


def test_sqrt_cov_prop():
    W_check = np.random.randn(6, 6)
    Q_check = np.diag([0.1, 0.1, 0.1, 0, 0, 0])
    xh_check = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
    omega_check = np.array([1, 2, 3])
    dt_check = 0.1

    f_check = ca.Function(
        'check',
        [x_h, W, std_gyro, sn_gyro_rw, omega_b, dt],
        [ca.mtimes([F, W, W.T]) + ca.mtimes([W, W.T, F.T]) + Q - ca.mtimes(W, W_dot_sol.T) - ca.mtimes(W_dot_sol, W.T)])
    assert np.linalg.norm(f_check(xh_check, W_check, 1, 1, [1, 1, 1], 0.1)) < 1e-5
    assert np.linalg.norm(ca.triu(f_W_dot(xh_check, W_check, 1, 1, [1, 1, 1], 0.1), False)) < 1e-5

    P_test = np.eye(6)
    P_test[3, 0] = 0.1
    P_test[1, 2] = 0.3
    P_test = P_test + P_test.T
    xh_test = np.random.randn(6)
    W_test = np.linalg.cholesky(P_test)
    assert np.linalg.norm(W_test.dot(W_test.T) - P_test) < 1e-10

    W1_test = f_predict_W(xh_test, W_test, 1, 1, omega_check, dt_check)
    print('PASS')


test_sqrt_cov_prop()

# ### Magnetometer Correction
#
# For the case of the magnetic heading, we calculate:
#
# \begin{align} H_{mag} &= \frac{\partial}{\partial \boldsymbol{\xi}} \log_{G'}^{\vee}(R_3(\xi_3)) \\
# &=  \frac{\partial}{\partial \boldsymbol{\xi}} \xi_3 \\
# &= \begin{pmatrix} 0 & 0 & 1\end{pmatrix}
# \end{align}

# In[7]:


H_mag = ca.SX(1, 6)
H_mag[0, 2] = 1

decl = ca.SX.sym('decl')
incl = ca.SX.sym('incl')  # only useful for sim, neglected in correction
e1 = ca.SX([1, 0, 0])
e2 = ca.SX([0, 1, 0])
e3 = ca.SX([0, 0, 1])

B_n = ca.mtimes(Dcm.exp(decl * e3) * Dcm.exp(-incl * e2), ca.SX([1, 0, 0]))
f_measure_mag = ca.Function('measure_mag', [x, decl, incl], [ca.mtimes(x.r.to_dcm().T, B_n)], ['x', 'decl', 'incl'],
                            ['y'])
yh_mag = f_measure_mag(x_h, decl, 0)  # can ignore incl, not used
std_mag = ca.SX.sym('std_mag')
gamma = ca.acos(yh_mag[2] / ca.norm_2(yh_mag))
h = ca.fmax(ca.sin(gamma), 1e-3)

y_b = ca.SX.sym('y_b', 3, 1)

R_nb = x_h.r.to_dcm()
y_n = ca.mtimes(R_nb, y_b)

omega_c_mag_n = -ca.atan2(y_n[1], y_n[0]) * ca.SX([0, 0, 1]) + decl


# #### Square Root Factorization

def fault_detection(r, S):
    beta = ca.mtimes([r.T, ca.inv(S), r])
    return beta

std_rot = std_mag + ca.norm_2(ca.diag(W)[0:2])  # roll/pitch and mag uncertainty contrib. to projection uncertainty
Rs_mag = 2 * ca.asin(std_rot / (2 * h))

W_mag, K_mag, Ss_mag = sqrt_correct(Rs_mag, H_mag, W)
S_mag = ca.mtimes(Ss_mag, Ss_mag.T)
r_mag = omega_c_mag_n[2]
x_mag = SO3xR3.exp(ca.mtimes(K_mag, r_mag)) * x_h
beta_mag_c = ca.SX.sym('beta_mag_c')  # normalizes beta mag so that 1 represents exceeding thresh
beta_mag = fault_detection(r_mag, S_mag) / beta_mag_c
r_std_mag = ca.diag(Ss_mag)

# ignore correction when near singular point
mag_ret = ca.if_else(
    std_rot / 2 > h,  # too close to vertical
    1,
    ca.if_else(
        ca.norm_2(ca.diag(W)[0:2]) > 0.1,  # too much roll/pitch noise
        2,
        0
    )
)

x_mag = ca.if_else(mag_ret == 0, x_mag, x_h)
W_mag = ca.if_else(mag_ret == 0, W_mag, W)
f_correct_mag = ca.Function(
    'correct_mag',
    [x_h, W, y_b, decl, std_mag, beta_mag_c],
    [x_mag, W_mag, beta_mag, r_mag, r_std_mag, mag_ret],
    ['x_h', 'W', 'y_b', 'decl', 'std_mag', 'beta_mag_c'],
    ['x_mag', 'W_mag', 'beta_mag', 'r_mag', 'r_std_mag', 'error_code'])


def test_mag_correct():
    W_test = 0.01 * np.eye(6)
    W_test[5, 0] = 1
    P_test = W_test.dot(W_test.T)
    x_test_mag, W_test_mag, beta_mag, r_mag, r_std_mag, mag_ret = f_correct_mag([0, 0, 0, 0, 0, 0], W_test, [1, 0, 0],
                                                                                0.1, 0.01, 100)
    print('beta_mag', beta_mag, 'error code', mag_ret)
    print('PASS')


test_mag_correct()

# ### Accelerometer Correction
#
# For the case of the gravity alignment, we calculate:
#
# \begin{align} H_{accel} &= \frac{\partial}{\partial \boldsymbol{\xi}} \log_{G'}^{\vee}(R_{12}(\xi_1, \xi_2)) \\
# &=  \frac{\partial}{\partial \boldsymbol{\xi}} \begin{pmatrix}\xi_1 \\ \xi_2 \end{pmatrix} \\
# &= \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0\end{pmatrix}
# \end{align}

# In[11]:


H_accel = ca.SX(2, 6)
H_accel[0, 0] = 1
H_accel[1, 1] = 1

f_measure_accel = ca.Function('measure_accel', [x], [ca.mtimes(x.r.to_dcm().T, ca.SX([0, 0, -9.8]))], ['x'], ['y'])
yh_accel = f_measure_accel(x_h)

n3 = ca.SX([0, 0, 1])
R_nb = x_h.r.to_dcm()
y_n = ca.mtimes(R_nb, -y_b)
v_n = ca.cross(y_n, n3) / ca.norm_2(y_b) / ca.norm_2(n3)
norm_v = ca.norm_2(v_n)
vh_n = v_n / norm_v
omega_c_accel_n = ca.sparsify(ca.if_else(norm_v > 0, ca.asin(norm_v) * vh_n, ca.SX([0, 0, 0])))

std_accel = ca.SX.sym('std_accel')
std_accel_omega = ca.SX.sym('std_accel_omega')

Rs_accel = ca.SX.eye(2) * (std_accel + ca.norm_2(omega_b) ** 2 * std_accel_omega)
W_accel, K_accel, Ss_accel = sqrt_correct(Rs_accel, H_accel, W)
S_accel = ca.mtimes(Ss_accel, Ss_accel.T)
r_accel = omega_c_accel_n[0:2]
beta_accel = fault_detection(r_accel, S_accel)
beta_accel_c = ca.SX.sym('beta_accel_c')
beta_accel = beta_accel / beta_accel_c
r_std_accel = ca.diag(Ss_accel)
x_accel = SO3xR3.exp(ca.mtimes(K_accel, r_accel)) * x_h
x_accel = ca.sparsify(x_accel)

# ignore correction when near singular point
accel_ret = ca.if_else(
    ca.fabs(ca.norm_2(y_b) - 9.8) > 1.0,  # accel magnitude not close to g,
    1,
    0
)

x_accel = ca.if_else(accel_ret == 0, x_accel, x_h)
W_accel = ca.if_else(accel_ret == 0, W_accel, W)

f_correct_accel = ca.Function(
    'correct_accel', [x_h, W, y_b, omega_b, std_accel, std_accel_omega, beta_accel_c],
    [x_accel, W_accel, beta_accel, r_accel, r_std_accel, accel_ret],
    ['x_h', 'W', 'y_b', 'omega_b', 'std_accel', 'std_accel_omega', 'beta_accel_c'],
    ['x_accel', 'W_accel', 'beta_accel', 'r_accel', 'r_std_accel', 'error_code'])


def test_accel_correct():
    x_sqrt_test_accel, W_test_accel, r_accel, r_std_accel_test, beta_test, test_valid = f_correct_accel(
        [0, 0, 0, 0, 0, 0], np.eye(6), [0, 0, 1], [1, 1, 1], [0.1], [1], 1)
    print('PASS')


test_accel_correct()

# #### Initialization
g_b = ca.SX.sym('g_b', 3, 1)
B_b = ca.SX.sym('B_b', 3, 1)

decl = ca.SX.sym('decl')
incl = ca.SX.sym('incl')  # only useful for sim, neglected in correction
e1 = ca.SX([1, 0, 0])
e2 = ca.SX([0, 1, 0])
e3 = ca.SX([0, 0, 1])
B_n = ca.mtimes(Dcm.exp(-incl * e2) * Dcm.exp(decl * e3), ca.SX([1, 0, 0]))

g_norm = ca.norm_2(g_b)
B_norm = ca.norm_2(B_b)

n3_b = -g_b / g_norm
Bh_b = B_b / B_norm

n2_dir = ca.cross(n3_b, Bh_b)
n2_dir_norm = ca.norm_2(n2_dir)
theta = ca.asin(n2_dir_norm)

# require
# * g_norm > 5
# * B_norm > 0
# * 10 degrees between grav accel and mag vector
init_ret = ca.if_else(
    ca.fabs(g_norm - 9.8) > 1,
    1,
    ca.if_else(
        B_norm <= 0,
        2,
        ca.if_else(
            theta < np.deg2rad(10),
            3,
            0
        )
    )
)

n2_b = n2_dir / n2_dir_norm

# correct based on declination to true east
n2_b = ca.mtimes(Dcm.exp(-decl * n3_b), n2_b)

tmp = ca.cross(n2_b, n3_b)
n1_b = tmp / ca.norm_2(tmp)

R0 = Dcm(ca.SX(3, 3))
R0[0, :] = n1_b
R0[1, :] = n2_b
R0[2, :] = n3_b

r0 = Mrp.from_dcm(R0)
r0 = ca.if_else(ca.norm_2(r0) > 1, r0.shadow(), r0)
b0 = ca.SX.zeros(3)  # initial bias
x0 = ca.if_else(init_ret == 0, ca.vertcat(r0, b0), ca.SX.zeros(6))
std0 = ca.SX.sym('std0', n_x)
W0 = ca.tril(ca.SX.zeros(6, 6))
for i in range(n_x):
    W0[i, i] = std0[i]
f_init = ca.Function('init', [g_b, B_b, decl, std0], [x0, W0, init_ret], ['g_b', 'B_b', 'decl', 'std_x0'],
                     ['x0', 'W0', 'error_code'])


def test_init():
    std0 = np.array([1, 1, 1, 0.1, 0.1, 0.1])
    assert np.linalg.norm(f_init([0, 0, 9.8], [1, 0, 0], 0, std0)[0] - np.array([1, 0, 0, 0, 0, 0])) < 1e-10
    assert f_init([-1.3, -0.12, -9.8], [0.15, -0.1, -0.44], 0.1, std0)[2] == 0
    assert f_init([0, 0, -9.8], [1, 0, 0], 0.1, std0)[2] == 0
    assert f_init([0, 0, 0], [1, 0, 0], 0.1, std0)[2] == 1
    assert f_init([0, 0, -9.8], [0, 0, 0], 0.1, std0)[2] == 2
    assert f_init([9.8, 0, 0], [1, 0, 0], 0.1, std0)[2] == 3
    print('PASS')


test_init()
f_get_W_diag = ca.Function('get_W_diag', [W], [ca.diag(W)], ['W'], ['diag_W'])
d = ca.SX.sym('d', n_x)
f_set_W_diag = ca.Function('set_W_diag', [d], [ca.diag(d)], ['d'], ['diag_d'])


def test_measure_with_init():
    x_test = [0.1, 0.2, 0.3, 0, 0, 0]
    std0 = [1, 1, 1, 0.1, 0.1, 0.1]
    y_accel_test = f_measure_accel(x_test)
    y_mag_test = f_measure_mag(x_test, 0.1, 0)
    x_init_test, W_init_test, test_ret = f_init(y_accel_test, y_mag_test, 0.1, std0)
    assert np.linalg.norm(x_test - x_init_test) < 1e-10
    print('PASS')
test_measure_with_init()

# Simulated Noise on Lie Group
w = ca.SX.sym('w', 3)
v = ca.SX.sym('v', 3)
f_noise_SO3 = ca.Function('noise_SO3', [w, v], [ca.mtimes(Mrp.exp(w).to_dcm(), v)], ['w', 'v'], ['n'])


def test_noise(plot=False):
    points = []
    for i in range(10000):
        w = np.array([0, 0.1, 0.1]) * np.random.randn(3) + [.1, .2, .3]
        points.append(np.reshape(f_noise_SO3(w, np.array([1, 0, 0])), -1))
    points = np.array(points).T

    if plot:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # draw sphere
        u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:40j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot3D(*points, '.', color='b', markersize=0.2)
        ax.plot_surface(x, y, z, color='grey', alpha=0.2, edgecolors='w')

        ax.view_init(elev=15, azim=30)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.title('noise on SO3')
    print('PASS')
test_noise(plot=False)

# Simulation/ Code Generation Data Structures
func_sim = {
    'measure_mag': f_measure_mag,
    'measure_accel': f_measure_accel,
    'f_noise_SO3': f_noise_SO3,
    'xi': ca.Function('log_eta_R', [x, x_h], [(x_h * x.inv()).shadow_if_needed().log()], ['x', 'x_h'], ['xi']),
    'predict_x': f_predict_x,
}
func = {
    'mrp_shadow': ca.Function('mrp_shadow', [x.r], [x.r.shadow()], ['r'], ['r_s']),
    'mrp_to_quat': ca.Function('mrp_to_quat', [x.r], [x.r.to_quat()], ['r'], ['q']),
    'quat_to_euler': ca.Function('quat_to_euler', [q], [q.to_euler()], ['q'], ['e']),
    'predict_x_W': f_predict_x_W,
    'correct_accel': f_correct_accel,
    'correct_mag': f_correct_mag,
    'init': f_init,
}

# Simulation

# rough calculus of expected noise levels
l_acc = np.linalg.norm([0, 0, -9.8])
l_mag = np.linalg.norm([0.47, 0.2, 0.26])
std_acc = np.sqrt(1e-3) / l_acc  # divide by l to convert to orientation noise power (dw = dtheta*r)
std_mag = np.sqrt(2e-6) / l_mag

print('std acc', std_acc, 'std mag', std_mag)

sim_funcs = {}
sim_funcs.update(func)
sim_funcs.update(func_sim)
sim_params['tf'] = 10
for i in range(3):
    sim_params['x0'] = np.random.randn(6) * np.array([1, 1, 1, 0.1, 0.1, 0.1])
    print('x0', sim_params['x0'])
    hist = sim(sim_funcs, sim_params, do_init=True)
    mean, std = analyze_hist(hist, sim_params['tf'] / 2)
    # if np.max(np.fabs(mean)) > 1 or np.max(np.fabs(mean)) > 1:
    plot_hist(hist, att=False, meas=False)
    sys.stdout.flush()

hist = sim(sim_funcs, sim_params, enable_progress=True, do_init=True)
analyze_hist(hist, sim_params['tf'] / 2)
plot_hist(hist, meas=False, att=False)

# Save data to csv file.
data = {}
for k in hist.keys():
    if k is 't':
        continue
    vals = hist[k]
    for i in range(vals.shape[1]):
        data['{:s}_{:d}'.format(k, i)] = hist[k][:, i]
res = pd.DataFrame(data=data, index=pd.Float64Index(np.reshape(hist['t'], -1)))
with open('results.csv', 'w') as f:
    f.write(res.to_csv())

# Code Generation
gen = ca.CodeGenerator('casadi_att_lgpekf.c', {'main': False, 'mex': False, 'with_header': True, 'with_mem': True})
for f_name in func:
    gen.add(func[f_name])

install_dir = os.path.abspath(os.path.join(os.environ['HOME'], 'git/phd/px4/src/modules/cei/gen/att_lgpekf'))
if not os.path.exists(install_dir):
    os.mkdir(install_dir)
gen.generate(install_dir + os.path.sep)

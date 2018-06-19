import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def f_omega(t):
    """angular velocity of vehicle as a function of time, in the body frame"""
    return 1 * np.array([0.1, 0.2, 0.3]) + 3 * np.sin(2 * np.pi * 0.1 * t)


sim_params = {
    # sampling
    't': 0,
    'tf': 10,
    'dt': 0.005,  # 200 Hz
    'mod_accel': 1,  # 200 Hz
    'mod_mag': 4,  # 50 Hz
    'mag_delay_periods': 0,  # simulates measurement lag by periods*dt
    'accel_delay_periods': 0,  # simulates measurement lag by periods*dt
    'max_delay_periods': 10,

    # measurement noise
    'std_accel': 35.0e-3,  # accel std/dev / accel norm, radians (since used for rotation)
    'std_accel_omega': 100e-3,  # increase uncertainty if rotating rapidly, could have centrip. accel, prop. to omega^2

    'std_mag': 2.5e-3,  # mag std/dev / mag norm, radians (since used for rotation)

    # initial uncertainty
    'std_x0': np.array([1, 1, 1, 0.1, 0.1, 0.1]),

    # process noise
    'std_gyro': 1e-3,  # attitude process noise (gyro noise) [rad/s]
    'sn_gyro_rw': 1e-5,  # gyro bias random walk noise power [rad/s * sqrt(s)]

    # angular velocity function
    'f_omega': f_omega,

    # centripetal acceleration distrubance
    'c_centrip': 0.05,

    'mag_decl': 0.1,  # magnetic declination, rad
    'mag_incl': 0,  # magnetic inclination, rad, ignored in estimation

    # initial conditions for simulated rigid body
    'x0': 1 * np.array([0.4, -0.2, 0.3, 0.1, 0, -0.1]),

    'beta_mag_c': 6.6,  # 99% for n=1
    'beta_accel_c': 9.2  # 99% for n=2
}


class Fifo:

    def __init__(self, n):
        self._data = []
        self._n = n

    def insert(self, d):
        self._data.insert(0, np.reshape(d, -1))
        if len(self._data) > self._n:
            self._data.pop(-1)

    def empty(self):
        self._data = []

    def mean(self):
        return np.mean(np.array(self._data), 0)

    def full(self):
        return len(self._data) == self._n

    def get(self, i):
        if len(self._data) == 0:
            raise ValueError('no data')
        elif i > len(self._data) - 1:
            return self._data[-1]
        elif i < 0:
            raise ValueError('must be positive')
        else:
            return self._data[i]


def handle_shadow(x, s, q, func):
    if np.linalg.norm(x[0:3]) > 1:
        x[0:3] = np.reshape(func['mrp_shadow'](x[0:3]), -1)
        s = not s
    q = func['mrp_to_quat'](x[0:3])
    if s:
        q *= -1
    return x, s, q


def sim(func, params, enable_progress=False, do_init=True, verbose=True):
    p = params

    # all data will be stored in this dictionary,
    hist = {}

    # noise matrices
    dt_mag = p['dt'] * p['mod_mag']
    dt_accel = p['dt'] * p['mod_accel']

    n_x = 6
    W = np.zeros((n_x, n_x))
    for i in range(n_x):
        W[i, i] = p['std_x0'][i]

    # data
    x = p['x0']
    q = np.reshape(func['mrp_to_quat'](x[0:3]), -1)
    xh = np.zeros(n_x)
    qh = np.reshape(func['mrp_to_quat'](xh[0:3]), -1)
    y_accel = np.zeros(3)
    y_mag = np.zeros(3)
    yh_accel = np.zeros(3)
    yh_mag = np.zeros(3)
    x_delayed = Fifo(p['max_delay_periods'])
    xh_delayed = Fifo(p['max_delay_periods'])
    beta_mag = np.zeros(1)
    r_std_mag = np.zeros(1)
    beta_accel = np.zeros(1)
    r_std_accel = np.zeros(2)
    r_mag = np.zeros(1)
    r_accel = np.zeros(2)
    mag_ret = 0
    accel_ret = 0

    # handle initial shadow state
    x, shadow, qh = handle_shadow(x, 0, qh, func)
    xh, shadowh, qh = handle_shadow(xh, 0, qh, func)
    shadow = 0  # need to track shadow state to give a consistent quaternion
    shadowh = 0  # need to track shadow state to give a consistent quaternion

    # handle initialization
    if do_init:
        initialized = False
    else:
        initialized = True
    accel_init_buffer = Fifo(10)
    mag_init_buffer = Fifo(10)

    # check initial guess is valid
    if not np.all(np.isfinite(np.array(xh))):
        raise RuntimeError('initial state NaN', xh)
    if not np.all(np.isfinite(np.array(xh))):
        raise RuntimeError('initial W NaN', W)

    i = 0

    t_vals = np.arange(0, p['tf'] + p['dt'], p['dt'])

    f_predict = lambda u: lambda t, x: func['dynamics_T'](x, u)

    # tqdm creates a progress bar from the range
    for t in tqdm(t_vals, disable=not enable_progress):
        i += 1

        # get gyro bias and rotation rate
        omega = p['f_omega'](t)

        # simulate the actual motion of the rigid body
        omega_meas = omega + x[3:6] + np.random.randn(3) * p['std_gyro']
        w_gyro_rw = np.random.randn(3) * p['sn_gyro_rw'] * np.sqrt(p['dt'])
        x = func['predict_x'](x, omega_meas, w_gyro_rw, p['dt'])
        x, shadow, q = handle_shadow(x, shadow, q, func)
        x_delayed.insert(x)

        # measurement indicators
        mag_updated = False
        accel_updated = False

        # measure accel
        if i % p['mod_accel'] == 0:
            w = p['std_accel'] * np.random.randn(2) / np.sqrt(2)
            x_old = x_delayed.get(p['accel_delay_periods'])
            y_accel = func['f_noise_SO3']([w[0], w[1], 0], func['measure_accel'](x_old))

            # add centripetal term
            y_accel += p['c_centrip'] * np.linalg.norm(omega_meas) ** 2 * np.array([1, 0, 0])

            accel_updated = True
            if not initialized:
                accel_init_buffer.insert(y_accel)

        # measure mag
        if i % p['mod_mag'] == 0:
            # simulate measurement
            w = np.random.randn(1) * p['std_mag']
            x_old = x_delayed.get(p['mag_delay_periods'])
            y_mag = func['f_noise_SO3']([0, 0, w], func['measure_mag'](x_old, p['mag_decl'], p['mag_incl']))
            mag_updated = True
            if not initialized:
                mag_init_buffer.insert(y_mag)

        # intialization
        if not initialized:

            if mag_init_buffer.full() and accel_init_buffer.full():
                mag_mean = mag_init_buffer.mean()
                accel_mean = accel_init_buffer.mean()
                x0, W0, init_ret = func['init'](accel_mean, mag_mean, p['mag_decl'], p['std_x0'])
                if int(init_ret) == 0:
                    initialized = True
                    xh = x0
                    W = W0
                    mag_init_buffer.empty()
                    accel_init_buffer.empty()

        # prediction/ correction
        else:

            # predict the motion of the rigid body and covariance
            xh, W = func['predict_x_W'](xh, W, omega_meas, p['std_gyro'], p['sn_gyro_rw'], p['dt'])
            if not (W.is_regular() and xh.is_regular()):
                raise RuntimeError('covariance propagation NaN', t, xh, W)
            xh, shadowh, qh = handle_shadow(xh, shadowh, qh, func)
            xh_delayed.insert(xh)

            # correction for accel
            if accel_updated:
                yh_accel = func['measure_accel'](xh)
                xh_old = xh_delayed.get(p['accel_delay_periods'])
                x1, W1, beta_accel, r_accel, r_std_accel, accel_ret = func['correct_accel'](
                    xh, W, y_accel, omega_meas, p['std_accel'], p['std_accel_omega'], p['beta_accel_c'])
                if beta_accel > 1:
                    if verbose:
                        print('accel fault, beta', beta_accel)
                if accel_ret == 0:
                    if not W1.is_regular() or not x1.is_regular():
                        raise RuntimeError('accel correction NaN')
                    xh = x1
                    W = W1
                    xh, shadowh, qh = handle_shadow(xh, shadowh, qh, func)
                xh, shadowh, qh = handle_shadow(xh, shadowh, qh, func)

            # correction for mag
            if mag_updated:
                yh_mag = func['measure_mag'](xh, p['mag_decl'], 0)
                xh_old = xh_delayed.get(p['mag_delay_periods'])
                x1, W1, beta_mag, r_mag, r_std_mag, mag_ret = func['correct_mag'](
                    xh, W, y_mag, p['mag_decl'], p['std_mag'], p['beta_mag_c'])
                if beta_mag > 1:
                    if verbose:
                        print('mag fault, beta', beta_mag)
                if mag_ret == 0:
                    if not W1.is_regular() or not x1.is_regular():
                        raise RuntimeError('mag correction NaN')
                    xh = x1
                    W = W1
                    xh, shadowh, qh = handle_shadow(xh, shadowh, qh, func)

        data = ({
            'omega': omega,
            'omega_meas': omega_meas,
            'mag_updated': mag_updated,
            'accel_updated': accel_updated,
            'euler': func['quat_to_euler'](q),
            'eulerh': func['quat_to_euler'](qh),
            'q': q,
            'qh': qh,
            'shadow': shadow,
            'shadowh': shadowh,
            't': t,
            'x': x,
            'xh': xh,
            'xi': func['xi'](x, xh),
            'y_accel': y_accel,
            'y_mag': y_mag,
            'yh_accel': yh_accel,
            'yh_mag': yh_mag,
            'std': np.reshape(np.diag(W), -1),
            'beta_mag': beta_mag,
            'r_mag': r_mag,
            'r_std_mag': r_std_mag,
            'beta_accel': beta_accel,
            'r_accel': r_accel,
            'r_std_accel': np.reshape(r_std_accel, -1),
            'mag_ret': mag_ret,
            'accel_ret': accel_ret
        })
        for key in data.keys():
            if key not in hist.keys():
                hist[key] = []
            hist[key].append(np.reshape(data[key], -1))
        t += p['dt']

    for k in hist.keys():
        hist[k] = np.array(hist[k])
    return hist


def analyze_hist(hist, t):
    ti = int(t / (hist['t'][1] - hist['t'][0]))
    if ti < len(hist['t']):
        mean = list(np.rad2deg(np.mean(hist['xi'][ti:], 0)))
        std = list(np.rad2deg(np.std(hist['xi'][ti:], 0)))
        print('error statistics after {:0.0f} seconds'.format(t))
        print('mean (deg)\t: {:10.4f} roll, {:10.4f} pitch, {:10.4f} yaw'.format(*mean))
        print('std  (deg)\t: {:10.4f} roll, {:10.4f} pitch, {:10.4f} yaw'.format(*std))
    return mean, std


def plot_hist(hist, figsize=(10, 5), att=True, meas=True, est=True):
    t = hist['t']
    tf = t[-1]
    t10 = int(len(t) / 10)
    r = hist['x'][:, 0:3]
    bg = hist['x'][:, 3:6]
    rh = hist['xh'][:, 0:3]
    bgh = hist['xh'][:, 3:6]

    if att:
        fig = plt.figure(figsize=figsize)

        ax = plt.subplot(411)
        ax.plot(t, r, 'r')
        ax.plot(t, rh, 'k')
        ax.grid()
        # ax.xlabel('t, sec')
        ax.set_ylabel('mrp')
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, tf)
        ax.set_title('attitude representations')

        ax = plt.subplot(412)
        ax.plot(t, hist['shadow'], 'r')
        ax.plot(t, hist['shadowh'], 'k')
        ax.set_ylabel('mrp shadow id')
        ax.set_xlim(0, tf)
        ax.grid()
        ax.set_xlabel('t, sec')

        ax = plt.subplot(413)
        ax.plot(t, hist['q'], 'r')
        ax.plot(t, hist['qh'], 'k')
        # ax.set_xlabel('t, sec')
        ax.set_ylabel('q')
        ax.grid()
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, tf)

        ax = plt.subplot(414)
        ax.plot(t, np.rad2deg(hist['euler']), 'r')
        ax.plot(t, np.rad2deg(hist['eulerh']), 'k')
        ax.set_ylabel('euler, deg')
        ax.grid()
        ax.set_ylim(-200, 200)
        ax.set_xlim(0, tf)
        ax.set_xlabel('t, sec')

    if meas:
        fig = plt.figure(figsize=figsize)

        ax = plt.subplot(211)
        ax.plot(t, hist['y_accel'], 'r')
        ax.plot(t, hist['yh_accel'], 'k')
        ax.set_ylabel('accel., norm.')
        ax.set_xlim(0, tf)
        ax.grid()
        ax.set_title('measurements')

        ax = plt.subplot(212)
        ax.plot(t, hist['y_mag'], 'r')
        ax.plot(t, hist['yh_mag'], 'k')
        ax.set_xlabel('t, sec')
        ax.set_ylabel('mag., norm.')
        ax.set_xlim(0, tf)
        ax.grid()

    if est:
        fig = plt.figure(figsize=figsize)

        ax = plt.subplot(211)
        std_r = np.rad2deg(hist['std'][:, 0:3])
        r_err = np.rad2deg(hist['xi'][:, 0:3])

        ax.set_prop_cycle(plt.cycler('color', 'rgb') + plt.cycler('linestyle', ['-.', '-.', '-.']))
        h_sig = ax.plot(t, std_r, t, -std_r)
        ax.set_prop_cycle(plt.cycler('color', 'rgb') + plt.cycler('linestyle', ['-', '-', '-']))
        h_eta = ax.plot(t, r_err)
        ax.set_ylabel('rotation error, deg')
        ax.set_xlim(0, tf)
        ax.grid()
        ax.legend([h_eta[0], h_eta[1], h_eta[2], h_sig[0]],
                  ['$\\xi_1$', '$\\xi_2$', '$\\xi_3$', '$\sigma$'],
                  loc='lower right', ncol=6)
        y_lim = 3 * np.max(np.abs(np.hstack([r_err, std_r]))[t10:, :])
        ax.set_ylim(-y_lim, y_lim)

        ax = plt.subplot(212)
        ax.set_title('estimation')
        std_b = hist['std'][:, 3:6]
        b_err = hist['xi'][:, 3:6]
        ax.set_prop_cycle(plt.cycler('color', 'rgb') + plt.cycler('linestyle', ['-.', '-.', '-.']))
        h_sig = ax.plot(t, std_b, t, -std_b)
        ax.set_prop_cycle(plt.cycler('color', 'rgb') + plt.cycler('linestyle', ['-', '-', '-']))
        h_eta = ax.plot(t, b_err)
        ax.set_ylabel('gyro bias error, deg/s')
        ax.set_xlim(0, tf)
        ax.grid()
        ax.set_xlabel('t, sec')
        ax.legend([h_eta[0], h_eta[1], h_eta[2], h_sig[0]],
                  ['$\\xi_4$', '$\\xi_5$', '$\\xi_6$', '$\sigma$'],
                  loc='lower right', ncol=6)
        y_lim = 3 * np.max(np.abs(np.hstack([b_err, std_b]))[t10:, :])
        ax.set_ylim(-y_lim, y_lim)

    plt.show()

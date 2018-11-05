import numpy as np
import simpy

import pyecca2.msgs as msgs
import pyecca2.uros as uros


class Simulator:

    def __init__(self, core, eqs, x0):
        self.core = core

        # publications
        self.pub_sim = uros.Publisher(core, 'sim_state', msgs.VehicleState)
        self.pub_imu = uros.Publisher(core, 'imu', msgs.Imu)
        self.pub_mag = uros.Publisher(core, 'mag', msgs.Mag)

        # subscriptions
        self.sub_params = uros.Subscriber(core, 'params', msgs.Params, self.params_callback)

        # parameters
        self.param_list = []

        def add_param(name, value, type):
            p = uros.Param(self.core, 'sim/' + name, value, type)
            self.param_list.append(p)
            return p

        self.std_mag = add_param('std_mag', 2.5e-3, 'f8')
        self.std_accel = add_param('std_accel', 35.0e-3, 'f8')
        self.std_gyro = add_param('std_gyro', 1e-3, 'f8')
        self.sn_gyro_rw = add_param('sn_gyro_rw', 1e-5, 'f8')
        self.dt_sim = add_param('dt_sim', 1.0 / 400, 'f8')
        self.dt_mag = add_param('dt_mag', 1.0 / 50, 'f8')
        self.dt_imu = add_param('dt_imu', 1.0 / 200, 'f8')
        self.mag_decl = add_param('mag_decl', 0, 'f8')
        self.mag_incl = add_param('mag_incl', 0, 'f8')
        self.mag_str = add_param('mag_str', 1e-1, 'f8')
        self.g = add_param('g', 9.8, 'f8')
        self.enable_noise = add_param('enable_noise', True, '?')

        # msgs
        self.msg_sim_state = msgs.VehicleState()
        self.msg_imu = msgs.Imu()
        self.msg_mag = msgs.Mag()

        # misc
        self.t_last_sim = 0
        self.t_last_imu = 0
        self.t_last_mag = 0
        self.x0 = x0

        self.eqs = eqs
        np.random.seed()
        simpy.Process(core, self.run())

    def params_callback(self, msg):
        for p in self.param_list:
            p.update()

    def randn(self, *args, **kwargs):
        return np.random.randn(*args) * self.enable_noise.get()

    def run(self):
        x = self.x0
        i = 0
        time_eps = 1e-7  # small period of time to prevent missing pub

        while True:

            # time
            t = self.core.now

            # true angular velocity in body frame
            time_varying_omega = True
            if time_varying_omega:
                omega_b = 10 * np.array([
                    (1 + np.sin(2 * np.pi * 0.1 * t + 1)) / 2,
                    -(1 + np.sin(2 * np.pi * 0.2 * t + 2)) / 2,
                    (1 + np.cos(2 * np.pi * 0.3 * t + 3)) / 2])
            else:
                omega_b = 10*np.array([1, 2, 3])

            # compute dt
            dt = t - self.t_last_sim
            self.t_last_sim = t

            # propagate
            w_gyro_rw = self.randn(3)
            if t != 0:
                x = self.eqs['sim']['simulate'](
                    t, x, omega_b, self.sn_gyro_rw.get(), w_gyro_rw, dt)

            # measure and publish accel/gyro
            if t== 0 or t - self.t_last_imu >= self.dt_imu.get() - time_eps:
                self.t_last_imu = t

                # publish sim state at same rate as estimators, which are based
                # on imu pub so that logger doesn't grab data out of sync and
                # report larger error than exists in reality due to delayed data
                q, r, b_g = self.eqs['sim']['get_state'](x)

                self.msg_sim_state.data['time'] = t
                self.msg_sim_state.data['q'] = q.T
                self.msg_sim_state.data['r'] = r.T
                self.msg_sim_state.data['b'] = b_g.T
                self.msg_sim_state.data['omega'] = omega_b.T
                self.pub_sim.publish(self.msg_sim_state)

                # measure
                w_gyro = self.randn(3)
                w_accel = self.randn(3)
                y_gyro = self.eqs['sim']['measure_gyro'](
                    x, omega_b, self.std_gyro.get(), w_gyro).T

                y_accel = self.eqs['sim']['measure_accel'](
                    x, self.g.get(), self.std_accel.get(), w_accel).T

                # fake centrip acceleration term to model disturbance
                # y_accel += 1e-3*np.array([[0, 1, 0]]) * np.linalg.norm(omega_b)**2

                # publish
                self.msg_imu.data['time'] = t
                self.msg_imu.data['gyro'] = y_gyro
                self.msg_imu.data['accel'] = y_accel
                self.pub_imu.publish(self.msg_imu)

            # measure and publish mag
            if t - self.t_last_mag >= self.dt_mag.get() - time_eps:
                self.t_last_mag = t

                # measure
                w_mag = self.randn(3)
                y_mag = self.eqs['sim']['measure_mag'](
                    x, self.mag_str.get(), self.mag_decl.get(), self.mag_incl.get(),
                    self.std_mag.get(), w_mag).T

                # publish
                self.msg_mag.data['time'] = t
                self.msg_mag.data['mag'] = y_mag
                self.pub_mag.publish(self.msg_mag)

            i += 1
            yield simpy.Timeout(self.core, self.dt_sim.get())

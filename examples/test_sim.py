import simpy
import pandas as pd
import numpy as np
from collections import namedtuple
import pickle
import os
import sys
import matplotlib.pyplot as plt
import multiprocessing as mp




class Orb:
    """publish/subscribe object request broker"""

    def __init__(self, env, type_check=True, verbose=False):
        self.pub_types = {}
        self.publishers = {}
        self.subscribers = {}
        self.env = env
        self.type_check = type_check
        self.verbose = verbose

    def get_type(self, topic):
        return self.pub_types[topic]

    def publish(self, topic, type):
        assert topic not in self.publishers.keys()
        store = simpy.Store(self.env)
        self.publishers[topic] = store
        self.env.process(self.update(topic))
        self.pub_types[topic] = type
        return store

    def subscribe(self, topic, method):
        if self.verbose:
            print('subscribing to ', topic, 'with ', method)
        if topic not in self.subscribers.keys():
            self.subscribers[topic] = []
        store = simpy.Store(self.env)
        self.subscribers[topic].append(method)
        return store

    def update(self, topic):
        while True:
            data = yield self.publishers[topic].get()
            if self.type_check:
                assert isinstance(data, self.pub_types[topic])
            if topic in self.subscribers.keys():
                for i in range(len(self.subscribers[topic])):
                    self.subscribers[topic][i](data)


class Estimator:

    def __init__(self, env, orb, funcs, params):
        self.env = env
        self.orb = orb
        self.funcs = funcs
        self.params = params

        # subscriptions
        self.mag_sub = orb.subscribe('mag', self.mag_callback)
        self.imu_sub = orb.subscribe('imu', self.imu_callback)

        self.x = np.zeros(6)
        self.W = np.zeros((6, 6))

    def mag_callback(self, msg: MagData):
        y = np.array([msg.x, msg.y, msg.z])
        x1, W1, beta, r, r_std, ret = self.funcs['correct_mag'](
            self.x, self.W, y,
            self.params['decl'], self.params['std_mag'], self.params['beta_mag_c'])

    def imu_callback(self, msg: ImuData):
        dt = 1.0/200
        y = np.array([msg.ax, msg.ay, msg.az])
        omega = np.array([msg.gx, msg.gy, msg.gz])
        x1, W1, beta, r, r_std, ret = self.funcs['correct_accel'](
            self.x, self.W, y, omega,
            self.params['std_accel'], self.params['std_accel_omega'], self.params['beta_accel_c'])
        self.funcs['predict_x_W'](self.x, self.W, omega,
                                  self.params['std_gyro'], self.params['sn_gyro_rw'], dt)


class KinematicSim:

    def __init__(self, env, orb, funcs, params):
        self.env = env
        self.orb = orb
        self.funcs = funcs
        self.params = params

        # publications
        self.mag_pub = orb.publish('mag', MagData)
        self.imu_pub = orb.publish('imu', ImuData)

        # initial state
        self.x = np.zeros(6)
        self.shadow = 0
        self.omega = np.array([1, 2, 3])

        # start processes
        env.process(self.propagate())
        env.process(self.mag_update())
        env.process(self.imu_update())

    def print_functions(self):
        print('\nsimulation functions')
        for k in self.funcs.keys():
            print('{:15s}: {:s}'.format(k, str(self.funcs[k])))

    def propagate(self):
        dt = 1.0/200
        while True:
            w_gyros_rw = 0.001*np.random.randn(3)
            x = self.funcs['predict_x'](self.x, self.omega, w_gyros_rw, dt)
            if np.linalg.norm(x[0:3]) > 1:
                x[0:3] = self.funcs['mrp_shadow'](x[0:3])
                self.shadow = not self.shadow
            self.x = x
            yield self.env.timeout(dt)

    def mag_update(self):
        while True:
            w = np.random.randn(3) * self.params['std_mag']
            y = self.funcs['measure_mag'](self.x, self.params['decl'], self.params['incl']) + w
            yield self.mag_pub.put(MagData(self.env.now, y[0], y[1], y[2]))
            yield self.env.timeout(1.0/50)

    def imu_update(self):
        while True:
            w_gyro = self.params['std_gyro']*np.random.randn(3)
            omega_bias = self.x[3:6]
            omega_meas = self.omega + omega_bias + w_gyro
            w_accel = np.random.randn(3) * self.params['std_accel']
            y = self.funcs['measure_accel'](self.x) + w_accel
            yield self.imu_pub.put(ImuData(
                self.env.now, omega_meas[0], omega_meas[1], omega_meas[2], y[0], y[1], y[2]))
            yield self.env.timeout(1.0/200)


class Logger:

    def __init__(self, env, orb):
        self.env = env
        self.orb = orb
        self.subs = {}
        self.topics = ['mag', 'imu']

        def callback(topic):
            return lambda x: self.log_topic(str(topic), x)

        for topic in self.topics:
            self.subs[topic] = orb.subscribe(topic, callback(topic))
        self.data = {}

    def log_topic(self, topic, msg):
        if topic not in self.data.keys():
            self.data[topic] = []
        self.data[topic].append(list(msg))

    def write(self):
        pd_data = {}
        for topic in self.data.keys():
            orb_type = self.orb.get_type(topic)
            data = np.array(self.data[topic])
            pd_data[topic] = pd.DataFrame(
                data=data[:, 1:],
                columns=orb_type._fields[1:],
                index=pd.Float64Index(data=data[:,0], name='t, sec'))
        return pd_data


# load simulation functions
pkl_path = os.path.join(os.path.dirname(__file__), 'sim_funcs.pkl')
with open(pkl_path, 'rb') as f:
    funcs = pickle.load(f)


def do_sim(i, progress=False):
    params = {
        'std_mag': 2.5e-3,
        'decl': 0,
        'incl': 1,
        'beta_mag_c': 6.2,
        'std_accel': 35e-3,
        'std_accel_omega': 100e-3,
        'std_gyro': 1e-3,
        'sn_gyro_rw': 0.01e-3,
        'beta_accel_c': 9.6,
        'tf': 10,
    }
    env = simpy.Environment()
    orb = Orb(env)
    sim = KinematicSim(env, orb, funcs, params)
    #sim.print_functions()
    estimator = Estimator(env, orb, funcs, params)
    logger = Logger(env, orb)

    t_vals = np.linspace(0, params['tf'], 10)[1:]
    from tqdm import tqdm
    if progress:
        for t in tqdm(t_vals):
            env.run(until=t)
    else:
        env.run(until=params['tf'])
    return logger.write()


def test_sim():
    data = {}

    do_sim(1)
    sim_count = 1
    #pool = mp.Pool(mp.cpu_count())

    #res = pool.map(do_sim, range(sim_count))
    #mag = pd.concat([res[i]['mag'] for i in range(sim_count)])

    #plt.figure()
    #mag.plot(style='.')

    #plt.figure()
    #imu = pd.concat([res[i]['imu'] for i in range(sim_count)])
    #imu.plot(style='.')
    #plt.show()
import simpy
import pandas as pd
import numpy as np
import pickle
import os
from pyecca.sim import msgs


class Core(simpy.Environment):

    def __init__(self, type_check=True, verbose=False):
        super().__init__()
        self.params = {}
        self.pub_types = {}
        self.publishers = {}
        self.subscribers = {}
        self.type_check = type_check
        self.verbose = verbose

    def get_type(self, topic):
        return self.pub_types[topic]

    def publish(self, topic, type):
        assert topic not in self.publishers.keys()
        store = simpy.Store(self)
        self.publishers[topic] = store
        self.process(self._update(topic))
        self.pub_types[topic] = type
        return store

    def subscribe(self, topic, method):
        if self.verbose:
            print('subscribing to ', topic, 'with ', method)
        if topic not in self.subscribers.keys():
            self.subscribers[topic] = []
        store = simpy.Store(self)
        self.subscribers[topic].append(method)
        return store

    def _update(self, topic):
        while True:
            data = yield self.publishers[topic].get()
            if self.type_check:
                assert isinstance(data, self.pub_types[topic])
            if topic in self.subscribers.keys():
                for i in range(len(self.subscribers[topic])):
                    self.subscribers[topic][i](data)


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


class Node:

    def __init__(self, core):
        self.core = core


class Estimator(Node):

    def __init__(self, core, funcs):
        super().__init__(core)
        self.funcs = funcs
        self.time_last_imu = self.core.now

        # subscriptions
        self.mag_sub = core.subscribe('mag', self.mag_callback)
        self.imu_sub = core.subscribe('imu', self.imu_callback)

        # publications
        self.est_pub = core.publish('estimator_status', msgs.EstimatorStatus)

        self.x = np.zeros(6)
        self.W = np.zeros((6, 6))

        self.mag_init_buf = Fifo(20)
        self.accel_init_buf = Fifo(20)
        self.initialized = False
        self.shadow = 0

    def handle_correct(self, x, W, beta, ret, msg):
        if ret != 0:
            return
        if beta > 1:
            print(msg, 'fault', beta)
        x = np.array(x)
        W = np.array(W)
        if np.linalg.norm(x[0:3]) > 1:
            x[0:3] = np.array(self.funcs['mrp_shadow'](x[0:3]))
            self.shadow = not self.shadow
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(W))):
            raise RuntimeError('Nan in', msg)
        self.x = x
        self.W = W

    def mag_callback(self, msg: msgs.Mag):
        y = np.array(msg.m)
        if not self.initialized:
            self.mag_init_buf.insert(y)
        else:
            p = self.core.params
            x1, W1, beta, r, r_std, ret = self.funcs['correct_mag'](
                self.x, self.W, y,
                p['decl'], p['std_mag'], p['beta_mag_c'])
            self.handle_correct(x1, W1, beta, ret, 'mag')

    def imu_callback(self, msg: msgs.Imu):
        now = self.core.now
        p = self.core.params
        dt = now - self.time_last_imu
        self.time_last_imu = now
        y_acc = np.array(msg.a)
        omega = np.array(msg.g)

        if not self.initialized:
            self.accel_init_buf.insert(y_acc)
            if self.accel_init_buf.full() and self.mag_init_buf.full():
                x1, W1, ret = self.funcs['init'](
                    self.accel_init_buf.mean(), self.mag_init_buf.mean(),
                    p['decl'], p['std_x0'])
                self.handle_correct(x1, W1, 0, ret, 'init')
                self.initialized = True
        else:
            x1, W1, beta, r, r_std, ret = self.funcs['correct_accel'](
                self.x, self.W, y_acc, omega,
                p['std_accel'], p['std_accel_omega'], p['beta_accel_c'])
            self.handle_correct(x1, W1, beta, ret, 'accel')

            # prediction
            x1, W1 = self.funcs['predict_x_W'](self.x, self.W, omega,
                                      p['std_gyro'], p['sn_gyro_rw'], dt)
            self.handle_correct(x1, W1, 0, 0, 'predict')

            self.est_pub.put(msgs.EstimatorStatus(
                now, np.reshape(self.x, -1), np.reshape(self.W, -1)))


class KinematicSim(Node):

    def __init__(self, core, funcs):
        super().__init__(core)
        self.funcs = funcs

        # publications
        self.mag_pub = core.publish('mag', msgs.Mag)
        self.imu_pub = core.publish('imu', msgs.Imu)

        # initial state
        self.x = np.zeros(6)
        self.shadow = 0
        self.omega = np.array([1, 2, 3])

        # start processes
        core.process(self.propagate())
        core.process(self.mag_update())
        core.process(self.imu_update())

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
            yield self.core.timeout(dt)

    def mag_update(self):
        while True:
            p = self.core.params
            w = np.random.randn(3) * p['std_mag']
            y = self.funcs['measure_mag'](self.x, p['decl'], p['incl']) + w
            yield self.mag_pub.put(msgs.Mag(self.core.now, y))
            yield self.core.timeout(1.0/50)

    def imu_update(self):
        while True:
            p = self.core.params
            w_gyro = p['std_gyro']*np.random.randn(3)
            omega_bias = self.x[3:6]
            omega_meas = self.omega + omega_bias + w_gyro
            w_accel = np.random.randn(3) * p['std_accel']
            y = self.funcs['measure_accel'](self.x) + w_accel
            yield self.imu_pub.put(msgs.Imu(
                self.core.now, omega_meas, y))
            yield self.core.timeout(1.0/200)


class Logger(Node):

    def __init__(self, core, topics):
        super().__init__(core)
        self.subs = {}
        self.topics = topics

        def callback(topic):
            return lambda x: self.log_topic(str(topic), x)

        for topic in self.topics:
            self.subs[topic] = core.subscribe(topic, callback(topic))
        self.data = {}

    def log_topic(self, topic, msg):
        if topic not in self.data.keys():
            self.data[topic] = []
        self.data[topic].append(msg)

    def write(self):
        pd_data = {}
        for topic in self.data.keys():
            orb_type = self.core.get_type(topic)
            topic_data = self.data[topic]
            field_data = {}
            index = None
            for i_field, field in enumerate(orb_type._fields):
                vector_data = np.vstack([topic_data[j][i_field] for j in range(len(topic_data))])
                print('vector_data shape', vector_data.shape)
                if field == 't':
                    index = pd.Float64Index(data=np.array(vector_data[:, 0]), name='time, s')
                elif vector_data.shape[1] == 1:
                    field_data[field] = vector_data
                else:
                    for j in range(vector_data.shape[1]):
                        field_data[field + '_{:d}'.format(j)] = vector_data[:, j]
            pd_data[topic] = pd.DataFrame(field_data, index=index)
        return pd_data


def test_sim():
    # load simulation functions
    pkl_path = os.path.join(os.path.dirname(__file__), '../../examples/sim_funcs.pkl')
    with open(pkl_path, 'rb') as f:
        funcs = pickle.load(f)

    core = Core()
    core.params = {
        'std_mag': 2.5e-3,
        'decl': 0,
        'incl': 1,
        'beta_mag_c': 6.2,
        'std_accel': 35e-3,
        'std_accel_omega': 100e-3,
        'std_gyro': 1e-3,
        'sn_gyro_rw': 0.01e-3,
        'beta_accel_c': 9.6,
        'std_x0': [1, 1, 1, 0.1, 0.1, 0.1],
        'tf': 10,
    }
    sim = KinematicSim(core, funcs)
    sim.print_functions()
    estimator = Estimator(core, funcs)
    logger = Logger(core, ['mag', 'imu', 'estimator_status'])
    core.run(until=core.params['tf'])
    data = logger.write()


if __name__ == "__main__":
    test_sim()
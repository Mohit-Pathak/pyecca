from collections import namedtuple
from dataclasses import dataclass
import numpy as np
import typing


class Vector3(typing.NamedTuple):
    x: float
    y: float
    z: float


@dataclass
class Mag:
    timestamp: np.uint64
    magnetometer_ga: np.array(dtype=(np.float32, 3))


@dataclass
class Imu:
    timestamp: np.uint64
    gyroscope: np.float32[3]


@dataclass
class Quaternion(Msg):
    q: np.float32[4]


Mag = namedtuple('Mag', (
    't', # timestamp
    'm'  # magnetometer, ga, [3] axis
))

Imu = namedtuple('Imu', (
    't', # timestamp
    'g', # gyroscope, [3] axis, rad/s
    'a'  # acceleromter, [3] axis rad/s
))

Quaternion = namedtuple('Quaternion', (
    't', # timestamp
    'q'  # quaternion, [4]
))

EstimatorStatus = namedtuple('EstimatorStatus', (
    't', # timestamp
    'x', # estimator states [*]
    'W'  # estimator sqrt P [*]
))
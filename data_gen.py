import numpy as np
from random import random, uniform, normalvariate
import random as rd
import math

def plot_circle(N, R, centre=None, add_noise = False, sigma=0.01, seed = None):
    if seed is not None:
        rd.seed(seed)
    if centre is None:
        centre = [0, 0]
    X = np.zeros((N,2))
    label = np.zeros((N))
    for i in range(N):
        r = R*math.sqrt(random())
        if add_noise:
            r = r + normalvariate(0,sigma)
        theta = 2*math.pi*random()
        X[i,0] = centre[0] + r*math.cos(theta)
        X[i, 1] = centre[1] + r * math.sin(theta)
        label[i] = 0
    return X, label

def plot_two_circles(N,R1,centre1,R2,centre2, add_noise = False, sigma=0.01, seed = None):
    if seed is not None:
        rd.seed(seed)
    X = np.zeros((N, 2))
    label = np.zeros((N), 'int32')
    for i in range(int(N / 2)):
        r = R1 * math.sqrt(random())
        if add_noise:
            r = r + normalvariate(0,sigma)
        theta = 2 * math.pi * random()
        X[i, 0] = centre1[0] + r * math.cos(theta)
        X[i, 1] = centre1[1] + r * math.sin(theta)
        label[i] = 0
    for i in range(int(N / 2), N):
        r = R2 * math.sqrt(random())
        if add_noise:
            r = r + normalvariate(0,sigma)
        theta = 2 * math.pi * random()
        X[i, 0] = centre2[0] + r * math.cos(theta)
        X[i, 1] = centre2[1] + r * math.sin(theta)
        label[i] = 1
    return X, label

def circle_within_circle(N, R1, R2_inner, R2_outer, centre1=None, centre2=None, add_noise = False, sigma=0.01, seed = None):
    if seed is not None:
        rd.seed(seed)
    if centre1 is None:
        centre1 = [0, 0]
    if centre2 is None:
        centre2 = [0, 0]
    R2_thickness = R2_outer - R2_inner
    X = np.zeros((N, 2))
    label = np.zeros((N), 'int32')
    for i in range(int(N / 2)):
        r = R1 * math.sqrt(random())
        if add_noise:
            r = r + normalvariate(0,sigma)
        theta = 2 * math.pi * random()
        X[i, 0] = centre1[0] + r * math.cos(theta)
        X[i, 1] = centre1[1] + r * math.sin(theta)
        label[i] = 0
    for i in range(int(N / 2), N):
        f = R2_thickness / R2_outer
        r = R2_outer * (uniform(1 - f, 1))
        if add_noise:
            r = r + normalvariate(0,sigma)
        theta = 2 * math.pi * random()
        X[i, 0] = centre2[0] + r * math.cos(theta)
        X[i, 1] = centre2[1] + r * math.sin(theta)
        label[i] = 1
    return X, label

def half_moons(N,R_inner,R_outer,d, add_noise = False, sigma=0.01, seed = None):
    if seed is not None:
        rd.seed(seed)
    R_thickness = R_outer - R_inner
    X = np.zeros((N, 2))
    label = np.zeros((N), 'int32')
    for i in range(int(N / 2)):
        f = R_thickness / R_outer
        r = R_outer * (uniform(1 - f, 1))
        if add_noise:
            r = r + normalvariate(0,sigma)
        theta = math.pi * random()
        if add_noise:
            theta = theta + normalvariate(0,sigma)
        X[i, 0] = r * math.cos(theta)
        X[i, 1] = r * math.sin(theta)
        label[i] = 0
    for i in range(int(N / 2), N):
        f = R_thickness / R_outer
        r = R_outer * (uniform(1 - f, 1))
        if add_noise:
            r = r + normalvariate(0,sigma)
        theta = math.pi * random() + math.pi
        if add_noise:
            theta = theta + normalvariate(0, sigma)

        X[i, 0] = R_inner + R_thickness / 2 + r * math.cos(theta)
        X[i, 1] = -d + r * math.sin(theta)
        label[i] = 1
    np.random.seed(10)
    np.random.permutation(10)
    randomize = np.random.permutation(np.arange(N))
    X = X[randomize]
    label = label[randomize]
    return X, label

def curve_sep(N,hor_gap,add_noise=False,sigma=0.01, seed = None):
    if seed is not None:
        rd.seed(seed)
    X = np.zeros((N, 2))
    label = np.zeros((N), 'int32')
    count = 0
    while count < N:
        x = 2 * random()
        y = 2 * random()
        # d = (y - x ** 3)
        d = (x - y ** (1/3))
        if abs(d) > hor_gap:
            if add_noise:
                x = x + normalvariate(0, sigma)
                y = y + normalvariate(0, sigma)
            X[count, 0] = x
            X[count, 1] = y
            if d > 0:
                label[count] = 0
            else:
                label[count] = 1
            count = count + 1
    return X, label

def get_introductory_example_data():
    X, label = curve_sep(20, hor_gap=0, add_noise=False, sigma=0.08, seed=1)
    sorted_ind = np.argsort(X[:, 0])
    X = X[sorted_ind]
    label = label[sorted_ind]
    test = np.zeros((1, 2), 'float32')
    test[:, 0], test[:, 1] = 0.95, 1.25
    return X, label, test

def get_introductory_example_data_knn():
    X, label = curve_sep(20, hor_gap=0, add_noise=False, sigma=0.08, seed=1)
    sorted_ind = np.argsort(X[:, 0])
    X = X[sorted_ind]
    label = label[sorted_ind]
    test = np.zeros((1, 2), 'float32')
    test[:, 0], test[:, 1] = 1.1, 1.25
    return X, label, test
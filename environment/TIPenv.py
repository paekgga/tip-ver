import numpy as np
from math import sin, cos
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from copy import deepcopy

class TIPSystem:

    def __init__(self):
        self.render_activate = False
        self.L1 = 0.1645; self.L2 = 0.210; self.L3 = 0.245
        self.sample_time = 0.002
        self.frame_skip = 10
        self.control_frequency = self.sample_time * self.frame_skip
        self.rail_length = 1.0
        self.action_max = 18.0
        self.action_dim = 1
        self.n_history = 1
        self.state_dim = 11

    def step(self, a):
        self._do_simulation(a[0])
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[0]) <= self.rail_length/2)
        notdone = notdone and np.all(np.abs(ob[8:11])<27.)
        pos = (1 + np.exp(-(ob[0] ** 2) * np.log(10) / 4)) / 2
        act = (4 + np.maximum(1 - ((a[0]/self.action_max)**2), 0.0)) / 5
        angle1 = (1 + ob[4]) / 2
        angle2 = (1 + ob[5]) / 2
        angle3 = (1 + ob[6]) / 2
        angle = angle1 * angle2 * angle3
        vel1 = (1 + np.exp(-(ob[8] ** 2) * np.log(10) / 25)) / 2
        vel2 = (1 + np.exp(-(ob[9] ** 2) * np.log(10) / 25)) / 2
        vel3 = (1 + np.exp(-(ob[10] ** 2) * np.log(10) / 25)) / 2
        vel = np.minimum(vel1, np.minimum(vel2, vel3))
        r = np.array([angle * pos * act * vel]).reshape([-1])

        done = not notdone
        ob = np.reshape(ob, [1, -1])
        return ob, r, done, {}

    def reset(self):
        q = np.zeros(4)     #[x, th1, th2, th3]
        q[0] = 0.01*np.random.randn()
        q[1] = np.pi + .01*np.random.randn()
        q[2] = np.pi + .1*np.random.randn()
        q[3] = np.pi + .1*np.random.randn()
        qd = .01*np.random.randn(4)
        self.x = np.concatenate([q,qd])
        ob = self._get_obs()
        init_obs = np.reshape(ob, [1, -1])
        return init_obs

    def _do_simulation(self, a):
        for i in range(self.frame_skip):
            acc = self._v2a(a)
            xd1 = self._derivative(self.x, acc)
            xd2 = self._derivative(self.x + (self.sample_time/2)*xd1, acc)
            xd3 = self._derivative(self.x + (self.sample_time/2)*xd2, acc)
            xd4 = self._derivative(self.x + self.sample_time*xd3, acc)
            xd = (xd1 + 2*xd2 + 2*xd3 + xd4)/6
            self.x += self.sample_time*xd

    def _get_obs(self):
        return np.array([self.x[0],
                         sin(self.x[1]), sin(self.x[2]-self.x[1]), sin(self.x[3]-self.x[2]),
                         cos(self.x[1]), cos(self.x[2]-self.x[1]), cos(self.x[3]-self.x[2]),
                         self.x[4], self.x[5], self.x[6]-self.x[5], self.x[7]-self.x[6]])

    def _derivative(self, x, a):

        l1 = self.L1; l2 = self.L2
        a1 = 0.077792803144923; a2 = 0.118304871010206; a3 = 0.142848166462599
        m1 = 0.209031904484509; m2 = 0.114999198087930; m3 = 0.146790233805106
        J1 = 0.001145259713644; J2 = 0.0008050827698029238; J3 = 0.001600887002439
        d1 = 0.001547562611052; d2 = 0.0008598850094355782; d3 = 0.0006480318611403285
        g = 9.81

        q, qd = x[:4], x[4:]

        m11 = J1 + (a1**2)*m1 + (l1**2)*(m2 + m3)
        m12 = a2*l1*m2*cos(q[1]-q[2]) + l1*l2*m3*cos(q[1]-q[2])
        m13 = a3*l1*m3*cos(q[1]-q[3])
        m21 = a2*l1*m2*cos(q[1]-q[2]) + l1*l2*m3*cos(q[1]-q[2])
        m22 = J2 + (a2**2)*m2 + (l2**2)*m3
        m23 = a3*l2*m3*cos(q[2]-q[3])
        m31 = a3*l1*m3*cos(q[1]-q[3])
        m32 = a3*l2*m3*cos(q[2]-q[3])
        m33 = J3 + (a3**2)*m3
        M = np.array([[m11,m12,m13],
                      [m21,m22,m23],
                      [m31,m32,m33]])

        c11 = c22 = c33 = 0
        c12 = qd[2]*a2*l1*m2*sin(q[1]-q[2]) + qd[2]*l1*l2*m3*sin(q[1]-q[2])
        c13 = qd[3]*a3*l1*m3*sin(q[1]-q[3])
        c21 = -qd[1]*a2*l1*m2*sin(q[1]-q[2]) - qd[1]*l1*l2*m3*sin(q[1]-q[2])
        c23 = qd[3]*a3*l2*m3*sin(q[2]-q[3])
        c31 = -qd[1]*a3*l1*m3*sin(q[1]-q[3])
        c32 = -qd[2]*a3*l2*m3*sin(q[2]-q[3])
        C = np.array([[c11,c12,c13],
                      [c21,c22,c23],
                      [c31,c32,c33]])

        d11 = d1 + d2
        d12 = -d2
        d13 = 0
        d21 = -d2
        d22 = d2 + d3
        d23 = -d3
        d31 = 0
        d32 = -d3
        d33 = d3
        D = np.array([[d11, d12, d13],
                      [d21, d22, d23],
                      [d31, d32, d33]])

        g11 = -g*(a1*m1 + l1*m2 + l1*m3)*sin(q[1])
        g21 = -g*(a2*m2 + l2*m3)*sin(q[2])
        g31 = -a3*g*m3*sin(q[3])
        G = np.array([g11,g21,g31]).T

        b11 = a1*m2*cos(q[1]) + l1*m2*cos(q[1]) + l1*m3*cos(q[1])
        b21 = a2*m2*cos(q[2]) + l2*m3*cos(q[2])
        b31 = a3*m3*cos(q[3])
        B = np.array([b11,b21,b31]).T

        Minv = np.linalg.inv(M)
        F0 = -Minv @ (C @ qd[1:].T + D @ qd[1:].T + G)
        F1 = Minv @ B

        fx = np.concatenate([qd, np.zeros(1), F0.T])
        gx = np.concatenate([np.zeros(4),np.ones(1),F1.T])
        xd = fx + gx * a
        return xd

    def _v2a(self, v):
        xdot = deepcopy(self.x[-4])
        k = 0.053
        J = 0.000023896
        R = 1.3
        r = 0.0051
        b = 0.00035
        acc = (k*r*v - (k*k+b*R)*xdot)/(J*R)
        return acc

    def _render_tool(self):
        self.render_activate = True
        fig, ax = plt.subplots(figsize=(6,4))
        plt.grid()
        ax.set_xlim((-1.2, 1.2)); ax.set_ylim((-1.0, 1.0))
        # create line objects
        self.line0, = ax.plot([], [], lw=2,c='k')
        self.line1, = ax.plot([], [], lw=4,c='k')
        self.line2, = ax.plot([], [], lw=3,c='k')
        self.line3, = ax.plot([], [], lw=3,c='k')
        self.line4, = ax.plot([], [], lw=3,c='k')
        self.line0.set_data(np.linspace(-self.rail_length / 2, self.rail_length / 2, 100), 0 * np.linspace(-2, 2, 100))
        # create joint objects
        self.c1 = Circle((0, 0), radius=0.016, color='k')
        self.c2 = Circle((0, 0), radius=0.016, color='k')
        ax.add_patch(self.c1)
        ax.add_patch(self.c2)
        plt.show(block=False)
        plt.pause(0.00001)

    def render(self):
        if self.render_activate == False: self._render_tool()
        p0 = np.array([self.x[0],0])
        p1 = p0 + np.array([-self.L1*sin(self.x[1]), self.L1*cos(self.x[1])])
        p2 = p1 + np.array([-self.L2*sin(self.x[2]), self.L2*cos(self.x[2])])
        p3 = p2 + np.array([-self.L3*sin(self.x[3]), self.L3*cos(self.x[3])])
        # update line data
        self.line1.set_data([p0[0]-0.05, p0[0]+0.05],[p0[1], p0[1]])
        self.line2.set_data([p0[0], p1[0]],[p0[1], p1[1]])
        self.line3.set_data([p1[0], p2[0]],[p1[1], p2[1]])
        self.line4.set_data([p2[0], p3[0]],[p2[1], p3[1]])
        # update circle data
        self.c1.center = (p1[0],p1[1])
        self.c2.center = (p2[0], p2[1])
        plt.show(block=False)
        plt.pause(0.000001)

    def close(self):
        pass
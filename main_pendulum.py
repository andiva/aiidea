import numpy as np
import matplotlib.pyplot as plt
import physical_models as phys


def main():
    T = 16
    # generate data
    # train data (oscillation with initial angle = pi/6):
    t, train = phys.simulate_pendulum(0, np.array([np.pi/6, 0]), T, dt=0.1)
    # test data (oscillation with initial angle = pi/4):
    _, test1 = phys.simulate_pendulum(0, np.array([np.pi/4, 0]), T, dt=0.1)
    # test data (fixed point with angle = 0):
    _, test2 = phys.simulate_pendulum(0, np.array([0.0, 0.0]), T, dt=0.1)

    plt.plot(t, train[:,0], 'b-')
    plt.plot(t, test1[:,0], 'g-')
    plt.plot(t, test2[:,0], 'g-')

    plt.show()

    return 0

if __name__ == '__main__':
    main()
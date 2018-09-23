import numpy as np
import matplotlib.pyplot as plt
from cntk import load_model

import physical_models as phys
import models


def plot(title, train, test1, test2, pred, pred1, pred2):
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle(title, fontsize=13)
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.set_title('State Space')
    plt.ylabel('Training data')
    plt.plot(train[:, 0], 'b-')
    plt.plot(pred[:, 0], 'r-')

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.set_title('Phase Space')
    plt.plot(train[:, 0], train[:, 1], 'b-')
    plt.plot(pred[:, 0], pred[:, 1], 'r-')
    
    ax3 = fig.add_subplot(3, 2, 3)
    plt.ylabel('Test1 and Prediction1')  
    plt.plot(test1[:, 0], 'g-')
    plt.plot(pred1[:, 0], 'r-')

    ax4 = fig.add_subplot(3, 2, 4)
    plt.plot(test1[:, 0], test1[:, 1], 'g-')
    plt.plot(pred1[:, 0], pred1[:, 1], 'r-')

    ax5 = fig.add_subplot(3, 2, 5)
    plt.ylabel('Test2 and Prediction2')
    plt.plot(test2[:, 0], 'g-')
    plt.plot(pred2[:, 0], 'r-')

    ax6 = fig.add_subplot(3, 2, 6) 
    plt.plot(test2[:, 0], test2[:, 1], 'g-')
    plt.plot(pred2[:, 0], pred2[:, 1], 'r-')

    for ax in plt.gcf().get_axes():
        ax.grid() 


    plt.show()
    return 0


def main():
    # generate data from physical model
    T = 16 # time interval
    dt = 0.1 # integration step
    # train data (oscillation with initial angle = pi/6):
    t, train = phys.simulate_pendulum(0, np.array([np.pi/6, 0]), T, dt=dt)
    # test data (oscillation with initial angle = pi/4):
    _, test1 = phys.simulate_pendulum(0, np.array([np.pi/4, 0]), T, dt=dt)
    # test data (fixed point with angle = 0):
    _, test2 = phys.simulate_pendulum(0, np.array([0.0, 0.0]), T, dt=dt)

    
    N = 5 # size of the history window (for LSTM)

    # # train lstm model
    model_lstm = models.train_lstm(train, N)
    model_lstm.save('models/pendulum_lstm')
    # # or load pre-built one
    # model_lstm = load_model('models/pendulum_lstm')

    # use lstm for prediction with training initial angle = pi/6
    pred = models.predict(model_lstm, train[:N], step_count=int(T/dt-N), N=N)
    # use lstm for prediction with initial angle = pi/4
    pred1 = models.predict(model_lstm, test1[:N], step_count=int(T/dt-N), N=N)
    # use lstm for prediction with initial angle = 0
    pred2 = models.predict(model_lstm, test2[:N], step_count=int(T/dt-N), N=N)

    plot("LSTM: non-physical predictions, training data are memorized", train, test1, test2, pred, pred1, pred2)
    

    return 0

if __name__ == '__main__':
    main()
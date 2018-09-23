import numpy as np


def integrate_rk4(f, t0, x0, t, dt = 0.1):
    """Numerically integrates ODE.

    Arguments:
    f -- right-hand side of the system of ODE
    t0 -- initial time
    x0 -- initial state
    t -- end of the time interval
    dt -- integration step
    """
    xi=x0.copy()
    times = np.arange(t0, t, step=dt)
    x = np.empty((len(times), len(x0)))
    for i, time in enumerate(times):
        k1 = f(time, xi)
        k2 = f(time+dt/2.0, xi+dt*k1/2.0)
        k3 = f(time+dt/2.0, xi+dt*k2/2.0)
        k4 = f(time+dt, xi+dt*k3)

        xi += dt*(k1+2*k2+2*k3+k4)/6.0  
        x[i] = xi

    return times+dt, np.array(x)


def pendulum_right_hand(t, x, l=15, g=9.8):
    """ ODE for a simple pendulum
    """
    theta, omega = x[0], x[1]
    return np.array([omega, -g*np.sin(theta)/l])


def simulate_pendulum(t0, x0, t, dt = 0.1):
    return integrate_rk4(pendulum_right_hand, t0, x0, t, dt)


def SIR_right_hand(t, x, b=0.5, g=0.1, k=6):
    """ ODE for a simple SIR model
        https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model_is_dynamic_in_three_senses
    """
    S, I, R = x[0], x[1], x[2]
    return k*np.array([-b*I*S,
                        b*I*S - g*I,
                        g*I])


def simulate_epidemiology(t0, x0, t, dt = 0.1):
    return integrate_rk4(SIR_right_hand, t0, x0, t, dt)
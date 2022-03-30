from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import math

#constants
P = 60*10**(-4) #cm/min
dc = 6*10**(-3) #cell density 10^5cells/ml x10^-6 to account for units
Vmax = 0.28 #rate cst for cell membrane trasnport ng/10^5 cells/min
Ke = 0.219*10**(-3) #Michaelis constant mg/ml
Ki = 1.37 # also mich constant (ng/10^5 cells)
St = 200 # vascular density (cm^-1)
A = 0.13*10**3 # L-1 inverse volume of patient
t_half = 4.93 #half life of doxorubicin in  blood
alpha = 0.693/t_half # time constant for DR decay in blood 
D = 80 # Dose (mg) can go up to 285 mg
T = 8*60 #infusion time (minutes)
phi = 0.4 #volume fraction of extracellular space in tumor 
#cv = 0.0006#temporary constant -> make variable from first model

def integrate(y_vals, h):
    i = 1
    total = y_vals[0] + y_vals[-1]
    for y in y_vals[1:-1]:
        if i % 2 == 0:
            total += 2 * y
        else:
            total += 4 * y
        i += 1
    return total * (h / 3.0)

# Runge-Kutta (RK4) Numerical Integration for System of First-Order Differential Equations
def ode_system(_t, _y):
    """
    system of first order differential equations
    _t: discrete time step value
    _y: state vector [ce, ci]
    """
    ce = _y[0]
    ci = _y[1]

    termA = (ce/float(ce+Ke*phi)) - (ci/float(ci+Ki))
    #print(termA)
    ce_new = P*St*(cv-ce)-dc*Vmax*termA   #mg/ml/min = ng/105cells/min* cells/ml *10^-6mg/ng*10^-5
    ci_new = Vmax*termA #ng/105cells/min  
    return np.array([ce_new,ci_new])


def rk4(func, tk, _yk, _dt=0.1, **kwargs):
    """
    single-step fourth-order numerical integration (RK4) method
    func: system of first order ODEs
    tk: current time step
    _yk: current state vector [y1, y2, y3, ...]
    _dt: discrete time step size
    **kwargs: additional parameters for ODE system
    returns: y evaluated at time k+1
    """

    # evaluate derivative at several stages within time interval
    f1 = func(tk, _yk, **kwargs)
    f2 = func(tk + _dt / 2, _yk + (f1 * (_dt / 2)), **kwargs)
    f3 = func(tk + _dt / 2, _yk + (f2 * (_dt / 2)), **kwargs)
    f4 = func(tk + _dt, _yk + (f3 * _dt), **kwargs)

    # return an average of the derivative over tk, tk + dt
    return _yk + (_dt / 6) * (f1 + (2 * f2) + (2 * f3) + f4)


# ==============================================================
# simulation harness

dt = 0.01
time = np.arange(0, 60*24 + dt, dt)

# second order system initial conditions [y1, y2] at t = 1
y0 = np.array([0, 0])

# ==============================================================
# propagate state

# simulation results
state_history = []

# initialize yk
yk = y0

# intialize time
t = 0
cv_saved = []
# approximate y at time t
for t in time:
    cv = (D/T)*(A/alpha)*(1-math.exp(-alpha*t))
    if(t>T):
        cv = (D/T)*(A/alpha)*(math.exp(alpha*T)-1)*math.exp(-alpha*(t))
    cv_saved.append(cv)
    state_history.append(yk)
    yk = rk4(ode_system, t, yk, dt)

# convert list to numpy array
state_history = np.array(state_history)

print(f'y evaluated at time t = {t} seconds: {yk[0]}')


print(f'AUC = {integrate(state_history[:,1], dt)}')
# ==============================================================
# plot history

fig, ax = plt.subplots()
ax.plot(time, state_history[:, 0], color ='tab:blue', label='Ce')
ax.set_xlabel('t(min)')
ax.set_ylabel('Ce (mg/ml)', color='tab:blue')
ax2 = ax.twinx()
ax2.plot(time, state_history[:, 1], color='darkgreen', label='Ci')
ax2.set_ylabel('Ci (ng/10^5 cells)', color='darkgreen')
for label in ax2.get_yticklabels():
    label.set_color('darkgreen')
plt.show()

plt.plot(time, cv_saved)
plt.show()

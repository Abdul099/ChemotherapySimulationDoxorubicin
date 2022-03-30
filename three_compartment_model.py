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
A = 0.13*10**-3 # ml-1 inverse volume of patient
t_half = 4.93 #half life of doxorubicin in  blood
alpha = 0.693/t_half # time constant for DR decay in blood 
D = 80 # Dose (mg) can go up to 285 mg
T = 8*60 #infusion time (minutes)
phi = 0.4 #volume fraction of extracellular space in tumor 

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
# ==============================================================
def simulate(infusion_time, simulation_time, Dose = 80, plot=True, title=""):
    # Runge-Kutta (RK4) Numerical Integration for System of First-Order Differential Equations
    def ode_system(_t, _y):
        ce = _y[0]
        ci = _y[1]
        termA = (ce/float(ce+Ke*phi)) - (ci/float(ci+Ki))
        ce_new = P*St*(cv-ce)-dc*Vmax*termA   #mg/ml/min = ng/105cells/min* cells/ml *10^-6mg/ng*10^-5
        ci_new = Vmax*termA #ng/105cells/min  
        return np.array([ce_new,ci_new])

    def rk4(func, tk, _yk, _dt=0.1, **kwargs):
        # evaluate derivative at several stages within time interval
        f1 = func(tk, _yk, **kwargs)
        f2 = func(tk + _dt / 2, _yk + (f1 * (_dt / 2)), **kwargs)
        f3 = func(tk + _dt / 2, _yk + (f2 * (_dt / 2)), **kwargs)
        f4 = func(tk + _dt, _yk + (f3 * _dt), **kwargs)
        # return an average of the derivative over tk, tk + dt
        return _yk + (_dt / 6) * (f1 + (2 * f2) + (2 * f3) + f4)
    # ==============================================================
    dt = 0.01
    time = np.arange(0, simulation_time + dt, dt)
    y0 = np.array([0, 0])# initial conditions  at t = 0
    state_history = []# simulation results
    # initialize yk
    yk = y0
    # intialize time
    t = 0
    cv_saved = []
    # approximate y at time t
    for t in time:
        cv = (Dose/infusion_time)*(A/alpha)*(1-math.exp(-alpha*t))
        if(t>infusion_time):
            cv = (Dose/infusion_time)*(A/alpha)*(math.exp(alpha*infusion_time)-1)*math.exp(-alpha*(t))
        cv_saved.append(cv)
        state_history.append(yk)
        yk = rk4(ode_system, t, yk, dt)
    # convert list to numpy array
    state_history = np.array(state_history)
    #calculate AUC
    AUC = round(integrate(state_history[:,1], dt), 2)
    print(f'AUC = {AUC}')
    # ==============================================================
    # plot history
    if plot:
        fig, ax = plt.subplots()
        ax.plot(time, state_history[:, 0], color ='tab:blue', label='Ce')
        ax.set_xlabel('t(min)')
        ax.set_ylabel('Ce (mg/ml)', color='tab:blue')
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax2.plot(time, state_history[:, 1], color='darkgreen', label='Ci')
        ax2.set_ylabel('Ci (ng/10^5 cells)', color='darkgreen')
        ax3.plot(time, cv_saved, color='tab:red', label='Cv')
        ax3.get_yaxis().set_ticks([])
        for label in ax2.get_yticklabels():
            label.set_color('darkgreen')
        ax.legend(loc='center right')
        ax2.legend(loc='upper right')
        ax3.legend(loc='lower right')
        plt.title(title + f'\nAUC = {round(AUC,2)}')
        plt.show()
    return state_history, cv_saved, time

simulate(1, 30, Dose=80, title='Concentration Profiles for D=80mg and 1 min Infusion' )
simulate(8*60, 24*60, Dose=80, title='Concentration Profiles for D=80mg and 8 hours Infusion' )

infusion_times = [0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] #in hours
Doses = [50, 150, 250, 350]
dictionary = {}
for Dose in Doses:
    dictionary[Dose] = []
plt.figure()
for i in infusion_times:
    for Dose in Doses:
        state_history, _, _ = simulate(i*60, 18*60, Dose=Dose, plot=False)
        peak = max(state_history[:,1])
        print(peak)
        dictionary[Dose].append(peak)
optimal_times = []
for Dose in Doses:
    plt.plot(infusion_times, dictionary[Dose], label=f'{Dose}mg')
    optimal_times.append(infusion_times[np.argmax(dictionary[Dose])])
plt.legend()
plt.title("Peak Intracellular Concentration")
plt.xlabel("Infusion Duration")
plt.ylabel("Peak Concentration(ng/10^5cells)")
plt.show()

plt.plot(Doses, optimal_times)
plt.title("Optimal Infusion Time for Different Doses")
plt.xlabel("Dose (mg)")
plt.ylabel("Optimal Infusion Time (hrs)")
plt.show()
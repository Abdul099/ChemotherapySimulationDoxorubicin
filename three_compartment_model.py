import numpy as np
import matplotlib.pyplot as plt
import math

#constants
dc = 6*10**(-3) #cell density 10^5cells/ml x10^-6 to account for units
St = 200 # vascular density (cm^-1)
A = 0.13*10**-3 # ml-1 inverse volume of patient
phi = 0.4 #volume fraction of extracellular space in tumor 
# ==============================================================
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

def square_plot(low_T, high_T, low_D, high_D, title=""):
    plt.figure()
    plt.axvline(x=low_T)
    plt.axvline(x=high_T)
    plt.axhline(y=low_D)
    plt.axhline(y=high_D)
    plt.xlim([0, 6])
    plt.ylim([0, high_D*2])
    plt.xlabel("Infusion Time (Hours)")
    plt.ylabel("Total Dosage (mg)")
    plt.title(title)
    plt.savefig(f'./saved_graphs/{title}.png')
    #plt.show()
# ==============================================================
def simulate(infusion_time, simulation_time, Dose = 80, plot=True, title="", Ke = 0.219*10**(-3), Ki=1.37, Vmax = 0.28, P = 60*10**(-4), t_half = 4.75):
    '''
    infusion_time: (mins) duration at which continous injection is made starting t=0 
    simulaition_time: (mins) duration of simulation 
    Dose: (mg) Total Dose of Chemotherapy Session 
    plot: (boolean) plots concentration vs time graph for cv, ce, and ci
    title: (string) title of the concentration time graphs
    ke: (mg/ml) michaelis constant
    ki: (ng/10^5 cells) other michaelis constant 
    Vmax: (ng/10^5 cells) rate cst for cell membrane trasnport
    P: (cm/min) cell membrane permeability
    t_half: (min) half life of the drug in plasma
    '''

    # Runge-Kutta (RK4) Numerical Integration for System of 1st-Order ODEs
    def ode_system(_t, _y):
        ce = _y[0]
        ci = _y[1]
        termA = (ce/float(ce+Ke*phi)) - (ci/float(ci+Ki))
        ce_new = P*St*(cv-ce)-dc*Vmax*termA   #mg/ml/min 
        ci_new = Vmax*termA #ng/105cells/min  
        return np.array([ce_new,ci_new])

    def rk4(func, tk, _yk, _dt=0.01, **kwargs):
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
        alpha = 0.693/t_half # time constant for DR decay in blood 
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
    if plot:
        fig, ax = plt.subplots()
        ax.plot(time, state_history[:, 0], color ='tab:blue', label='Ce')
        ax.set_xlabel('t(min)')
        ax.set_ylabel('Ce & Cv (mg/ml)', color='tab:blue')
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax2.plot(time, state_history[:, 1], color='darkgreen', label='Ci')
        ax2.set_ylabel('Ci (ng/10\u2075cells)', color='darkgreen')
        ax3.plot(time, cv_saved, color='tab:red', label='Cv')
        ax3.get_yaxis().set_ticks([])
        for label in ax2.get_yticklabels():
            label.set_color('darkgreen')
        ax.legend(loc='center right')
        ax2.legend(loc='upper right')
        ax3.legend(loc='lower right')
        plt.autoscale()
        plt.title(title + f'\nAUC = {round(AUC,2)}')
        plt.savefig(f'./saved_graphs/{title}.png')
        #plt.show()
    return state_history, cv_saved, time
# ==============================================================
def find_peak_intracellular_concentration(Doses, infusion_times, sim_length=10*60, plot=True, title="", Ke = 0.219*10**(-3), Ki=1.37, Vmax = 0.28, P = 60*10**(-4), cutoff=3.5, t_half=4.75):
        dictionary = {}
        for Dose in Doses:
            dictionary[Dose] = []
        plt.figure()
        Ki_chosen=Ki
        t_half_chosen = t_half
        for i in infusion_times:
            for j, Dose in enumerate(Doses):
                if isinstance(Ki, list):
                    Ki_chosen = Ki[j]
                if isinstance(t_half, list):
                    t_half_chosen = t_half[j]
                state_history, _, _ = simulate(i*60, sim_length, Dose=Dose, plot=False, Ki=Ki_chosen, Ke=Ke, Vmax=Vmax, P=P, t_half=t_half_chosen)
                peak = max(state_history[:,1])
                print(peak)
                dictionary[Dose].append(peak)
        optimal_times = []
        plt.figure()
        for Dose in Doses:
            plt.plot(infusion_times, dictionary[Dose], label=f'{Dose}mg')
            optimal_times.append(infusion_times[np.argmax(dictionary[Dose])])
        plt.xlabel("Infusion Duration (h)")
        plt.ylabel("Peak Concentration(ng/10\u2075cells)")
        plt.axhline(y=cutoff, linestyle='--', label='lethal concentration')
        plt.legend()
        plt.autoscale()
        if title == "":
            plt.title(f'Peak Intracellular Concentration  for Ki {Ki} and Ke {Ke}')
        else:
            plt.title(title)
            plt.savefig(f'./saved_graphs/{title}.png')
        #plt.show()
        # plt.figure()
        # plt.plot(Doses, optimal_times)
        # plt.title(f'Optimal Infusion Time for Different Doses for Ki {Ki} and Ke {Ke}')
        # plt.xlabel("Dose (mg)")
        # plt.ylabel("Optimal Infusion Time (hrs)")
        # plt.savefig(f'./saved_graphs/Optimal Infusion Time for Ki {Ki} and Ke {Ke}.png')
        # plt.show()
# ==============================================================
#Driver Code

simulate(2, 40, Dose=80, title='Concentration Profiles for D=80mg and 2 min Infusion')
simulate(8*60, 12*60, Dose=80, title='Concentration Profiles for D=80mg and 8 hours Infusion' )
infusion_times = [0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8] #in hours
Doses = [50, 150, 250, 350]
find_peak_intracellular_concentration(Doses, infusion_times,Ke = 0.219*10**(-3), Ki=1.37, Vmax = 0.28, title='Peak Concentrations For Different Doses')
Doses = [40, 45, 50] 
find_peak_intracellular_concentration(Doses, infusion_times,Ke = 0.219*10**(-3), Ki=1.37, Vmax = 0.28, title='Minimum Dose to Kill Cancer Cells')#D LOW = 45mg, TLow = 1hr
Doses = [350, 360, 375]
find_peak_intracellular_concentration(Doses, infusion_times,Ke = 0.219*10**(-3), Ki=1.37/5, Vmax = 0.28, title = 'Maximum Dose that Starts Killing Healthy Cells')#D HIGH =360mg , T HIGH =3hrs
square_plot(low_T=1, high_T=3, low_D=45, high_D=360, title="Optimal Dosage and Infusion Time Range Conventional Treatment")

infusion_times = [0.1, 0.25, 0.5, 1, 1.5, 2, 10] #in hours
Doses = [0.5, 1, 2]
find_peak_intracellular_concentration(Doses, infusion_times,Ke = 0.219*10**(-3), Ki=2*1.37, Vmax = 0.28, t_half=60*45, P = 100*60*10**(-4), sim_length=30*60, title = 'Minimum Dose to Kill Cancer Cells New Treatment')
Doses = [9, 10, 11]
find_peak_intracellular_concentration(Doses, infusion_times,Ke = 0.219*10**(-3), Ki=1.37/5, Vmax = 0.28, t_half=60*45, P = 60*10**(-4), sim_length=30*60, title = 'Maximum Dose that Starts Killing Healthy Cells New Treatment')
square_plot(low_T=0, high_T=0, low_D=1, high_D=9, title="Optimal Dosage and Infusion Time Range - New Treatment")

#final time concentration simulation
simulate(1, 30*60, Dose = 1,Ke = 0.219*10**(-3), Ki=1.37*2, Vmax = 0.28, t_half=60*45, P = 100*60*10**(-4), title='Simulation Using New Treatment - Cancer Cells')
simulate(1, 30*60, Dose = 9,Ke = 0.219*10**(-3), Ki=1.37/5, Vmax = 0.28, t_half=60*45, P = 60*10**(-4), title='Simulation Using New Treatment - Healthy Cells')

'''
TO DO:

Basic Model Sim DONE
Different Ki's Peak IC DONE
Peter's graph for survival DONE
Revisit Ki slide DONE
Graphs with recomended treatment params --> square graph (optional figure out exact shape) DONE
Ultrasound: increase P:
Liposomes: Increase Half life
Final Plot
summary for ci vs t graphs at end 
'''
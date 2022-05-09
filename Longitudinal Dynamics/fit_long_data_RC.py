import matplotlib.pyplot as plt
import pyomo.environ as pyo
import pandas as pd
import numpy as np

# .20 Trial 2 has bad data
bad_trial_index = 4
file_names = [
    '.15_Trial_1.xlsx',
    '.15_Trial_2.xlsx',
    '.15_Trial_3.xlsx',
    '.20_Trial_1.xlsx',
    '.20_Trial_2.xlsx',
    '.20_Trial_3.xlsx',
    '.25_Trial_1.xlsx',
    '.25_Trial_2.xlsx',
    '.25_Trial_3.xlsx',
    '.30_Trial_1.xlsx',
    '.30_Trial_2.xlsx',
    '.30_Trial_3.xlsx',
]
mass = 1.5  # kg


def getStartTime(motor_input):
    if motor_input == .15:
        start_time = .3
    elif motor_input == .20:
        start_time = .39
    elif motor_input == .25:
        start_time = .35
    else:
        start_time = .5
    return start_time


def getEndTime(motor_input):
    if motor_input == .15:
        end_time = 18
    elif motor_input == .20:
        end_time = 5.5
    elif motor_input == .25:
        end_time = 3.5
    else:
        end_time = 3.0
    return end_time


def plotData(b_arr, F_f_arr, C_d_arr):
    trials = []
    for file in file_names:
        trials.append(pd.read_excel(file))

    for i in range(4):
        fig = plt.figure(i)
        trial_speed = (i * 5) / 100 + .15
        for k in range(3):
            trial_index = (i * 3) + k
            if trial_index != bad_trial_index:
                trial = trials[trial_index]
                time_list = trial['time'].values
                # plt.plot(time_list, trial['vx'].values, 'b')
                # plt.plot(time_list, trial['vy'].values, 'r')
                plt.plot(time_list, trial['vz'].values, 'g')
                plt.ylabel('Velocity [m/s]')
                plt.xlabel('Time [s]')

        if i == 0:
            start_index = 0
            end_index = 3
        elif i == 1:
            start_index = 3
            end_index = 5
        elif i == 2:
            start_index = 5
            end_index = 8
        else:
            start_index = 8
            end_index = 11

        b = np.average(b_arr[start_index: end_index])
        F_f = np.average(F_f_arr[start_index: end_index])
        C_d = np.average(C_d_arr[start_index: end_index])

        vel_model, time_arr = velocityModel(b, F_f, C_d, trial_speed)
        print(vel_model[-1])
        plt.plot(time_arr, vel_model, '--', color='k')
        fig.legend(["Vz", "Optimizer Fit"])
        # fig.legend(["Vx", "Vy", "Vz", "Fitted Model"])

        fig.suptitle(str(trial_speed) + ' Trials', fontsize=16)
    plt.show()


def runOptimizer(file_name, motor_input):
    data = pd.read_excel(f'ConstantInputData/{file_name}')

    # Make sure to change label below for correct velocity data
    v = data['vz'].values
    t_data = data['time'].values
    n = len(t_data)

    model = pyo.ConcreteModel()
    # Model Variables
    model.z1 = pyo.Var()  # z1 - Motor input (b),
    model.z2 = pyo.Var()  # z2 - force of friction (F_f),
    model.z3 = pyo.Var()  # z3 - coefficient of drag (C_d)

    model.cost = pyo.Objective(expr=sum(((v[t + 1] - v[t]) - (t_data[t + 1] - t_data[t])
                                        * ((2 / mass) * (model.z1 * motor_input - model.z2 - model.z3 * v[t]**2)))**2
                                        for t in range(n + 1) if t < n - 1), sense=pyo.minimize)

    model.Constraint1 = pyo.Constraint(expr=model.z1 >= 0)
    model.Constraint2 = pyo.Constraint(expr=model.z2 >= 0)
    model.Constraint3 = pyo.Constraint(expr=model.z3 >= 0)
    model.Constraint4 = pyo.Constraint(expr=model.z1 <= 200)

    results = pyo.SolverFactory('ipopt').solve(model).write()
    b = pyo.value(model.z1)
    F_f = pyo.value(model.z2)
    C_d = pyo.value(model.z3)

    return b, F_f, C_d


def velocityModel(b, F_f, C_d, motor_input):
    print("--- Averages ---")
    print(motor_input)
    print(b)
    print(F_f)
    print(C_d)
    end_time = getEndTime(motor_input)
    n = 100
    v0 = 0
    t_arr = np.linspace(getStartTime(motor_input), end_time, n)
    delta_t = end_time / n
    velocity = [v0]
    for i in range(n):
        v_next = delta_t * (b * motor_input - F_f - C_d * velocity[i]**2) / mass + velocity[i]
        velocity.append(v_next)
    return velocity, np.append(t_arr, t_arr[-1] + int(end_time) / 100)


def getForceParameters():
    b_arr = []
    F_f_arr = []
    C_d_arr = []
    for i in range(len(file_names)):
        if i != bad_trial_index:
            if i < 3:
                b, F_f, C_d = runOptimizer(file_names[i], .15)
            elif i < 6:
                b, F_f, C_d = runOptimizer(file_names[i], .20)
            elif i < 9:
                b, F_f, C_d = runOptimizer(file_names[i], .25)
            else:
                b, F_f, C_d = runOptimizer(file_names[i], .30)
            b_arr.append(b)
            F_f_arr.append(F_f)
            C_d_arr.append(C_d)

    print('--- Parameters ---')
    print(b_arr)
    print(F_f_arr)
    print(C_d_arr)

    # print('--- Lengths ---')
    # print(len(b_arr))
    # print(len(F_f_arr))
    # print(len(C_d_arr))

    # print('--- Averages ---')
    # print(np.average(b_arr))
    # print(np.average(F_f_arr))
    # print(np.average(C_d_arr))

    # print('--- Std Deviations ---')
    # print(np.std(b_arr))
    # print(np.std(F_f_arr))
    # print(np.std(C_d_arr))

    return b_arr, F_f_arr, C_d_arr


plotData(*getForceParameters())
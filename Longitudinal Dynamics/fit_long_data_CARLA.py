import matplotlib.pyplot as plt
import pyomo.environ as pyo
import pandas as pd
import numpy as np

carla_files = [
    '.20 CARLA Trial.xlsx',
    '.30 CARLA Trial.xlsx',
    '.40 CARLA Trial.xlsx',
    '.50 CARLA Trial.xlsx',
    '.60 CARLA Trial.xlsx',
    '.70 CARLA Trial.xlsx',
    '.80 CARLA Trial.xlsx',
    '.90 CARLA Trial.xlsx',
    '1.0 CARLA Trial.xlsx',
]
mass = 1845  # kg


def plotData(b_arr, F_f_arr, C_d_arr):
    trials = []
    for file in carla_files:
        trials.append(pd.read_excel(f'CARLAData/{file}'))
    print('--- Final Velocities ---')
    for i in range(len(trials)):
        fig = plt.figure(i)
        trial_speed = round(i * .1 + .2, 1)
        trial = trials[i]
        time_data = trial['time'].values

        b = b_arr[i]
        F_f = F_f_arr[i]
        C_d = C_d_arr[i]

        vel_model, time_fit = velocityModel(b, F_f, C_d, trial_speed, time_data[0])
        print(vel_model[-1])

        plt.plot(time_data, trial['vz'].values, 'g', label='Vz Data')
        plt.plot(time_fit, vel_model, '--', color='k', label='Optimizer Fit')
        plt.ylabel('Velocity [m/s]')
        plt.xlabel('Time [s]')
        fig.legend()
        fig.suptitle(str(trial_speed) + ' Trial', fontsize=16)

    plt.show()


def runOptimizer(file_name, motor_input):
    data = pd.read_excel(f'CARLAData/{file_name}')

    # Make sure to change label below for correct velocity data
    v = data['vz'].values
    t_data = data['time'].values
    n = len(t_data)

    model = pyo.ConcreteModel()
    # Model Variables
    model.z1 = pyo.Var()  # z1 - Motor input (b),
    model.z2 = pyo.Var()  # z2 - force of friction (F_f),
    model.z3 = pyo.Var()  # z3 - coefficient of drag (C_d)

    model.cost = pyo.Objective(
        expr=sum(((v[t + 1] - v[t]) - (t_data[t + 1] - t_data[t])
                  * ((2 / mass) * (model.z1 * motor_input - model.z2 - model.z3 * v[t] ** 2))) ** 2
                 for t in range(n + 1) if t < n - 1), sense=pyo.minimize)

    model.Constraint1 = pyo.Constraint(expr=model.z1 >= 0)
    model.Constraint2 = pyo.Constraint(expr=model.z2 >= 0)
    model.Constraint3 = pyo.Constraint(expr=model.z3 >= 0)
    model.Constraint4 = pyo.Constraint(expr=model.z1 <= 3000 * 3.6)
    model.Constraint5 = pyo.Constraint(expr=model.z2 <= 200 * 3.6)
    model.Constraint6 = pyo.Constraint(expr=model.z3 <= 30 * 3.6)

    results = pyo.SolverFactory('ipopt').solve(model).write()
    b = pyo.value(model.z1)
    F_f = pyo.value(model.z2)
    C_d = pyo.value(model.z3)

    return b, F_f, C_d


def velocityModel(b, F_f, C_d, motor_input, startTime):
    n = 60
    v0 = 0
    endTime = 45
    t_arr = np.linspace(startTime, endTime, n)
    delta_t = endTime / n
    velocity = np.zeros(n)
    velocity[0] = v0
    for i in range(n - 1):
        v_next = delta_t * (
                    b * motor_input - F_f - C_d * velocity[i] ** 2) / mass + \
                 velocity[i]
        velocity[i+1] = v_next

    return velocity, t_arr


def getForceParameters():
    b_arr = []
    F_f_arr = []
    C_d_arr = []
    carla_input = .10
    file_names = carla_files

    for i in range(len(file_names)):
        carla_input = carla_input + .10
        b, F_f, C_d = runOptimizer(file_names[i], carla_input)
        b_arr.append(b)
        F_f_arr.append(F_f)
        C_d_arr.append(C_d)

    startAvgIndex = 2
    print('--- Parameters ---')
    print(b_arr[startAvgIndex:])
    print(F_f_arr[startAvgIndex:])
    print(C_d_arr[startAvgIndex:])

    # print('--- Lengths ---')
    # print(len(b_arr))
    # print(len(F_f_arr))
    # print(len(C_d_arr))

    print('--- Averages ---')
    print(np.average(b_arr[startAvgIndex:]))
    print(np.average(F_f_arr[startAvgIndex:]))
    print(np.average(C_d_arr[startAvgIndex:]))

    print('--- Std Deviations ---')
    print(np.std(b_arr[startAvgIndex:]))
    print(np.std(F_f_arr[startAvgIndex:]))
    print(np.std(C_d_arr[startAvgIndex:]))

    return b_arr, F_f_arr, C_d_arr


def plot_multiple_data(b_arr, F_f_arr, C_d_arr):
    trials = []
    colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k']

    for file in carla_files:
        trials.append(pd.read_excel(f'CARLA_Data/{file}'))

    print('--- Final Velocities ---')
    fig = plt.figure()
    for i in range(len(trials)):
        trial_speed = round(i * .1 + .2, 1)
        trial = trials[i]
        time_data = trial['time'].values
        if i > 1 and i < 7:
            # trial_speed = round(i * .1 + .2, 1)
            # trial = trials[i]
            # time_data = trial['time'].values

            b = b_arr[i]
            F_f = F_f_arr[i]
            C_d = C_d_arr[i]

            vel_model, time_fit = velocityModel(b, F_f, C_d, trial_speed,
                                                time_data[0])
            color = colors[i-2]
            plt.plot(time_data, trial['vz'].values, color,
                     label=f'Vx Data, {trial_speed} input')
            plt.plot(time_fit, vel_model, color, linestyle='--',
                     label=f'Model Fit, {trial_speed} input')

    plt.ylabel('Velocity [km/hr]')
    plt.xlabel('Time [s]')
    leg = fig.legend()
    leg.set_draggable(state=True)
    plt.show()


b, F_f, C_d = getForceParameters()
plot_multiple_data(b, F_f, C_d)
# plotData(*getForceParameters())

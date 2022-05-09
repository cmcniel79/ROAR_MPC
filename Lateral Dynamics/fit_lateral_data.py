import matplotlib.pyplot as plt
import pyomo.environ as pyo
import pandas as pd
import numpy as np

data_files = [
    '0.05 Yaw Trial.xlsx',
    '0.2 Yaw Trial.xlsx',
    '0.4 Yaw Trial.xlsx',
    '0.6 Yaw Trial.xlsx',
    '0.8 Yaw Trial.xlsx',
    '1.0 Yaw Trial.xlsx',
]

steer_inputs = np.array([
    0.05,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0
])

yaw_rates = {
    '0.2': [
        1.19,
        4.38,
        8.54,
        12.76,
        14.9,
        17.49
    ],
    '0.4': [
        2.54,
        9.03,
        16.75,
        22.89,
        31.46,
        34.64
    ]
}

# Carla Tesla Mass Properties
mass = 1845  # kg
wheelbase = 3.0
Izz = 0.95 * mass / (wheelbase / 2) ** 2
Lf = 1.62
Lr = 1.38
weight_per_axle = 0.5 * mass * 9.81
gravity = 9.81
height_COM = .5

b = 7834.780906711209
F_f = 164.24157438565635
C_d = 3.2252643317411605

def plotData():
    motor_inputs = [0.2, 0.4]
    plot_ranges = [[110, 200],
                   [60, 120]]

    for i in range(len(motor_inputs)):
        trials = []
        indexes = plot_ranges[i]

        for file in data_files:
            trials.append(pd.read_excel(
                f'Yaw Trials/{str(motor_inputs[i])} Motor/{file}'))

        fig = plt.figure(i + 1)
        for j in range(len(trials)):
            trial = trials[j]
            time_data = trial['time'].values[indexes[0]:]
            trial_data = trial['yaw'].values[indexes[0]:]
            speed_data = np.sqrt(trial['vx'].values[indexes[0]:] ** 2
                                 + trial['vy'].values[
                                   indexes[0]:] ** 2
                                 + trial['vz'].values[
                                   indexes[0]:] ** 2)
            slope = (trial_data[-1] - trial_data[0]) / (
                    time_data[-1] - time_data[0])
            constant = trial_data / (steer_inputs[j]*70 * speed_data)
            # plt.plot(time_data, trial_data,label=round(slope / (steer_inputs[j]*70*speed_data), 2))
            plt.plot(time_data, trial_data, label=f'δ = {steer_inputs[j]}')
            # plt.plot(time_data, trial['vx'].values[indexes[0]: indexes[1]],
            #          time_data, trial['vy'].values[indexes[0]: indexes[1]],
            #          time_data, trial['vz'].values[indexes[0]: indexes[1]])
    plt.ylabel('Yaw [degrees]')
    plt.xlabel('Time [s]')
    leg = fig.legend()
    leg.set_draggable(state=True)
    # fig.suptitle(f'Constant Steering Input Trials, Motor Input = {(i + 1) * .20}', fontsize=16)
    plt.show()


def runOptimizer(v_steady_arr, r_arr, steer_angle, beta, af, ar, motor_input):
    n = len(v_steady_arr)

    model = pyo.ConcreteModel()
    model.kidx = pyo.Set(initialize=range(0, n))

    # Model Variables
    model.B = pyo.Var()  # B - Stiffness Factor
    model.C = pyo.Var()  # C - Shape Factor
    model.mu = pyo.Var()  # mu

    model.F_f_y = pyo.Var(model.kidx)  # Front Lateral Force
    model.F_r_y = pyo.Var(model.kidx)  # Rear Lateral Force

    model.F_f_z = pyo.Var()
    model.F_r_z = pyo.Var()

    model.F_r_x = pyo.Var(model.kidx)

    model.cost = pyo.Objective(
        expr=sum(((1 / (mass * v_steady_arr[k])) * (
                model.F_f_y[k] * pyo.cos(steer_angle[k]) + model.F_r_y[k]) - r_arr[k])**2
                 + ((1 / Izz) * (Lf * model.F_f_y[k] * pyo.cos(steer_angle[k]) - Lr * model.F_r_y[k]))**2
                 for k in model.kidx), sense=pyo.minimize)

    model.constraint_front_lateral = pyo.Constraint(
        model.kidx,
        rule=lambda model,
        k: model.F_f_y[k] == -model.mu * model.F_f_z * pyo.sin(
        model.C * pyo.atan(model.B * af[k])))

    model.constraint_rear_lateral = pyo.Constraint(
        model.kidx,
        rule=lambda model,
        k: model.F_r_y[k] == -model.mu * model.F_r_z * pyo.sin(
        model.C * pyo.atan(model.B * ar[k])))

    model.constraint_front_normal = pyo.Constraint(expr=(model.F_f_z == mass * gravity - model.F_r_z))

    model.constraint_front_normal2 = pyo.Constraint(expr=(model.F_f_z >= 0.4 * mass * gravity))
    model.constraint_rear_normal = pyo.Constraint(expr=(model.F_r_z >= 0.4 * mass * gravity))

    model.constraint_rear_long = pyo.Constraint(
        model.kidx,
        rule=lambda model,
        k: model.F_r_x[k] == b * motor_input - F_f - C_d * v_steady_arr[k]**2)

    model.constraint_rear_total = pyo.Constraint(
        model.kidx,
        rule=lambda model,
        k: model.F_r_y[k] <= pyo.sqrt(model.F_r_z**2 - model.F_r_x[k]**2 + 10e-12))

    model.constraint_B = pyo.Constraint(expr=(model.B == 4.55))
    model.constraint_C = pyo.Constraint(expr=(2.1, model.C, 2.2))
    model.constraint_mu = pyo.Constraint(expr=model.mu == 1.0)

    # results = pyo.SolverFactory('ipopt').solve(model).write()
    solver = pyo.SolverFactory('ipopt')
    solver.options['max_iter'] = 100000
    results = solver.solve(model, tee=False)
    JOpt = pyo.value(model.cost)
    B = pyo.value(model.B)
    C = pyo.value(model.C)
    mu = pyo.value(model.mu)

    Ff_z = pyo.value(model.F_f_z)
    Fr_z = pyo.value(model.F_r_z)

    Ff_y = np.zeros(n)
    Fr_y = np.zeros(n)

    Fx = np.zeros(n)

    for k in model.kidx:
        Ff_y[k] = pyo.value(model.F_f_y[k])
        Fr_y[k] = pyo.value(model.F_r_y[k])
        # Ff_z[k] = pyo.value(model.F_f_z[k])
        # Fr_z[k] = pyo.value(model.F_r_z[k])
        Fx[k] = pyo.value(model.F_r_x[k])

    print(f'Front Normal: {Ff_z}')
    print(f'Rear Normal: {Fr_z}')
    print(f'Rear Long: {Fx}')

    return B, C, mu, Ff_y, af, Fr_y, ar, Ff_z, Fr_z, np.average(Fx)


def get_tire_parameters(motor_input):
    file_names = data_files
    v_steady_arr = []

    for file_name in file_names:
        data = pd.read_excel(f'Yaw Trials/{str(motor_input)} Motor/{file_name}')
        v = np.sqrt(data['vx'].values ** 2
                    + data['vy'].values ** 2
                    + data['vz'].values ** 2)
        v_steady_arr.append(v[-1])

    v_steady_arr = np.array(v_steady_arr)
    r_arr = (yaw_rates[str(motor_input)])  # in angles
    steer_angle_arr = (steer_inputs * 70.0) # in angles

    beta = np.arctan((Lr / wheelbase) * np.tan(steer_angle_arr))  # in angles

    alpha_f = np.zeros(len(beta))
    alpha_r = np.zeros(len(beta))
    for i in range(len(beta)):
        alpha_f[i] = np.arctan(beta[i] + ((r_arr[i] * Lf) / v_steady_arr[i])) - steer_angle_arr[i]  # in angles
        alpha_r[i] = np.arctan(beta[i] - (r_arr[i] * Lr) / v_steady_arr[i])  # in angles

    return runOptimizer(v_steady_arr, np.deg2rad(r_arr), np.deg2rad(steer_angle_arr), np.deg2rad(beta), np.deg2rad(alpha_f), np.deg2rad(alpha_r), motor_input)


# plotData()
B1, C1, mu1, F_f_y1, alpha_f1, F_r_y1, alpha_r1, F_f_z1, F_r_z1, F_x1 = get_tire_parameters(0.2)
B2, C2, mu2, F_f_y2, alpha_f2, F_r_y2, alpha_r2, F_f_z2, F_r_z2, F_x2 = get_tire_parameters(0.4)

print('B Values')
print(B1)
print(B2)
print('C Values')
print(C1)
print(C2)
print('Mu Values')
print(mu1)
print(mu2)
print('F_z Values')
print(F_f_z1)
print(F_r_z1)

alpha_crit = np.tan(np.sin(np.sqrt(F_r_z1**2 - F_x1**2) / (mu1 * F_r_z1))/C1)/B1
# print(alpha_crit)
alpha = np.deg2rad(np.linspace(-60, 10, 1000))
Fy1 = -mu1 * F_f_z1 * np.sin(C1 * np.arctan(B1 * alpha))
Fy2 = -mu1 * F_r_z1 * np.sin(C1 * np.arctan(B1 * alpha))

fig, axs = plt.subplots(1, sharex=True)
# axs[0].plot(alpha, Fy1, alpha, Fy2)
# axs.plot(alpha_r1, F_r_y1, 'ro', label='Recorded Rear Wheel Data')
# axs.plot(alpha_f1, F_f_y1, 'b^', label='Recorded Front Wheel Data')
# axs[0].set_title('Lateral Forces at 0.2 Input')

Fy3 = -mu2 * F_f_z2 * np.sin(C2 * np.arctan(B2 * alpha))
Fy4 = -mu2 * F_r_z2 * np.sin(C2 * np.arctan(B2 * alpha))

axs.plot(alpha_f2, F_f_y2, 'b^', label='Recorded Front Wheel Data')
axs.plot(alpha_r2, F_r_y2, 'ro', label='Recorded Rear Wheel Data')

axs.plot(alpha, Fy3, label='Front Wheel Modelled Force')
axs.plot(alpha, Fy4, label="Rear Wheel Modelled Force")
# axs.set_title('Lateral Forces at 0.4 Input')

plt.xlabel('Side Slip Angle α [rad]')
plt.ylabel('Tire Lateral Force [N]')

leg = fig.legend()
leg.set_draggable(state=True)
plt.xlim([-np.pi/4, 0.2])
fig.tight_layout()
plt.show()

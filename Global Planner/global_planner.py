import numpy as np
import pyomo.environ as pyo
import matplotlib
import matplotlib.pyplot as plt

# Model parameters
# Longitudinal Dynamics Parameters
b_motor = 9503  # dimensionless motor constant
mass = 1845  # kg
F_friction = 133  # N
C_d = .46  # Drag coefficient

# Lateral Dynamics Parameters
B = 4.52
C = 2.16
mu = 1.0
wheelbase = 3.0
Izz = 0.95 * mass / (wheelbase / 2) ** 2
Lf = 1.62
Lr = 1.38
Ff_z = 7239
Fr_z = 10859
max_angle = np.deg2rad(70.0)

dt0 = 0.03


def get_track(should_plot_track, m):
    fname = 'easy_map_track_model.csv'
    # Load the CSV file (note that we skip the last line)
    data = np.genfromtxt('easy_map_track_model.csv', delimiter=',',
                         skip_header=1, skip_footer=1)
    # Parse the CSV data into fields we can use easily:
    x_ref = data[:, 0]
    z_ref = data[:, 1]
    if should_plot_track:
        plt.plot(z_ref[::m], x_ref[::m])
        plt.show()
    return z_ref[::m], x_ref[::m]


def solve_nlp(nz, nu, N, init_state, init_input, final_state):
    """Solve the NLP"""
    model = pyo.ConcreteModel()
    model.N = N
    model.nz = nz
    model.nu = nu
    model.tidx = pyo.Set(initialize=range(model.N + 1))
    model.zidx = pyo.Set(initialize=range(model.nz))
    model.uidx = pyo.Set(initialize=range(model.nu))

    model.z = pyo.Var(model.zidx, model.tidx)
    model.u = pyo.Var(model.uidx, model.tidx)
    model.Ts = pyo.Var()

    model.slack_x = pyo.Var()
    model.slack_y = pyo.Var()

    eps = 10e-12

    time_range = range(0, N+1)
    # Objective function
    objective_terms = [
        sum(model.Ts * time_range[t]
            for t in model.tidx if t < N),  # sum of time
        # sum(10**-5 * model.u[i, t]**2 for i in model.uidx for t in model.tidx),
        sum(10**5 *(model.z[i, N] - final_state[i])**2 for i in model.zidx if i < 2),
    ]
    model.cost = pyo.Objective(expr=sum(objective_terms), sense=pyo.minimize)

    # Dynamics constraints
    # x(k+1), state 0
    model.x_dynamics = pyo.Constraint(model.tidx,
                                      rule=lambda model, t: model.z[0, t + 1] == model.z[0, t] + model.Ts * (
                                          pyo.sqrt(model.z[5, t]**2 + (model.z[5, t] * model.z[3, t])**2 + eps) * pyo.cos(np.pi/2 + model.z[2, t] + model.z[3, t]))
                                      if t < N else pyo.Constraint.Skip)
    # y(k+1), state 1
    model.y_dynamics = pyo.Constraint(model.tidx,
                                      rule=lambda model, t: model.z[1, t + 1] == model.z[1, t] + model.Ts * (
                                          pyo.sqrt(model.z[5, t]**2 + (model.z[5, t] * model.z[3, t])**2 + eps) * pyo.sin(np.pi/2 + model.z[2, t] + model.z[3, t]))
                                      if t < N else pyo.Constraint.Skip)
    # yaw(k+1), state 2
    model.yaw_dynamics = pyo.Constraint(model.tidx,
                                            rule=lambda model, t: model.z[2, t + 1] == model.z[2, t] + model.Ts * (
                                                model.z[4, t])
                                            if t < N else pyo.Constraint.Skip)
    # beta(k+1), state 3
    model.beta_dynamics = pyo.Constraint(model.tidx,
                                         rule=lambda model, t: model.z[3, t + 1] == model.z[3, t] + model.Ts * (
                                             -model.z[4, t] + (1/(mass * model.z[5, t] + eps)) * (model.z[7, t] + (model.u[0, t] * pyo.cos(np.pi/2 + model.z[6, t]))))
                                         if t < N else pyo.Constraint.Skip)
    # r(k+1), state 4, this is yaw angle rate
    model.r_dynamics = pyo.Constraint(model.tidx,
                                      rule=lambda model, t: model.z[4, t + 1] == model.z[4, t] + model.Ts * (
                                          (1/Izz) * (Lf * model.u[0, t] * pyo.cos(np.pi/2 + model.z[6, t]) - Lr * model.z[7, t]))
                                      if t < N else pyo.Constraint.Skip)

    # v_x(k+1), state 5
    model.v_x_dynamics = pyo.Constraint(model.tidx,
                                        rule=lambda model, t: model.z[5, t + 1] == model.z[5, t] + model.Ts * (
                                            (model.z[3, t] * model.z[4, t] * model.z[5, t]) + (1/mass) * (model.u[1, t] - model.u[0, t] * pyo.sin(np.pi/2 + model.z[6, t])))
                                        if t < N else pyo.Constraint.Skip)

    # # delta(k), state 6, but this constraint not needed
    # model.delta_dynamics = pyo.Constraint(model.tidx,
    #                                       rule=lambda model, t: model.z[6, t] == pyo.atan(
    #                                           pyo.tan(model.z[2, t]) * (wheelbase / Lr))
    #                                       if t < N else pyo.Constraint.Skip)

    # # Fry(k), state 7, but this constraint not needed
    # model.Fry_dynamics = pyo.Constraint(model.tidx,
    #                                     rule=lambda model, t: model.z[7, t] == -mu * Fr_z * pyo.sin(
    #                                         C * pyo.atan(B * model.z[7, t]))
    #                                     if t < N else pyo.Constraint.Skip)

    # Initial State Conditions
    # model.init_state = pyo.Constraint(model.zidx, rule=lambda model, i: model.z[i, 0] == init_state[i] if i != 2 or i != 3 else pyo.Constraint.Skip)
    model.init_state = pyo.Constraint(model.zidx, rule=lambda model, i: model.z[i, 0] == init_state[i])

    # Initial Input Conditions
    # model.init_input = pyo.Constraint(model.uidx, rule=lambda model, i: model.u[i, 0] == init_input[i])

    # State Constraints - Max
    model.constraint_x_max = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[0, t] <= 450)
    model.constraint_y_max = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[1, t] <= 450)
    model.constraint_yaw_max = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[2, t] <= 2 * np.pi)
    model.constraint_beta_max = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[3, t] <= 2 * np.pi)
    model.constraint_r_max = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[4, t] <= 10)
    model.constraint_v_x_max = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[5, t] <= 100.0)
    model.constraint_delta_max = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[6, t] <= np.pi/2)
    model.constraint_fry_max = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[7, t] <= Fr_z / 2)

    # State Constraints - Min
    model.constraint_x_min = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[0, t] >= -450)
    model.constraint_y_min = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[1, t] >= -450)
    model.constraint_yaw_min = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[2, t] >= -2 * np.pi)
    model.constraint_beta_min = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[3, t] >= -2 * np.pi)
    model.constraint_r_min = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[4, t] >= -10)
    model.constraint_v_x_min = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[5, t] >= -5.0)
    model.constraint_delta_min = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[6, t] >= -np.pi/2)
    model.constraint_fry_min = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[7, t] >= -Fr_z / 2)

    # Input constraints
    model.constraint_ffy_max = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[7, t] <= Ff_z)
    model.constraint_frx_max = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[1, t] <= b_motor * 1.0 - F_friction - C_d * 100**2)
    model.constraint_ffy_min = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[7, t] >= -Ff_z)
    model.constraint_frx_min = pyo.Constraint(model.tidx,
        rule=lambda model,
        t: model.z[1, t] >= 0)


    # Terminal conditions
    # model.final_x = pyo.Constraint(expr=model.z[0, N] == final_state[0])
    # model.final_y = pyo.Constraint(expr=model.z[1, N] == final_state[1])

    # Time step constraints
    model.Ts_min = pyo.Constraint(expr=model.Ts >= 0.1)
    model.Ts_max = pyo.Constraint(expr=model.Ts <= 1)
    model.slack_x_max = pyo.Constraint(expr=model.slack_x <= 5)
    model.slack_y_max = pyo.Constraint(expr=model.slack_y <= 5)

    # Solve:
    solver = pyo.SolverFactory('ipopt')
    # solver.options['max_iter'] = 10000
    results = solver.solve(model, tee=False)
    Ts = pyo.value(model.Ts)
    zOpt = np.asarray([[model.z[i, t]() for i in model.zidx] for t in model.tidx]).T
    uOpt = np.asarray([model.u[:, t]() for t in model.tidx]).T
    return zOpt, uOpt, Ts

def do_everything():
    # m is used to get m-other point in the track model
    m = 100
    track_x, track_y = get_track(False, m)

    section_x = track_x[1:3]
    section_y = track_y[1:3]

    init_yaw = 0
    init_state = np.array([section_x[0], section_y[0], init_yaw, 0, 0, 0, 0, 0, 0])
    init_input = np.array([0, 0])
    final_state = np.array([section_x[-1], section_y[-1]])

    full_x = np.array([])
    full_y = np.array([])

    N_opt_steps = 7
    N_track_segments = 15
    for j in range(N_track_segments):
        print(f'Solving segment {j}')
        zOpt, uOpt, Ts = solve_nlp(8, 2, N_opt_steps, init_state, init_input, final_state)

        x_opt = zOpt[0, :]
        y_opt = zOpt[1, :]
        yaw_opt = zOpt[2, :]
        # print(yaw_opt)
        # print(Ts)
        # plt.plot(track_x[0:5], track_y[0:5], '-bo')
        # plt.plot(x_opt, y_opt, '-ro')
        # plt.plot(section_x, section_y, '-ko')
        # plt.plot(x_opt[0], y_opt[0], 'go')
        # plt.plot(section_x[0], section_y[0], 'bo')
        # plt.show()
        # print(f'zOpt {j}')
        # print(zOpt[:, -1])

        full_x = np.concatenate((full_x, x_opt))
        full_y = np.concatenate((full_y, y_opt))

        section_x = track_x[j+2:j+4]
        section_y = track_y[j+2:j+4]
        init_state = zOpt[:, -1]
        init_input = uOpt[:, -2]
        final_state = np.array([section_x[-1], section_y[-1]])

    fig = plt.figure()
    plt.plot(full_x, full_y, '--r', label='Optimized Trajectory')
    plt.plot(track_x[1:N_track_segments + 2], track_y[1:N_track_segments + 2], 'b', label='Track Centerline')
    plt.legend().set_draggable(True)
    plt.xlabel('Z Position [m]')
    plt.ylabel('X Position [m]')
    # fig.suptitle(f'Optimized Trajectory from Non-Linear, Global Planner Program', fontsize=16)
    plt.show()


for i in range(20):
    try:
        do_everything()
    except:
        print('couldnt do it')


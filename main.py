import plotly.graph_objects as go
import pandas as pd
import numpy as np

def generate_particle_data(num_points, magnetic_field_strength, electric_field_strength, forward_speed, angle):
    t = np.linspace(0, 10, num_points)

    charge = 1.0
    mass = 1.0

    x, y, z = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    u, v, w = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)

    # Calculate initial velocity components based on the angle
    angle_rad = np.radians(angle)
    initial_speed = np.cos(angle_rad) * forward_speed
    initial_vertical_speed = np.sin(angle_rad) * forward_speed

    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]

        magnetic_force = charge * np.cross(np.array([u[i - 1], v[i - 1], w[i - 1]]),
                                           np.array([0, 0, magnetic_field_strength]))

        electric_force = charge * electric_field_strength * np.array([1, 0, 0])

        total_force = magnetic_force + electric_force

        acceleration = total_force / mass

        u[i] = u[i - 1] + acceleration[0] * dt
        v[i] = v[i - 1] + acceleration[1] * dt
        w[i] = w[i - 1] + acceleration[2] * dt

        x[i] = x[i - 1] + u[i] * dt
        y[i] = y[i - 1] + v[i] * dt
        z[i] = z[i - 1] + w[i] * dt + initial_vertical_speed * dt

    return pd.DataFrame({'x': x, 'y': y, 'z': z, 'u': u, 'v': v, 'w': w})


magnetic_field_strength = 5
electric_field_strength = 2
forward_speed = 1
angle = 90 # Angle of incidence in degrees

df_particle_data = generate_particle_data(num_points=300,
                                          magnetic_field_strength=magnetic_field_strength,
                                          electric_field_strength=electric_field_strength,
                                          forward_speed=forward_speed,
                                          angle=angle)

fig = go.Figure(data=go.Cone(
    x=df_particle_data['x'],
    y=df_particle_data['y'],
    z=df_particle_data['z'],
    u=df_particle_data['u'],
    v=df_particle_data['v'],
    w=df_particle_data['w'],
    colorscale='blues',
    sizemode="absolute",
    sizeref=50
))

fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),
                             camera_eye=dict(x=1.2, y=1.2, z=0.6)))

fig.show()

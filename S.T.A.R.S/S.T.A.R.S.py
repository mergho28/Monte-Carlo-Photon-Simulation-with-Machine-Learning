import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Scale, Label, Button, Radiobutton, IntVar, HORIZONTAL
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Photon class definition with 3D coordinates
class Photon:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0])  # Starting at the origin in 3D
        self.path = [self.position.copy()]  # Path history
        self.alive = True

    def move(self, step_size):
        # Randomize direction of travel in 3D
        theta = np.random.uniform(0, 2 * np.pi)  # Random azimuthal angle
        phi = np.random.uniform(0, np.pi)  # Random polar angle
        movement = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]) * step_size
        self.position += movement
        self.path.append(self.position.copy())

# Monte Carlo Simulation Function (3D version)
def simulate_photons(num_photons, max_steps, step_size, medium_density, absorption_prob, scattering_angle_std):
    photons = [Photon() for _ in range(num_photons)]
    for i, photon in enumerate(photons):
        for step in range(max_steps):
            if not photon.alive:
                break
            photon.move(step_size)
            if np.random.rand() < medium_density:
                if np.random.rand() < absorption_prob:
                    photon.alive = False  # Photon absorbed
                    print(f"Photon {i} absorbed.")
                else:
                    # Scatter the photon
                    scatter_angle = np.random.normal(0, scattering_angle_std)
                    photon.move(np.random.uniform(0, 2 * np.pi))
                    print(f"Photon {i} scattered.")
    return photons

# Function to run the animation-based simulation
def run_simulation():
    # Get values from the sliders
    medium_density = medium_density_slider.get() / 100  # Scale to 0-1 range
    absorption_prob = absorption_slider.get() / 100  # Scale to 0-1 range
    scattering_angle_std = scattering_slider.get() / 100 * np.pi  # Scale to radians

    # Create the photons
    photons = simulate_photons(100, 100, 1, medium_density, absorption_prob, scattering_angle_std)

    # Set up the figure and axes for 3D animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize lines for each photon
    lines = [ax.plot([], [], [], 'o', markersize=3)[0] for _ in range(len(photons))]

    # Set plot limits
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_zlim([-50, 50])

    # Function to initialize the animation
    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return lines

    # Function to update the animation frame by frame
    def update(frame):
        for i, photon in enumerate(photons):
            if frame < len(photon.path):
                path = np.array(photon.path)
                lines[i].set_data(path[:frame, 0], path[:frame, 1])
                lines[i].set_3d_properties(path[:frame, 2])
        return lines

    ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=False, interval=50)
    plt.show()

# Function to run the simulation and use ML for prediction
def run_simulation_with_ml():
    medium_density = medium_density_slider.get() / 100  # Scale to 0-1 range
    absorption_prob = absorption_slider.get() / 100  # Scale to 0-1 range
    scattering_angle_std = scattering_slider.get() / 100 * np.pi  # Scale to radians

    outcome = predict_with_ml(medium_density, absorption_prob, scattering_angle_std)
    print(f"Predicted outcome: {outcome}")
    run_simulation()  # Run the animated simulation

# Data preparation and training a machine learning model
def generate_simulation_data(num_samples=10000):
    X = []
    y = []
    for _ in range(num_samples):
        medium_density = np.random.uniform(0, 1)
        absorption_prob = np.random.uniform(0, 1)
        scattering_angle_std = np.random.uniform(0, np.pi)
        photon = Photon()
        photon.move(1)
        if np.random.rand() < medium_density:
            outcome = 0 if np.random.rand() < absorption_prob else 1  # 0 = absorbed, 1 = scattered
        else:
            outcome = 1
        X.append([medium_density, absorption_prob, scattering_angle_std])
        y.append(outcome)
    return np.array(X), np.array(y)

X, y = generate_simulation_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

def predict_with_ml(medium_density, absorption_prob, scattering_angle_std):
    prediction = model.predict([[medium_density, absorption_prob, scattering_angle_std]])[0]
    return "Absorbed" if prediction == 0 else "Scattered"

# Add radiobuttons for selecting the environment
def update_environment():
    selected_env = environment_choice.get()
    if selected_env == 1:
        medium_density_slider.set(50)  # Star's interior preset
        absorption_slider.set(10)
        scattering_slider.set(40)
    elif selected_env == 2:
        medium_density_slider.set(20)  # Interstellar matter preset
        absorption_slider.set(5)
        scattering_slider.set(25)
    elif selected_env == 3:
        medium_density_slider.set(70)  # Planetary atmosphere preset
        absorption_slider.set(15)
        scattering_slider.set(10)

# GUI Setup
root = Tk()
root.title("Monte Carlo Simulation: Radiative Transfer")

Label(root, text="Medium Density").pack()
medium_density_slider = Scale(root, from_=0, to=100, orient=HORIZONTAL)
medium_density_slider.set(10)  # Initial value
medium_density_slider.pack()

Label(root, text="Absorption Probability").pack()
absorption_slider = Scale(root, from_=0, to=100, orient=HORIZONTAL)
absorption_slider.set(5)  # Initial value
absorption_slider.pack()

Label(root, text="Scattering Angle (Standard Deviation)").pack()
scattering_slider = Scale(root, from_=0, to=100, orient=HORIZONTAL)
scattering_slider.set(25)  # Initial value
scattering_slider.pack()

environment_choice = IntVar()
Label(root, text="Select Environment").pack()
Radiobutton(root, text="Star's Interior", variable=environment_choice, value=1, command=update_environment).pack()
Radiobutton(root, text="Interstellar Matter", variable=environment_choice, value=2, command=update_environment).pack()
Radiobutton(root, text="Planetary Atmosphere", variable=environment_choice, value=3, command=update_environment).pack()

environment_choice.set(1)
update_environment()

Button(root, text="Run 3D Simulation", command=run_simulation).pack()
Button(root, text="Run Simulation with ML", command=run_simulation_with_ml).pack()

root.mainloop()




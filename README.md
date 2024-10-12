# Monte Carlo Photon Simulation with Machine Learning

This project is a Python-based simulation of photon paths using a Monte Carlo method in a 3D environment. It models the behavior of photons as they travel through a medium, where they can be scattered or absorbed. The project integrates machine learning (Random Forest Classifier) to predict photon interactions based on environmental parameters.

## Features

- **Monte Carlo Simulation**: Simulates photon movement in a 3D space, allowing for random absorption and scattering events based on medium properties.
- **3D Visualization**: Uses Matplotlib to animate the movement of photons in a customizable environment, allowing users to visualize the paths of individual photons.
- **Machine Learning Prediction**: A trained Random Forest model predicts whether photons will be absorbed or scattered based on medium density, absorption probability, and scattering angle.
- **Customizable Environment**: Users can adjust environmental factors such as medium density, absorption probability, and scattering angle via an interactive GUI.
- **Environment Presets**: Predefined environmental settings for different astronomical scenarios, including a star's interior, interstellar matter, and planetary atmospheres.

## Installation

### Prerequisites
Ensure you have Python 3.x installed. Additionally, you will need the following Python libraries:
- `numpy`
- `matplotlib`
- `tkinter` (built into Python)
- `scikit-learn`

You can install the required libraries using `pip`:

```bash
pip install numpy matplotlib scikit-learn
Run the code from S.T.A.R.S.py in a Python environment


# CS225A Project

Computer-vision-guided robotic tracking system.

Pipeline:

Camera → Perception → State Estimation → Controller → Robot

## Structure

- `src/perception`: object detection, tracking, and camera calibration
- `src/estimation`: filtering and velocity estimation
- `src/control`: operational space control and trajectory generation
- `src/robot`: robot model, kinematics, and hardware/simulation interface
- `src/pipeline`: main system loop
- `config`: tunable parameters
- `scripts`: quick run scripts
- `data`: logs, videos, and processed outputs
- `docs`: design notes and setup guide

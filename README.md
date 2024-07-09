# Pick-and-place Operation by a two-fingered Robot using Faster R-CNN

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Background](#project-background)
3. [Features](#features)
4. [Contents](#contents)
5. [Requirements](#requirements)
6. [Installation](#installation)
7. [Data Preparation](#data-preparation)
8. [Usage](#usage)
    - [Training the Model](#training-the-model)
    - [Testing the Model](#testing-the-model)
9. [Results](#results)
10. [Detailed Information](#detailed-information)
11. [Contributing](#contributing)
12. [License](#license)
13. [Acknowledgments](#acknowledgments)

## Project Overview

This project aims to utilize Faster R-CNN for the Food Packaging Process, with a specific focus on packaging various types of chocolates in a factory's production line. The objective is to identify chocolates in images captured by a camera to automate the pick-and-place task using a two-fingered gripper and a delta parallel robot. The primary aim is to enhance the precision and efficacy of detecting and packaging vulnerable soft chocolates during the pick-and-place operation without causing any damage while maintaining stringent hygiene standards and reducing the need for manual labor.

## Features

- **Object Detection**: Utilizing Faster R-CNN to accurately detect the position of chocolates.
- **Robotic Integration**: Integration with a two-fingered gripper and Delta Parallel Robot for pick-and-place operation.
- **High Accuracy**: Achieves high accuracy and success rates in detecting and automated pick-and-place operations on practical tests.


## Pick-and-place Experimental Setup

### The Pick-and-place Experimental Setup
The initial setup includes a partially filled box in a random position and orientation with scattered pieces of chocolate, all placed in the Delta Parallel Robot workspace (depicted in Fig. 1, 2, and Fig. 3). The robot’s movement is directed by classical trajectory planning methods, such as the 4-5-6-7 interpolating polynomial and cubic spline. In order to allow the robot to interact with target objects, a two-fingered gripper is mounted on the end-effector. The gripper will be controlled with a data cable connected to an Arduino kit. The generated results will be wirelessly transmitted to the robot utilizing the Transmission Control Protocol (TCP). 

### Delta Parallel Robot (DPR) Structure 
The DPR is a parallel structure comprising three upper arms and three lower arms. Each upper arm connects to the base plate with a revolute joint on one end and to the lower arm with a universal joint on the other end (shown in Fig. 4). The lower arms are connected to the traveling plate using a universal joint. This configuration results in three kinematic chains, yielding 3 DOFs in our specific design. Consequently, the 3 DOF DPR can move along three main axes x, y, and z.

### Delta Parallel Robot Kinematics
The Inverse Kinematic (IK) of a DPR involves determining the joint configurations needed to achieve a specified end-effector position, while the Forward Kinematic (FK) entails computing the end effector’s position based on the given joint configurations. The main actuators are placed on the three revolute joints which connect the upper arms to the base. The IK model used comes from solving the kinematic closed chain:

$OAi + AiBi + BiCi + CiE + EO = 0$

### Delta Parallel Robot Trajectory Planning
Trajectory planning in a DPR involves determining a sequence of desired end-effector positions over time, considering the robot’s kinematics and dynamics, to achieve smooth and efficient motion in its operational space. Assuming proper kinematic modeling, it is feasible to transition trajectory planning calculations from workspace to joint space. With the initial and final states of joint-space configuration represented by $\Theta_I$ and $\Theta_F$, respectively, one can interpolate between these two configurations using a method known as the 4-5-6-7 interpolating polynomial:

$\Theta(t) = \Theta_I + (\Theta_F - \Theta_I) p(t_{norm})$

Where $t_{norm}$ denotes normalized time, and the interpolating polynomial $p$ can be expressed as:

$p(t) = -20t^7 + 70t^6 - 84t^5 + 35t^4$

Alternatively, if there are specific points designated as targets along the path (in addition to the initial and final points), the use of cubic splines becomes a viable option. The cubic spline method efficiently interpolates a trajectory through a series of given points. When dealing with $n + 1$ points, $n$ polynomials with distinct parameters are employed to interpolate the entire path. The mathematical definition is as follows:

$p_i (t) = a_i0 + a_i1 (t − t_i) + a_i2 (t − t_i)^2 + a_i3 (t − t_i)^3$

And the overall path will be defined as:

$p_{path}(t) = p_i(t), t \in [t_i, t_{i+1}] \quad and \quad i = 0, \ldots, n-1$

where $t$ represents the overall time, $i$ represents the number of polynomials or the number of the current target point, and $t_i$ is the time when the end-effector will hit the target point number $i$.

### 2-Fingered Gripper
The gripper connected to the end-effector is a 2-fingered design based on US 9,327,411 B2 Robotic Gripper Patent, which is a research-only development with no commercial intent. This design is proficient in securing small items from a multitude of angles parallel to the operational surface. Its conceptual inspiration is drawn from the Hand-E gripper by ROBOTIQ, indicating a homage to its design principles. The operation of the gripper is facilitated through a pinion activated by a stepper motor. This setup actuates two racks, enabling precise manipulation of the fingers’ movement along a linear path. Graphite bushings coupled with hard chrome shafts provide a smooth sliding mechanism for the racks. The chosen stepper motor is the Nema17 HS8401 model, known for its 5.2 kg/cm torque output. It is driven by a TB6600 motor driver, which supports micro-stepping functionality. Incorporated at the tip of the gripper is a Force-Sensitive Resistor (FSR) that operates on the principle of variable resistance, where applied force decreases resistance by bringing a conductive and a non-conductive layer into contact. Resistance measurement is conducted through a basic voltage divider circuit, with calibration capabilities for setting a predefined force threshold necessary for gripping actions. The gripper’s structure, fabricated from heat-resistant ABS plastic, is produced via 3D printing technology, emphasizing durability and design flexibility. Control over the gripper is achieved through serial communication from an Arduino Uno board, facilitating adjustments in the gripper’s aperture and orientation via software. It boasts an expansive opening capacity of up to 60mm, accommodating diverse object sizes. Connection to the Arduino is ensured through a 5-meter shielded cable, guaranteeing stable communication. The software allows for precise control over the gripper’s states, ranging from fully opened to the act of grasping, enabling object manipulation from various angles in a horizontal orientation. The integration of the gripper with the Delta robot’s arms enables coordinated control over the object’s orientation.


------------------------------------------------------------------------------

## Contents

- `Faster R-CNN-[Chocolate Detection].ipynb`: Jupyter notebook containing the implementation of the Faster R-CNN model for chocolate detection.

## Tools and Libraries

- Python 3.7+
- PyTorch
- scikit-learn
- OpenCV
- NumPy
- albumentations
- Roboflow

## Data Preparation

1. **Collect Images**: Gathered images of chocolates in various arrangements.
2. **Annotate Images**: Useed the Roboflow annotation tool to annotate the chocolates in the raw images and saveed the annotations in XML format.
3. **Organize Dataset**: Structured the dataset directory as follows:
   ```
   dataset/
       images/
           img1.jpg
           img2.jpg
           ...
       annotations/
           img1.xml
           img2.xml
           ...
   ```

### Training the Model

1. **Prepare the Dataset**: Ensure your dataset is organized as described in the [Data Preparation](#data-preparation) section.
2. 

### Testing and Evaluation of the Model

1. **Load Trained Model**: Use the trained model saved during the training phase.
2. **Test on New Images**: Run the notebook cells designed for testing and visualize the results.
3. 

## Results

- images ...
- tables ...
- ...


## Contribute

- sarmad zandi goharrizi
- mona mohades mojtahedi

## License

.....

## Acknowledgments

- Special thanks to Dr. Tale Masouleh as the advisor and the Taarlab laboratory members for their guidance and support.

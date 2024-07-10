# Pick-and-place Operation by a Two-fingered Robot using Faster R-CNN

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Collection and Pre-Processing](#Data-Collection-and-Pre-Processing)

## Project Overview

This project aims to use the Faster R-CNN object detection model for the Food Packaging Process, with a specific focus on packaging various types of chocolates in a factory's production line. The objective is to identify chocolates in images captured by a camera to automate the pick-and-place task using a two-fingered gripper and a delta parallel robot. The primary aim is to enhance the precision and efficacy of detecting and packaging vulnerable soft chocolates during the pick-and-place operation without causing any damage while maintaining stringent hygiene standards and reducing the need for manual labor.

## Data Collection and Pre-Processing

- **Image Collection**: Captured $50$ high-quality images of chocolates from $6$ different brands with varying shapes using the [ODROID USB CAM 720P HD](https://en.odroid.se/products/usb-kamera-720p) camera with a resolution of $1280 \times 720$. All chocolates were placed on a consistent background to ensure the model focused only on the target objects. 

- **Pre-processing**: Raw images were pre-processed using the following techniques:
    - **Resizing:** Images are resized to $800 \times 600$ pixels.
    - **Normalization:** Pixel values are normalized to the range $[0, 1]$.
    - **Filtering:** Sharpening and blurring filters are applied in sequence to enhance images and reduce noise.

- **Image Annotation**: The images were captured by an RGB camera mounted on a robotic arm and saved in JPEG format. Each image was paired with XML annotation files created using the [Roboflow](https://github.com/roboflow) tool adhering to the PASCAL VOC standard to precisely delineate the boundaries and labels of individual chocolate pieces. This annotation file includes details like the image file name, dimensions (width, height, depth), and specifics about each chocolate in the image such as its label and the bounding box coordinates recorded as $(x_{min}, y_{min}, x_{max}, y_{max})$.

    The figure below displays a sample of the chocolates, each with its respective pre-processed bounding box and the corresponding label. 

    ![img1](https://github.com/Sarmadzandi/Faster-R-CNN/assets/44917340/26268304-17dc-4939-8cd8-273bc660170b)

    Furthermore, the table below presents the label corresponding to each chocolate type and its corresponding integer.

    | **Label** | **Integer** | **Description** |
    | --- | --- | --- |
    | Box | 1 | Label identifier for Chocolate Box |
    | Diamond | 2 | Label identifier for Diamond chocolate brand |
    | Domenica (Black) | 3 | Label identifier for Domenica chocolate brand (black) |
    | Domenica (Blue) | 4 | Label identifier for Domenica chocolate brand (blue) |
    | Section | 5 | Label identifier for compartments in Chocolate Box |
    | Shoniz | 6 | Label identifier for Shoniz chocolate brand |
    | Tickers | 7 | Label identifier for Tickers chocolate brand |
    | Triperz | 8 | Label identifier for Triperz chocolate brand |

- **Data Augmentation**: It's important to increase the amount of data in the dataset to expand its size and ensure reasonable diversity. This is crucial for improving the model's ability to generalize because it was not possible to collect data to cover all scenarios, different lighting conditions, and various positions of chocolates in the environment. Therefore, it's necessary to increase the amount of data in this process. To achieve this, the [Albumentations](https://github.com/albumentations-team/albumentations) library and various augmentation methods have been used, which are detailed in the table below.

    | **No.** | **Method**                | **Description**                                                                                                                                        |
    |---------|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
    | 1       | Horizontal and Vertical Flipping | Creating horizontal and vertical mirror versions of the original images to increase dataset diversity and help in identifying objects in different orientations. |
    | 2       | Random Rotation            | Rotating images to random angles between 90, 180, and 270 degrees to simulate different positions of chocolates and aid in recognition from any angle.   |
    | 3       | Brightness and Color Adjustment | Modifying the brightness, contrast, and color of images to simulate various lighting conditions and increase model resilience to changes like shadows and reflections. |
    | 4       | Adding Noise               | Adding random noise and Gaussian noise to images to make the model more robust against real-world defects like camera noise and dust on the lens.        |
    | 5       | Scaling and Rescaling      | Randomly adjusting the size of images to simulate different distances and scales, helping the model recognize objects at various scales and resolutions.  |
    | 6       | Combined Method            | Applying multiple augmentation methods in combination to create a rich and diverse training set, enhancing the model's robustness and generalization capability. |
    
    Moreover, the figure below shows an example of applying each data augmentation technique on a sample image from the dataset.
    
    ![img2](https://github.com/Sarmadzandi/Faster-R-CNN/assets/44917340/41639197-4e97-4bd3-8377-8ddc957c92b5)

## The Pick-and-place Experimental Setup
The initial setup includes a partially filled box in a random position and orientation with scattered pieces of chocolate, all placed in the Delta Parallel Robot workspace (depicted in Fig. 1, 2, and Fig. 3). The robot’s movement is directed by classical trajectory planning methods, such as the 4-5-6-7 interpolating polynomial and cubic spline. To allow the robot to interact with target objects, a two-fingered gripper is mounted on the end-effector. The gripper will be controlled with a data cable connected to an Arduino kit. The generated results will be wirelessly transmitted to the robot utilizing the Transmission Control Protocol (TCP). 

| --- | --- | --- |
| ![img3](https://github.com/Sarmadzandi/Faster-R-CNN/assets/44917340/35022709-8f89-460a-8bce-17a879b6cc8b) | ![img4](https://github.com/Sarmadzandi/Faster-R-CNN/assets/44917340/f3324181-030b-49a6-ae18-cbc2f291a15b) | ![img5](https://github.com/Sarmadzandi/Faster-R-CNN/assets/44917340/d5800496-47bb-4e42-90ad-6e808dc0745a) |
| Diamond | 2 | Label identifier for Diamond chocolate brand |

### Delta Parallel Robot (DPR) Structure 
The DPR is a parallel structure comprising three upper arms and three lower arms. Each upper arm connects to the base plate with a revolute joint on one end and to the lower arm with a universal joint on the other end (shown in Fig. 4). The lower arms are connected to the traveling plate using a universal joint. This configuration results in three kinematic chains, yielding 3 DOFs in our specific design. Consequently, the 3 DOF DPR can move along three main axes x, y, and z.

### Delta Parallel Robot Kinematics
The Inverse Kinematic (IK) of a DPR involves determining the joint configurations needed to achieve a specified end-effector position, while the Forward Kinematic (FK) entails computing the end effector’s position based on the given joint configurations. The main actuators are placed on the three revolute joints which connect the upper arms to the base. The IK model used comes from solving the kinematic closed chain:

$\overline{O A_i}+\overline{A_i B_i}+\overline{B_i C_i}+\overline{C_i E}+\overline{E O}=0$

### Delta Parallel Robot Trajectory Planning
Trajectory planning in a DPR involves determining a sequence of desired end-effector positions over time, considering the robot’s kinematics and dynamics, to achieve smooth and efficient motion in its operational space. Assuming proper kinematic modeling, it is feasible to transition trajectory planning calculations from workspace to joint space. With the initial and final states of joint-space configuration represented by $\Theta_I$ and $\Theta_F$, respectively, one can interpolate between these two configurations using a method known as the 4-5-6-7 interpolating polynomial:

$\Theta(t) = \Theta_I + (\Theta_F - \Theta_I) p(t_{norm})$

Where $t_{norm}$ denotes normalized time, and the interpolating polynomial $p$ can be expressed as:

$p(t) = -20t^7 + 70t^6 - 84t^5 + 35t^4$

Alternatively, if there are specific points designated as targets along the path (in addition to the initial and final points), the use of cubic splines becomes a viable option. The cubic spline method efficiently interpolates a trajectory through a series of given points. When dealing with $n + 1$ points, $n$ polynomials with distinct parameters are employed to interpolate the entire path. The mathematical definition is as follows:

$p_i(t) = a_{i0} + a_{i1}(t − t_i) + a_{i2}(t − t_i)^2 + a_{i3}(t − t_i)^3$

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

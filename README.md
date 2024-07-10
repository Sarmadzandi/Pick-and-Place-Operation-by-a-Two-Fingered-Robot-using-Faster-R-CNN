# Pick-and-place Operation by a Two-fingered Robot using Faster R-CNN

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Collection and Pre-Processing](#Data-Collection-and-Pre-Processing)
3. 

## Project Overview

This project aims to use the Faster R-CNN object detection model for the Food Packaging Process, with a specific focus on packaging various types of chocolates in a factory's production line. The objective is to identify chocolates in images captured by a camera to automate the pick-and-place task using a two-fingered gripper and a delta parallel robot. The primary aim is to enhance the precision and efficacy of detecting and packaging vulnerable soft chocolates during the pick-and-place operation without causing any damage while maintaining stringent hygiene standards and reducing the need for manual labor.

---
## Data Collection and Pre-Processing
- **Image Collection**: Captured $50$ high-quality images of chocolates from $6$ different brands with varying shapes using the [ODROID USB CAM 720P HD](https://en.odroid.se/products/usb-kamera-720p) camera with a resolution of $1280 \times 720$. All chocolates were placed on a consistent background to ensure the model focused only on the target objects. 

- **Pre-processing**: Raw images were pre-processed using the following techniques:
    - **Resizing:** Images are resized to $800 \times 600$ pixels.
    - **Normalization:** Pixel values are normalized to the range $[0, 1]$.
    - **Filtering:** Sharpening and blurring filters are applied in sequence to enhance images and reduce noise.

- **Image Annotation**: The images were captured by an RGB camera mounted on a robotic arm and saved in JPEG format. Each image was paired with XML annotation files created using the [Roboflow](https://github.com/roboflow) annotation tool adhering to the PASCAL VOC standard to precisely delineate the boundaries and labels of individual chocolate pieces. This annotation file includes details like the image file name, dimensions (width, height, depth), and specifics about each chocolate in the image such as its label and the bounding box coordinates recorded as $(x_{min}, y_{min}, x_{max}, y_{max})$.

    > The dataset directory is Structured as follows:
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
---
## Delta Parallel Robot (DPR) Structure and Components
The Delta Parallel Robot (DPR) workspace (shown in the figure below), consists of a parallel configuration with three upper and three lower arms. Each upper arm connects to the base plate via a revolute joint and to the lower arm with a universal joint. The lower arms connect to a traveling plate using another universal joint, forming three kinematic chains that provide 3 degrees of freedom (DOF) along the x, y, and z axes.

<div style="text-align: center;">
  <table style="margin: 0 auto; border-spacing: 10px;">
    <tr>
      <td style="padding: 10px;">
        <img src="https://github.com/Sarmadzandi/Faster-R-CNN/assets/44917340/6e399814-9d1b-4d00-ae05-51d4743d0b0b" alt="Delta Parallel Robot and 2-fingered Gripper" style="width: 100%; height: auto; vertical-align: top;"/>
      </td>
    </tr>
    <tr>
      <td style="text-align: center;">Delta Parallel Robot and 2-fingered Gripper - Human and Robot Interaction Laboratory, University of Tehran.</td>
    </tr>
  </table>
</div>

### DPR Kinematics
- **Inverse Kinematics (IK)**: Determines joint configurations for a desired end-effector position.
- **Forward Kinematics (FK)**: Computes the end-effector's position based on given joint configurations.
- The main actuators are on the revolute joints connecting the upper arms to the base.
- IK is modeled using the equation: $\overline{O A_i}+\overline{A_i B_i}+\overline{B_i C_i}+\overline{C_i E}+\overline{E O}=0$.

### DPR Trajectory Planning
Trajectory planning involves calculating a sequence of end-effector positions over time for smooth motion. This involves:
- Transitioning from workspace to joint space.
- Interpolating between initial and final joint configurations $\Theta_I$ and $\Theta_F$ using a 4-5-6-7 polynomial: 
  $\Theta(t) = \Theta_I + (\Theta_F - \Theta_I) p(t_{norm})$
  where $p(t) = -20t^7 + 70t^6 - 84t^5 + 35t^4$.
- Alternatively, cubic splines can interpolate a trajectory through multiple target points:
  $p_i(t) = a_{i0} + a_{i1}(t − t_i) + a_{i2}(t − t_i)^2 + a_{i3}(t − t_i)^3$
  for $t \in [t_i, t_{i+1}]$.

## 2-Fingered Gripper
The 2-fingered gripper attached to the DPR is based on the US 9,327,411 B2 Robotic Gripper Patent and Hand-E gripper by ROBOTIQ. Key features include:
- **Design**: Facilitates gripping small items from various angles using a pinion and rack mechanism powered by a Nema17 HS8401 stepper motor.
- **Components**: Utilizes graphite bushings, hard chrome shafts, and a Force-Sensitive Resistor (FSR) to measure grip force.
- **Fabrication**: Made from heat-resistant ABS plastic via 3D printing.
- **Control**: Operated through an Arduino Uno with serial communication, offering precise control over the gripper’s aperture and orientation. It has a maximum opening of 60mm and uses a 5-meter shielded cable for stable communication.

The integration of this gripper with the DPR enables coordinated object manipulation and orientation, enhancing the robot's functional versatility.

---
## The Pick-and-place Experimental Setup
The initial setup includes a partially filled box in a random position and orientation with scattered pieces of chocolate, all placed in the Delta Parallel Robot workspace. The robot’s movement is directed by classical trajectory planning methods, such as the 4-5-6-7 interpolating polynomial or cubic spline. To allow the robot to interact with target objects, a two-fingered gripper is mounted on the end-effector. The gripper will be controlled with a data cable connected to an Arduino kit. The generated results will be transmitted wirelessly to the robot using the Transmission Control Protocol (TCP).

---
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

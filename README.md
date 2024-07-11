# Pick-and-Place Operation by a Two-Fingered Robot using Faster R-CNN

## Table of Contents
1. [Project Overview](#project-overview)
2. [Contents](#contents)
3. [Tools and Libraries](#tools-and-libraries)
4. [Data Collection and Pre-Processing](#data-collection-and-pre-processing)
5. [Faster R-CNN Model](#faster-r-cnn-model)
6. [The Pick-and-place Experimental Setup](#the-pick-and-place-experimental-setup)
7. [Acknowledgments](#acknowledgments)

## Project Overview

This project aims to use the Faster R-CNN object detection model for the Food Packaging Process, with a specific focus on packaging various types of chocolates in a factory's production line. The objective is to identify chocolates in images captured by a camera to automate the pick-and-place task using a two-fingered gripper and a delta parallel robot. The primary aim is to enhance the precision and efficacy of detecting and packaging vulnerable soft chocolates during the pick-and-place operation without causing any damage while maintaining stringent hygiene standards and reducing the need for manual labor.

## Contents

- `Faster R-CNN-[Chocolate Detection].ipynb`: Jupyter notebook containing the implementation of the Faster R-CNN model for chocolate detection.

---

## Tools and Libraries

- Python 3.7+
- PyTorch
- scikit-learn
- OpenCV
- NumPy
- Pandas
- albumentations
- Roboflow

---

## Data Collection and Pre-Processing

- **Image Collection**: Data consisting of raw images of chocolates from six different brands with varying shapes were captured using an [ODROID USB CAM 720P HD](https://en.odroid.se/products/usb-kamera-720p) camera mounted on a robotic arm. The camera has a resolution of $1280 \times 720$, and the RGB images are saved in JPEG format. All chocolates were placed on a consistent background to ensure the model focused only on the target objects.

- **Pre-processing**: Raw images were pre-processed using the following techniques:
    - **Resizing:** Images are resized to $800 \times 600$ pixels.
    - **Normalization:** Pixel values are normalized to the range $[0, 1]$.
    - **Filtering:** Sharpening and blurring filters are applied in sequence to enhance images and reduce noise.

- **Image Annotation**: Each image was paired with XML annotation files created using the [Roboflow](https://github.com/roboflow) annotation tool adhering to the PASCAL VOC standard to precisely delineate the boundaries and labels of individual chocolate pieces. These annotation files include details like the image file name, dimensions (width, height, depth), and specifics about each chocolate in the image, such as its label and the bounding box coordinates recorded as $(x_{min}, y_{min}, x_{max}, y_{max})$.

    The initial dataset directory is Structured as follows:
   ```
   dataset/
       images/
           img1.jpg
           img1.xml
           img2.jpg
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

The **Data Augmentation** script in the `Faster R-CNN-[Chocolate Detection].ipynb` Jupyter Notebook provides a comprehensive data augmentation and preparation pipeline for the original images. It utilizes the *Roboflow API* to access the annotated dataset, uses *Albumentations* for image augmentations, and splits the augmented images into training, validation, and test sets. This process significantly increased the size and diversity of the dataset and also improved the generalization of the Faster R-CNN object detection model.

---

## Faster R-CNN Model

Faster R-CNN (Region-based Convolutional Neural Networks) is an efficient object detection model that combines region proposal networks (RPNs) with Fast R-CNN. The project uses a pre-trained Faster R-CNN model with a ResNet-50 backbone, previously trained on the COCO 2017 dataset. The process involves:

1. **Feature extraction:** ResNet-50 backbone network processes the input image and creates a feature map.
2. **Proposal Region Network (PRN):** Generates potential bounding boxes and classifies regions.
3. **Fast R-CNN:** Applies Region of Interest (ROI) Pooling, classifies objects, and refines the coordinates of the bounding boxes.
4. **Non-maximum suppression (NMS):** Eliminates redundant detections and removes overlapping boxes with lower confidence scores.

This architecture is visually illustrated in the figure below, and it successfully identifies samples from each of the six distinct chocolate brands presented in the dataset.

![img4](https://github.com/Sarmadzandi/Faster-R-CNN/assets/44917340/b088c0fa-c67a-4210-b4a7-7eeea3db8142)

---

## Data Preparation for Model Training
To train the Faster R-CNN model, it is essential to prepare the dataset meticulously. This involves organizing the data into a format that the model can understand and process efficiently. Below, we outline the steps involved in preparing the data for training the Faster R-CNN model.

### Dataset Organization
The dataset for training the Faster R-CNN model is divided into three subsets: training, validation, and test sets. Each subset contains images and corresponding annotations, which define the bounding boxes and labels for the objects present in the images. The dataset is organized as follows:

- **Training Set**: 832 images with 4263 annotated objects, averaging 5.12 objects per image.
- **Validation Set**: 208 images with 761 objects, averaging 3.66 objects per image.
- **Test Set**: 260 images with 940 objects, averaging 3.62 objects per image.

#### Dataset Statistics

| Set        | Total Images | Total Objects | Avg. Objects/Image |
|------------|--------------|---------------|--------------------|
| Train      | 832          | 4263          | 5.12               |
| Validation | 208          | 761           | 3.66               |
| Test       | 260          | 940           | 3.62               |
| Overall    | 1300         | 5964          | 4.59               |

#### Object Distribution
The dataset includes various types of chocolates, each represented by a different label. The distribution of objects across the training, validation, and test sets is detailed below:

| Object          | Train (%) | Validation (%) | Test (%) | Overall (%)  |
|-----------------|-----------|----------------|----------|--------------|
| Section         | 1206 (28.29) | 270 (35.48) | 324 (34.47) | 1800 (30.18) |
| Shoniz          | 469 (11.00)  | 106 (13.93) | 125 (13.30) | 700 (11.74)  |
| Diamond         | 396 (9.29)   | 95 (12.48)  | 109 (11.60) | 600 (10.06)  |
| Domenica Black  | 395 (9.26)   | 92 (12.09)  | 113 (12.02) | 600 (10.06)  |
| Tickers         | 387 (9.08)   | 91 (11.96)  | 122 (12.98) | 600 (10.06)  |
| Triperz         | 379 (8.89)   | 101 (13.27) | 120 (12.77) | 600 (10.06)  |
| Domenica Blue   | 266 (6.24)   | 61 (8.02)   | 73 (7.77)   | 400 (6.71)   |
| Box             | 201 (4.71)   | 45 (5.91)   | 54 (5.74)   | 300 (5.03)   |

---

## Training the Faster R-CNN Model
With the dataset prepared, the next step involves training the Faster R-CNN model. This process encompasses several key components, including model initialization, optimizer setup, and the training loop.

### Model Initialization
The Faster R-CNN model is initialized with a ResNet-50 backbone, pre-trained on the COCO dataset. This initialization leverages transfer learning, allowing the model to benefit from features learned on a large and diverse dataset. The final layer of the model is modified to output predictions for the nine classes in our dataset (eight chocolate types plus the background class).

### Optimizer and Learning Rate Scheduler
An optimizer is required to update the model parameters based on the computed gradients during backpropagation. For this task, we use the Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.005, momentum of 0.9, and weight decay of 0.0005. These hyperparameters are chosen to balance the speed of convergence with the stability of the training process.

To dynamically adjust the learning rate during training, a learning rate scheduler is employed. The scheduler reduces the learning rate by a factor of 10 every three epochs, allowing the model to fine-tune its weights more precisely in later stages of training.

### Training Loop
The training loop iterates over the dataset for a specified number of epochs (in this case, ten). During each epoch, the model is trained on the training set and evaluated on the validation set. Key steps in the training loop include:

1. **Forward Pass**: The input images are fed into the model to obtain predictions, including class labels and bounding box coordinates.
2

. **Loss Computation**: The loss function calculates the difference between the predicted and ground truth values for both classification and bounding box regression. The total loss is a weighted sum of these components.
3. **Backpropagation**: The gradients of the loss with respect to the model parameters are computed and propagated backward through the network.
4. **Parameter Update**: The optimizer updates the model parameters based on the computed gradients.
5. **Learning Rate Adjustment**: The learning rate scheduler adjusts the learning rate according to the predefined schedule.
6. **Checkpointing**: The model's state is periodically saved to disk, allowing for resumption of training in case of interruptions.

Throughout the training process, metrics such as loss, accuracy, and mean Average Precision (mAP) are tracked to monitor the model's performance.

## Evaluation of the Faster R-CNN Model
After training, the Faster R-CNN model is evaluated on the test set to assess its performance. The evaluation process involves calculating key metrics, including precision, recall, and mAP. These metrics provide insights into the model's ability to accurately detect and classify objects in new images.

### Non-Maximum Suppression (NMS)
Non-Maximum Suppression (NMS) is a crucial post-processing step applied during evaluation. It aims to eliminate redundant bounding boxes that overlap significantly with each other. NMS retains only the bounding box with the highest confidence score, thus improving the precision of the model's predictions.

### Precision, Recall, and mAP
- **Precision**: The ratio of true positive detections to the total number of detections. It measures the accuracy of the model in identifying true objects without false positives.
- **Recall**: The ratio of true positive detections to the total number of ground truth objects. It measures the model's ability to detect all objects in the image.
- **Mean Average Precision (mAP)**: The average of the precision values across all classes and IoU thresholds. It provides a comprehensive measure of the model's overall performance.

## Conclusion
Faster R-CNN offers a powerful and efficient solution for object detection tasks, combining region proposal networks with Fast R-CNN to achieve high accuracy. Its robust architecture, combined with meticulous data preparation and training processes, ensures reliable performance across diverse applications, such as automated chocolate packaging systems.

---

## Delta Parallel Robot Structure and Components

The Delta Parallel Robot (DPR) workspace (shown in the figure below), consists of a parallel configuration with three upper and three lower arms. Each upper arm connects to the base plate via a revolute joint and to the lower arm with a universal joint. The lower arms connect to a traveling plate using another universal joint, forming three kinematic chains that provide 3 degrees of freedom (DOF) along the x, y, and z axes.

<div style="text-align: center;">
  <table style="margin: 0 auto; border-spacing: 10px;">
    <tr>
      <td style="padding: 10px;">
        <img src="https://github.com/Sarmadzandi/Faster-R-CNN/assets/44917340/6e399814-9d1b-4d00-ae05-51d4743d0b0b" alt="Delta Parallel Robot and 2-fingered Gripper" style="width: 100%; height: auto; vertical-align: top;" />
      </td>
    </tr>
    <tr>
      <td>
        Delta Parallel Robot and Two-Fingered Gripper - Human and Robot Interaction Laboratory
        <a href="https://taarlab.com/" target="_blank">(TaarLab)</a>, University of Tehran.
      </td>
    </tr>
  </table>
</div>

### DPR Kinematics

- **Inverse Kinematics (IK)**: Determines joint configurations for a desired end-effector position.
- **Forward Kinematics (FK)**: Computes the end-effector's position based on given joint configurations.
- The main actuators are on the revolute joints connecting the upper arms to the base.
- IK is modeled using the equation:

$$
\overline{O A_i}+\overline{A_i B_i}+\overline{B_i C_i}+\overline{C_i E}+\overline{E O}=0
$$

### DPR Trajectory Planning

Trajectory planning involves calculating a sequence of end-effector positions over time for smooth motion. This involves:
- Transitioning from workspace to joint space.
- Interpolating between initial and final joint configurations $\Theta_I$ and $\Theta_F$ using a 4-5-6-7 polynomial:
  
$$
\Theta(t) = \Theta_I + (\Theta_F - \Theta_I) p(t_{norm}) \quad where \quad p(t) = -20t^7 + 70t^6 - 84t^5 + 35t^4
$$

- Alternatively, cubic splines can interpolate a trajectory through multiple target points:

$$
p_i(t) = a_{i0} + a_{i1}(t − t_i) + a_{i2}(t − t_i)^2 + a_{i3}(t − t_i)^3 \quad for \quad t \in [t_i, t_{i+1}]
$$

### Two-Fingered Gripper

The Two-Fingered gripper attached to the DPR is based on the US 9,327,411 B2 Robotic Gripper Patent and Hand-E gripper by ROBOTIQ. Key features include:
- **Design**: Facilitates gripping small items from various angles using a pinion and rack mechanism powered by a Nema17 HS8401 stepper motor.
- **Components**: Utilizes graphite bushings, hard chrome shafts, and a Force-Sensitive Resistor (FSR) to measure grip force.
- **Fabrication**: Made from heat-resistant ABS plastic via 3D printing.
- **Control**: Operated through an Arduino Uno with serial communication, offering precise control over the gripper’s aperture and orientation. It has a maximum opening of 60mm and uses a 5-meter shielded cable for stable communication.

The integration of this gripper with the DPR enables coordinated object manipulation and orientation, enhancing the robot's functional versatility.

---

## The Pick-and-place Experimental Setup

The initial setup includes a partially filled box in a random position and orientation with scattered pieces of chocolate, all placed in the Delta Parallel Robot workspace. The robot’s movement is directed by classical trajectory planning methods, such as the 4-5-6-7 interpolating polynomial or cubic spline. To allow the robot to interact with target objects, a two-fingered gripper is mounted on the end-effector. The gripper will be controlled with a data cable connected to an Arduino kit. The generated results will be transmitted wirelessly to the robot using the Transmission Control Protocol (TCP).

---


---
  
## Acknowledgments

- Special thanks to [Prof. Tale Masouleh](https://scholar.google.com/citations?hl=es&user=gkiFy20AAAAJ&view_op=list_works&sortby=pubdate) for his valuable guidance and to the members of the Taarlab laboratory for their support throughout this research project.

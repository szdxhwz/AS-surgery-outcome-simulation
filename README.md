# AS Simulation

## Introduction

In the surgical planning for ankylosing spondylitis (AS), the simulation of postoperative whole-body posture is very helpful for surgeons to obtain a global estimation of the outcome. We proposed a method that integrates the 3D body surface and the CT volumetric data to facilitate surgery outcome simulations for individuals diagnosed with AS. Firstly, we captured 3D body surface information from AS patients using a depth-sensing camera system. Subsequently, we designed and implemented a surface-to-volume pipeline to register the torso surface region and the CT data. With them registered, the surgical plan formulated by the surgeons in the CT volumetric data can then be transferred to the 3D surface. Finally, in order to obtain a realistic 3D surface after the surgery, the log-Euclidean transformation scheme was adopted. Our approach offers the advantage of intuitively demonstrating the anticipated surgical corrections.

AS Simulation is an open platform for the simulation of postoperative whole-body. Thanks to the openess of Slicer, the AS Simulation extension could also be further extended by you.

## Installation

Import the module into the 3D Slicer. Install python libraries in Slicer according to [requirements.txt](https://github.com/szdxhwz/AS-surgery-outcome-simulation/blob/main/requirements.txt).  This can be done in the Slicer Python console with, e.g.,
```
pip_install ("open3d")
```
Clip and paste the [serverCode](https://github.com/szdxhwz/AS-surgery-outcome-simulation/tree/main/serverCode) folder onto the server,Change the server address in the [As_Simulation.py](https://github.com/szdxhwz/AS-surgery-outcome-simulation/blob/main/As_Simulation/As_Simulation.py) and [predict.py](https://github.com/szdxhwz/AS-surgery-outcome-simulation/blob/main/serverCode/SubdivNet-master/predict.py) to your own server address.


## Modules
This module mainly includes three functions: model segmentation, model registration and postoperative simulation. Model segmentation provides two options: manual segmentation and deep learning segmentation. As for deep learning segmentation, because it is easy to crash by directly implanting the deep learning module in 3D Slicer, the method we take is to deploy the deep learning model on the server in the same local area network and obtain the segmentation result through socket communication.You can use manual or automatic segmentation depending on your situation.

## Usage

We have created a video to introduce the use of the module ([Module demo.mp4](https://www.youtube.com/watch?v=03WNj2pbsAs)), please follow the video process to operate.


## Example data
The sample data is in the [test_data](https://github.com/szdxhwz/AS-surgery-outcome-simulation/tree/main/test_data) folder, where the [Preoperative_data](https://github.com/szdxhwz/AS-surgery-outcome-simulation/tree/main/test_data/Preoperative_data) folder contains the patient's preoperative 3D mesh, and the [Postoperative_data](https://github.com/szdxhwz/AS-surgery-outcome-simulation/tree/main/test_data/Preoperative_data) folder contains the patient's postoperative 3D mesh. The Preoperative CT data is in [here](https://drive.google.com/file/d/1EFmzJlFl8DR4SG3KAqznszYpTDYIwsHr/view?usp=sharing).To protect patient privacy, the patient's face has been blurred. You can simulate the preoperative data and compare it with the postoperative data.


# AS Simulation

## Introduction

In the surgical planning for ankylosing spondylitis (AS), the simulation of postoperative whole-body posture is very helpful for surgeons to obtain a global estimation of the outcome. We proposed a method that integrates the 3D body surface and the CT volumetric data to facilitate surgery outcome simulations for individuals diagnosed with AS. Firstly, we captured 3D body surface information from AS patients using a depth-sensing camera system. Subsequently, we designed and implemented a surface-to-volume pipeline to register the torso surface region and the CT data. With them registered, the surgical plan formulated by the surgeons in the CT volumetric data can then be transferred to the 3D surface. Finally, in order to obtain a realistic 3D surface after the surgery, the log-Euclidean transformation scheme was adopted. Our approach offers the advantage of intuitively demonstrating the anticipated surgical corrections.

AS Simulation is an open platform for the simulation of postoperative whole-body.Thanks to the openess of Slicer, the BigImage extension could also be further extended by you.

## Installation

Install python requirements in Slicer.For example, enter pip_install ("open3d") in the Slicer Python console.


## Modules
This module mainly includes three functions: model segmentation, model registration and postoperative simulation. Model segmentation provides two options: manual segmentation and deep learning segmentation. Because it is easy to crash by directly implanting the deep learning module in 3D Slicer, the method we take is to deploy the deep learning model on the server and obtain the segmentation result through socket communication.

## Usage

### Example data
Example large scale wholse slide image can be downloaded at, e.g., the
[OpenSlide
website](https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs
"Brightfield WSI")

### Large whole slide image viewing
Swith to the "BigImageViewer" module in the BigImage category. As shown in the module panel, select the WSI file to view in the "Select WSI" box. Then click "Load WSI" below.

![image](https://user-images.githubusercontent.com/920557/174559913-77ccaee3-5063-4fa5-b562-dd1ad3b24236.png)

One can then view different region/scale of the image, using mouse dragging and mosue wheeling, as shown below:

![image](https://user-images.githubusercontent.com/89077084/174545844-83a5f601-32ca-4d88-b328-b3a0cba0e922.png)
![image](https://user-images.githubusercontent.com/89077084/174545870-063ae0a8-2e3d-49bd-8d61-08ca19c5dbb6.png)

### Staining decomposition
Histopathology images are often stained using different dyes. When a WSI is stained using multiple dyes, the different stains can be computationally separated using the module "ColorDecomposition", whose panel is shown below.

![image](https://user-images.githubusercontent.com/920557/174555656-1e227e15-2110-4bf1-8b9d-74b1fdddc823.png)

There are various options for color decomposition. This particular WSI is stained with H-E. Its original appearance is:
![image](https://user-images.githubusercontent.com/920557/174556082-81738b77-87f5-4111-bf31-bbca18501501.png)

After decomposition, the hematoxylin content is shown in gray-scale as:
![image](https://user-images.githubusercontent.com/920557/174556323-eb064126-c40b-48a2-95bd-f4a0c77d60b3.png)
where the dark regions corresponding to the high hematoxylin content.

If the eosin chanel is wanted, one can switch the output chanel in the module panel to the 2nd chanel, and the result will be like:
![image](https://user-images.githubusercontent.com/920557/174556464-e4e1d6d0-f1c3-4222-ad68-f490a520ae98.png)

### Zarr image reading/writing

The extension contains an experimental module ([NgffImageIO](https://github.com/gaoyi/SlicerBigImage/blob/main/NgffImageIO/NgffImageIO.py))
for reading [OME-NGFF](https://ngff.openmicroscopy.org/latest/) file format. Currently, only a simple image array can be saved and loaded
in Zarr format (with the `.zarr` file extension, with `ZipStorage` class), but we do not follow the NGFF specification yet.

This module may be used in the future instead of OpenSlide to make dependencies simpler, and to store more complete metadata.

## Citation

If you find this extension helpful please cite this paper:

Xiaxia Yu, Bingshuai Zhao, Haofan Huang, Mu Tian, Sai Zhang, Hongping Song, Zengshan Li, Kun Huang, Yi Gao, "An Open Source Platform for Computational Histopathology," in IEEE Access, vol. 9, pp. 73651-73661, 2021, doi: 10.1109/ACCESS.2021.3080429.

import logging
import os
import open3d as o3d
import vtk
import numpy as np
import pycpd
from scipy.linalg import expm,logm
from math import *
import pymeshlab as ml
import trimesh
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import qt
import json
import socket
import json
import random
from pathlib import Path
import pickle
# from jittor.dataset import Dataset

# import numpy as np
# import trimesh
# from scipy.spatial.transform import Rotation
# import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

# import argparse
# import random

# import numpy as np

# import jittor as jt
# import jittor.nn as nn
# from jittor.optim import Adam, SGD
# from jittor.lr_scheduler import MultiStepLR
# jt.flags.use_cuda = 1
# jt.cudnn.set_max_workspace_ratio(0.0)
# from tqdm import tqdm
# from subdivnet.dataset import SegmentationDataset
# from subdivnet.deeplab import MeshDeepLab
# from subdivnet.deeplab import MeshVanillaUnet
# from subdivnet.utils import to_mesh_tensor
# from subdivnet.utils import save_results
# from subdivnet.utils import update_label_accuracy
# from subdivnet.utils import compute_original_accuracy
# from subdivnet.utils import SegmentationMajorityVoting
# from subdivnet.mesh_tensor import MeshTensor


#
# jizhu
#
# def mesh_normalize(mesh: trimesh.Trimesh):
#         vertices = mesh.vertices - mesh.vertices.min(axis=0)
#         vertices = vertices / vertices.max()
#         mesh.vertices = vertices
#         return mesh

# def to_mesh_tensor(meshes,feats,fs):
#     return MeshTensor(jt.int32(meshes), 
#                     jt.float32(feats), 
#                     jt.int32(fs))


# def load_mesh(path, normalize=False, augments=[], request=[]):
#     mesh = trimesh.load_mesh(path, process=False)

#     if normalize:
#         mesh = mesh_normalize(mesh)

#     F = mesh.faces
#     V = mesh.vertices
#     Fs = mesh.faces.shape[0]

#     face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)
#     # corner = V[F.flatten()].reshape(-1, 3, 3) - face_center[:, np.newaxis, :]
#     vertex_normals = mesh.vertex_normals
#     face_normals = mesh.face_normals
#     face_curvs = np.vstack([
#         (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
#         (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
#         (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
#     ])
    
#     feats = []
#     if 'area' in request:
#         feats.append(mesh.area_faces)
#     if 'normal' in request:
#         feats.append(face_normals.T)
#     if 'center' in request:
#         feats.append(face_center.T)
#     if 'face_angles' in request:
#         feats.append(np.sort(mesh.face_angles, axis=1).T)
#     if 'curvs' in request:
#         feats.append(np.sort(face_curvs, axis=0))

#     feats = np.vstack(feats)

#     return mesh.faces, feats, Fs

# def save_results(preds,segment_colors):
#     mesh = trimesh.load_mesh("D:/kaifa/jizhu/jizhu/triResult1_maps.obj", process=False)
#     for i in range(preds.shape[0]):
#         mesh.visual.face_colors[:, :3] = segment_colors[preds[i, :mesh.faces.shape[0]]]
#         mesh.export("D:/kaifa/jizhu/jizhu/triResult1_maps_test.ply")
def sendData(data, host, port):
    print("####  Just enter sendData")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))

        print('(((( in sendData, before pickle')
        data_string = pickle.dumps(data)
        s.sendall(data_string)
        print('(((( in sendData, Data Sent to Server')

    return


def expectData(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))

        data = []
        print('after data = []')
        while True:
            #print('here 1')
            packet = s.recv(4096)
            #print('here 2')
            if not packet:
                break

            #print('here 3')
            data.append(packet)

        print("Data received on client")

        #print('Received', repr(data))

        image = pickle.loads(b"".join(data))

    return image


class jizhu(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "jizhu"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#jizhu">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        


#
# Register sample data sets in Sample Data module
#
#
# jizhuWidget
#

class jizhuWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/jizhu.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        

        # self.ui.inputSelector.nodeTypes = ( ("vtkMRMLModelNode"), "")
        # self.ui.inputSelector.addEnabled = False
        # self.ui.inputSelector.removeEnabled = False
        # self.ui.inputSelector.noneEnabled = True
        # self.ui.inputSelector.showHidden = False
        # self.ui.inputSelector.setMRMLScene(slicer.mrmlScene)

        

        #self.ui.pushButton_5.connect('clicked(bool)', self.button_1_clicked)
        self.ui.pushButton_5.clicked.connect(self.button_1_clicked)
        self.ui.pushButton_6.clicked.connect(self.button_2_clicked)
        self.ui.pushButton_7.clicked.connect(self.button_3_clicked)
        self.ui.pushButton_4.clicked.connect(self.button_4_clicked)
        self.ui.pushButton_8.clicked.connect(self.button_5_clicked)
        self.ui.pushButton_9.clicked.connect(self.button_6_clicked)
        self.ui.pushButton.clicked.connect(self.button_7_clicked)
        self.ui.pushButton_12.clicked.connect(self.button_8_clicked)
        self.ui.pushButton_13.clicked.connect(self.button_9_clicked)
        self.ui.pushButton_3.clicked.connect(self.button_10_clicked)
        

        
    
    def button_1_clicked(self):
        global fileName1
        fileName1= qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(),"open")  #设置文件扩展名过滤,注意用双分号间隔
        self.ui.lineEdit_5.setText(fileName1)
        
    def button_2_clicked(self):
        global fileName2
        fileName2= qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(),"open")  #设置文件扩展名过滤,注意用双分号间隔
        self.ui.lineEdit_6.setText(fileName2)
    
    def button_3_clicked(self):
        pcd = o3d.io.read_triangle_mesh(fileName1)
        pcd.compute_vertex_normals()
        #o3d.visualization.draw_geometries_with_vertex_selection([pcd])
        o3d.visualization.draw_geometries_with_editing([pcd])
        ms = ml.MeshSet()
        ms.load_new_mesh(fileName2)
        print(type(float(self.ui.lineEdit_14.text)))
        ms.compute_selection_by_small_disconnected_components_per_face(nbfaceratio=float(self.ui.lineEdit_14.text))
        ms.meshing_remove_selected_vertices_and_faces()
        ms.save_current_mesh(fileName2)
        pcd1 = o3d.io.read_triangle_mesh(fileName2)
        pcd1.compute_vertex_normals()
        #o3d.visualization.draw_geometries_with_vertex_selection([pcd])
        o3d.visualization.draw_geometries_with_editing([pcd1])

    def button_4_clicked(self):
        global fileName3
        fileName3= qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(),"open")  #设置文件扩展名过滤,注意用双分号间隔
        self.ui.lineEdit_4.setText(fileName3)

    def button_5_clicked(self):
        global fileName4
        fileName4= qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(),"open")  #设置文件扩展名过滤,注意用双分号间隔
        self.ui.lineEdit_7.setText(fileName4)

    def button_6_clicked(self):
        global fileName5
        fileName5= qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(),"open")  #设置文件扩展名过滤,注意用双分号间隔
        self.ui.lineEdit_8.setText(fileName5)

    def button_7_clicked(self):
        pcd = o3d.io.read_point_cloud(fileName5)#CT.ply
        pcd3 = o3d.io.read_point_cloud(fileName4)#3D.ply

        # charu_point=np.asarray([-182.34615384615405, 162.05882352941178, -1483.076923076923])

        downpcd1 = pcd3.voxel_down_sample(voxel_size=float(self.ui.lineEdit_15.text))
        # o3d.visualization.draw_geometries([pcd])
        downpcd = pcd.voxel_down_sample(voxel_size=float(self.ui.lineEdit_15.text))
        print("The number of points after downsampling is:",np.asarray(downpcd1.points).shape[0], np.asarray(downpcd.points).shape[0])
        X=np.asarray(downpcd1.points)
        Y=np.asarray(downpcd.points)
        Y[:,:2]= -Y[:,:2]
        # Y=np.vstack((charu_point, Y))
        t1=np.array([[-227.12065253,-36.94591913,2709.37443481]])
        t1=t1.reshape(1,3)
        reg =pycpd.AffineRegistration(**{'X': X, 'Y': Y,'t':t1})
        data,(B,t) = reg.register()
        # print(data[0,:])
        # reg1 = pycpd.DeformableRegistration(**{'X': X, 'Y': data, 'alpha': 0.001, 'beta': 5})
        # data1,(G,W) = reg1.register()
        # print(data1[0,:])
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(data)
        pcd2.paint_uniform_color([0, 1, 0])
        downpcd.paint_uniform_color([1, 0, 0])
        downpcd1.paint_uniform_color([0, 0, 1])
        # print("s",s)
        np.savetxt(f"{fileName3}/B.txt",B)
        np.savetxt(f"{fileName3}/t.txt",t)
        #np.savetxt("G:\\ac\\ljw\\W.txt",W)
        o3d.io.write_point_cloud(f"{fileName3}/peizhun.ply",pcd2)



        o3d.visualization.draw_geometries([downpcd1,downpcd,pcd2], window_name="Registration",
                                        width=1200, height=800,
                                        left=50, top=50,
                                        mesh_show_back_face=False)
        
    def button_8_clicked(self):
        B1=np.loadtxt(f"{fileName3}/B.txt")
        t1=np.loadtxt(f"{fileName3}/t.txt")
        f = open(fileName6,encoding='utf-8') 
        data = json.load(f) 
        data1=data['markups'][0]["controlPoints"][0]["position"]
        Y=np.asarray(data1)
        Y_aff=np.dot(Y, B1) + t1
        mesh = o3d.io.read_triangle_mesh(fileName1)
        vertices = np.asarray(mesh.vertices)
        triangles=np.asarray(mesh.triangles)
        all_ids=vertices.shape[0]
        T = np.eye(4,dtype=float)
        v=np.ones((4,1))
        t = Y_aff
        a=0
        b=1
        c=1
        d=-t[1]-t[2]
        static_ids = [idx for idx in np.where((vertices[:, 0]*a+vertices[:, 1]*b+vertices[:, 2]*c+d)/c>0)[0]]  
        R =mesh.get_rotation_matrix_from_xyz(((np.pi*int(self.ui.lineEdit_9.text))/180,0,0))#欧拉角转变换矩阵
        print(R)
        #欧拉角转变换矩阵
        # p1 = np.array([[0.1,1],[300,1]])
        # y = np.array([[0.001],[500]])
        # a=solve(p1,y)[0,0]
        # b=solve(p1,y)[1,0]
        # print(a,b)

        for id in static_ids:
            T[:3,3] = t
            T[3,3] = 1
            v[:3,0]=np.asarray([vertices[id,0]-t[0],vertices[id,1]-t[1],vertices[id,2]-t[2]])
            w=abs(vertices[id,0]*a+vertices[id,1]*b+vertices[id,2]*c+d)/sqrt(a**2+b**2+c**2)
            w1=exp(-100/w)
            T[:3,:3] = np.asarray(expm(w1*logm(R)))
            #T[:3,:3]=R
            vertices[id]=(np.dot(T,v))[:3,0].reshape(3,)

        mesh.compute_vertex_normals()
        if self.ui.comboBox.currentIndex==0:
            o3d.io.write_triangle_mesh(f"{fileName3}/{self.ui.lineEdit_10.text}.obj",mesh)
            with open(f"{fileName3}/{self.ui.lineEdit_10.text}.mtl", 'r') as mtl_file:
            # Read in the contents of the file
                mtl_data = mtl_file.readlines()
            mtl_data.append(f"map_Kd {self.ui.lineEdit_13.text}\n")
            print(mtl_data,type(mtl_data))

            with open(f"{fileName3}/{self.ui.lineEdit_10.text}.mtl", 'w') as mtl_file:
                mtl_file.writelines(mtl_data)
        else:
            o3d.io.write_triangle_mesh(f"{fileName3}/{self.ui.lineEdit_10.text}.stl",mesh)
        o3d.visualization.draw_geometries([mesh])

    def button_9_clicked(self):
        global fileName6
        fileName6= qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(),"open")  #设置文件扩展名过滤,注意用双分号间隔
        self.ui.lineEdit_3.setText(fileName6)

    
    def button_10_clicked(self):
        HOST = '172.21.141.27'  # The server's hostname or IP address
        PORT = 6000       # The port used by the server
        mesh = trimesh.load_mesh(fileName1, process=False)
        f =np.asarray(mesh.faces)
        h=f.shape[0]
        
        v =np.asarray(mesh.vertices)
        
        h1=np.array([h,0,0])
        imgArray1=np.vstack((f,v))
        imgArray=np.vstack((imgArray1,h1))


        print("**** Before sending ****")
        sendData(imgArray, HOST, PORT)
        print("**** After sending ****")

        # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        #     s.connect((HOST, PORT))

        #     data_string = pickle.dumps(img)
        #     s.sendall(data_string)
        #     print('Data Sent to Server')
        newImageArray = expectData(HOST, PORT)

        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(newImageArray.shape)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")


        f1=newImageArray[:int(newImageArray[-1,0]),:]
        v1=newImageArray[int(newImageArray[-1,0]):int(newImageArray[-1,0]+newImageArray[-1,1]),:]
        
        mesh1 = trimesh.Trimesh(vertices = v1, faces = f1, process = False)
        mesh1.visual.face_colors[:, :3] =newImageArray[int(newImageArray[-1,0]+newImageArray[-1,1]):-1,:]
        mesh1.export("C:/Users/hwz/Desktop/test/3d_maps_segmentation.ply")






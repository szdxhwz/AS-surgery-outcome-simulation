import logging
import os
import open3d as o3d
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
import random
from pathlib import Path
import pickle
import operator

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


class As_Simulation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "As_Simulation"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Examples"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#As_Simulation">module documentation</a>.
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

class As_SimulationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/As_Simulation.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

       
        self.ui.pushButton_5.clicked.connect(self.button_1_clicked)
        self.ui.pushButton_6.clicked.connect(self.button_2_clicked)
        self.ui.pushButton_7.clicked.connect(self.button_3_clicked)
        self.ui.pushButton_4.clicked.connect(self.button_4_clicked)
        self.ui.pushButton_2.clicked.connect(self.button_7_clicked)
        
        self.ui.pushButton_12.clicked.connect(self.button_8_clicked)
        self.ui.pushButton_13.clicked.connect(self.button_9_clicked)
        self.ui.pushButton_3.clicked.connect(self.button_10_clicked)
        self.ui.pushButton_10.clicked.connect(self.button_5_clicked)
        self.ui.pushButton_11.clicked.connect(self.button_6_clicked)
        
        self.ui.lineEdit_14.setText("0.1")
        self.ui.lineEdit_16.setText("2")
        self.ui.lineEdit.setText("0.1")
        self.ui.lineEdit_17.setText("15")

        
    
    def button_1_clicked(self):
        global fileName1
        fileName1= qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(),"open")  
        self.ui.lineEdit_5.setText(fileName1)
        
    def button_2_clicked(self):
        global fileName2
        fileName2= qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(),"open")  
        self.ui.lineEdit_6.setText(fileName2)
    
    def button_3_clicked(self):
        pcd = o3d.io.read_triangle_mesh(fileName1)
        pcd.compute_vertex_normals()
        o3d.visualization.draw_geometries_with_editing([pcd])
        ms = ml.MeshSet()
        ms.load_new_mesh(fileName2)
        print(type(float(self.ui.lineEdit_14.text)))
        ms.compute_selection_by_small_disconnected_components_per_face(nbfaceratio=float(self.ui.lineEdit_14.text))
        ms.meshing_remove_selected_vertices_and_faces()
        ms.save_current_mesh(fileName2)
        pcd1 = o3d.io.read_triangle_mesh(fileName2)
        pcd1.compute_vertex_normals()
        o3d.visualization.draw_geometries_with_editing([pcd1])

    def button_4_clicked(self):
        global fileName3
        fileName3= qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(),"open")  
        self.ui.lineEdit_4.setText(fileName3)

    def button_5_clicked(self):
        global fileName4
        fileName4= qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(),"open")  
        self.ui.lineEdit_11.setText(fileName4)

    def button_6_clicked(self):
        global fileName5
        fileName5= qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(),"open")  
        self.ui.lineEdit_12.setText(fileName5)

    def button_7_clicked(self):
        pcd = o3d.io.read_point_cloud(fileName5)#CT.ply
        pcd3 = o3d.io.read_point_cloud(fileName4)#3D.ply

        downpcd1 = pcd3.voxel_down_sample(voxel_size=int(self.ui.lineEdit_17.text))
        downpcd = pcd.voxel_down_sample(voxel_size=int(self.ui.lineEdit_17.text))
        print("The number of points after downsampling is:",np.asarray(downpcd1.points).shape[0], np.asarray(downpcd.points).shape[0])
        X=np.asarray(downpcd1.points)
        Y=np.asarray(downpcd.points)
        
        reg =pycpd.AffineRegistration(**{'X': X, 'Y': Y})
        data,(B,t) = reg.register()
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(data)
        pcd2.paint_uniform_color([0, 1, 0])
        downpcd.paint_uniform_color([1, 0, 0])
        downpcd1.paint_uniform_color([0, 0, 1])
       
        np.savetxt(f"{fileName3}/B.txt",B)
        np.savetxt(f"{fileName3}/t.txt",t)
       
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
        print(Y_aff)
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
        R =mesh.get_rotation_matrix_from_xyz(((np.pi*int(self.ui.lineEdit_9.text))/180,0,0))
        print(R)
       

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

        o3d.visualization.draw_geometries([mesh],
                                window_name="Simulation Results",
                                width=1024, height=768,
                                left=50, top=50,mesh_show_wireframe=True,
                                point_show_normal=True)
        
        

    def button_9_clicked(self):
        global fileName6
        fileName6= qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(),"open")  
        self.ui.lineEdit_3.setText(fileName6)

    
    def button_10_clicked(self):
        HOST = '172.21.141.27'  # The server's hostname or IP address
        PORT = 6000       # The port used by the server
        mesh = ml.MeshSet()
        mesh.load_new_mesh(fileName1)  
        mesh1 = ml.MeshSet()
        mesh1.load_new_mesh(fileName2)  
        v= mesh.current_mesh().vertex_matrix()
        f = mesh.current_mesh().face_matrix()
        v1= mesh1.current_mesh().vertex_matrix()
        f1 = mesh1.current_mesh().face_matrix()

        h=f.shape[0]
        c=v.shape[0]

        h2=f1.shape[0]
        
        h1=np.array([h,h+c,h+c+h2])
        h3=np.array([float(self.ui.lineEdit_16.text),0,0])
        imgArray1=np.vstack((f,v))
        imgArray=np.vstack((imgArray1,f1))
        imgArray2=np.vstack((imgArray,v1))
        imgArray3=np.vstack((imgArray2,h1))
        imgArray4=np.vstack((imgArray3,h3))
        print(h1)


        print("**** Before sending ****")
        sendData(imgArray4, HOST, PORT)
        print("**** After sending ****")

        newImageArray = expectData(HOST, PORT)

        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(newImageArray.shape)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")


        f1=newImageArray[:int(newImageArray[-2,0]),:]
        v1=newImageArray[int(newImageArray[-2,0]):int(newImageArray[-2,1]),:]
        f2=newImageArray[int(newImageArray[-2,2]):int(newImageArray[-1,0]),:]
        v2=newImageArray[int(newImageArray[-1,0]):int(newImageArray[-1,1]),:]
        
        mesh1 = trimesh.Trimesh(vertices = v1, faces = f1, process = False)
        mesh1.visual.face_colors[:, :3] =newImageArray[int(newImageArray[-2,1]):int(newImageArray[-2,2]),:]
        mesh2 = trimesh.Trimesh(vertices = v2, faces = f2, process = False)
        mesh2.visual.face_colors[:, :3] =newImageArray[int(newImageArray[-1,1]):-2,:]
        
        mesh1.export(f"{fileName3}/3d_maps_segmentation.ply")
        mesh2.export(f"{fileName3}/ct_maps_segmentation.ply")
        print("OK")
        mesh_3d=o3d.io.read_triangle_mesh(f"{fileName3}/3d_maps_segmentation.ply")
        o3d.visualization.draw_geometries([mesh_3d],
                                window_name="Simulation Results",
                                width=1024, height=768,
                                left=50, top=50,mesh_show_wireframe=True,
                                point_show_normal=True)
        mesh_ct=o3d.io.read_triangle_mesh(f"{fileName3}/ct_maps_segmentation.ply")
        o3d.visualization.draw_geometries([mesh_ct],
                                window_name="Simulation Results",
                                width=1024, height=768,
                                left=50, top=50,mesh_show_wireframe=True,
                                point_show_normal=True)

        label=[]
        mesh = trimesh.load_mesh(f"{fileName3}/3d_maps_segmentation.ply", process=False)
        num=mesh.visual.face_colors.shape[0]
        mesh2=np.array(mesh.faces)
        print(mesh2.shape)
        print(np.array(mesh.vertices).shape)
        print(num)
        for i in range(num):
            if operator.eq(list(mesh.visual.face_colors[i, :3]),[117, 142, 48]) :
                label.append(True)
            else:
                label.append(False)
        print(len(label))
        mesh.update_faces(label)
        print(np.array(mesh.vertices).shape)
        mesh.remove_unreferenced_vertices()
        print(np.array(mesh.vertices).shape)
        mesh.export(f"{fileName3}/cropped_1.ply")

        


        label1=[]
        mesh1 = trimesh.load_mesh(f"{fileName3}/ct_maps_segmentation.ply", process=False)
        num1=mesh1.visual.face_colors.shape[0]
        mesh21=np.array(mesh1.faces)
        print(mesh21.shape)
        print(np.array(mesh1.vertices).shape)
        print(num1)
        for i in range(num1):
            if operator.eq(list(mesh1.visual.face_colors[i, :3]),[117, 142, 48]) :
                label1.append(True)
            else:
                label1.append(False)
        print(len(label1))
        mesh1.update_faces(label1)
        print(np.array(mesh1.vertices).shape)
        mesh1.remove_unreferenced_vertices()
        print(np.array(mesh1.vertices).shape)
        mesh1.export(f"{fileName3}/cropped_2.ply")
    






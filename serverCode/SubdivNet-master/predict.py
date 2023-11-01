import sys
import socket,pickle
from jittor.dataset import Dataset
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import random
import pymeshlab as ml
import jittor as jt
import jittor.nn as nn
from jittor.optim import Adam, SGD
from jittor.lr_scheduler import MultiStepLR
jt.flags.use_cuda = 1
jt.cudnn.set_max_workspace_ratio(0.0)
from tqdm import tqdm
from subdivnet.dataset import SegmentationDataset
from subdivnet.deeplab import MeshDeepLab
from subdivnet.deeplab import MeshVanillaUnet
from subdivnet.utils import to_mesh_tensor
from subdivnet.utils import save_results
from subdivnet.utils import update_label_accuracy
from subdivnet.utils import compute_original_accuracy
from subdivnet.utils import SegmentationMajorityVoting
from subdivnet.mesh_tensor import MeshTensor
from datagen_maps import make_MAPS_shape


def mesh_normalize(mesh: trimesh.Trimesh):
        vertices = mesh.vertices - mesh.vertices.min(axis=0)
        vertices = vertices / vertices.max()
        mesh.vertices = vertices
        return mesh

def to_mesh_tensor(meshes,feats,fs):
    return MeshTensor(jt.int32(meshes), 
                    jt.float32(feats), 
                    jt.int32(fs))


def load_mesh(path, normalize=False, augments=[], request=[]):
    mesh = trimesh.load_mesh(path, process=False)

    if normalize:
        mesh = mesh_normalize(mesh)

    F = mesh.faces
    V = mesh.vertices
    Fs = mesh.faces.shape[0]

    face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)
    # corner = V[F.flatten()].reshape(-1, 3, 3) - face_center[:, np.newaxis, :]
    vertex_normals = mesh.vertex_normals
    face_normals = mesh.face_normals
    face_curvs = np.vstack([
        (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
    ])
    
    feats = []
    if 'area' in request:
        feats.append(mesh.area_faces)
    if 'normal' in request:
        feats.append(face_normals.T)
    if 'center' in request:
        feats.append(face_center.T)
    if 'face_angles' in request:
        feats.append(np.sort(mesh.face_angles, axis=1).T)
    if 'curvs' in request:
        feats.append(np.sort(face_curvs, axis=0))

    feats = np.vstack(feats)

    return mesh.faces, feats, Fs



def waitForData(socketThe):
    print("before accept")
    conn, addr = socketThe.accept()
    with conn:
        print('Connected by', addr)
        data = []
        print("server 1")
        while True:
            #print("server 2")
            packet = conn.recv(4096)
            #print("server 3")
            if not packet:
                break

            #print("server 4")
            data.append(packet)

        print("server 5")
        image = pickle.loads(b"".join(data))

        # for some reason, a (x, y, 3) RGB image will appear as (1, x,
        # y, 3) here. So need to squeeze here
        image = np.squeeze(image)

        # plt.figure()
        # plt.imshow(image)
        # plt.title("received image on server")
        # plt.show()


        return image
    
def sendProcessDataToClient(socketThe, data):
    conn, addr = socketThe.accept()
    with conn:
        data_string = pickle.dumps(data)
        print("Image pickled on server")
        conn.sendall(data_string)
        #conn.sendall(b"asdf")
        print("Image sent from server")

    return

def process(data):
    f1=data[:int(data[-2,0]),:]
    v1=data[int(data[-2,0]):int(data[-2,1]),:]
    f2=data[int(data[-2,1]):int(data[-2,2]),:]
    v2=data[int(data[-2,2]):-2,:]
    print("????",data[-2,:])
    mesh1 = trimesh.Trimesh(vertices = v1, faces = f1, process = False)
    mesh1.export("./texture_mesh_3d.ply")
    mesh2 = trimesh.Trimesh(vertices = v2, faces = f2, process = False)
    mesh2.export("./texture_mesh_ct.ply")
    ms = ml.MeshSet()
    ms.load_new_mesh("./texture_mesh_3d.ply")
    ms.generate_surface_reconstruction_vcg(voxsize=ml.Percentage(float(data[-1,0])))
    ms.save_current_mesh("./texture_mesh_3d_vcg.ply")

    ms1 = ml.MeshSet()
    ms1.load_new_mesh("./texture_mesh_ct.ply")
    ms1.generate_surface_reconstruction_vcg(voxsize=ml.Percentage(float(data[-1,0])))
    ms1.save_current_mesh("./texture_mesh_ct_vcg.ply")
    make_MAPS_shape('./texture_mesh_3d_vcg.ply', './texture_mesh_3d_vcg_MAPS.obj', 192, 4)
    make_MAPS_shape('./texture_mesh_ct_vcg.ply', './texture_mesh_ct_vcg_MAPS.obj', 192, 4)
    segment_colors = np.array([
        [0, 114, 189],
        [238, 177, 32],
        [117, 142, 48],
        [76, 190, 238]
    ])
    meshs1,feats,fs=load_mesh(path="./texture_mesh_3d_vcg_MAPS.obj",normalize=True,request=['area', 'face_angles', 'curvs', 'center', 'normal'])
    meshs=np.array(meshs1).reshape(1,-1,3)
    feats=np.array(feats.reshape(1,13,-1))
    fs=np.int32(fs)
    net = MeshVanillaUnet(13, 4, 'bilinear')
    net.load("./checkpoints/qz/qz_best.pkl")
    net.eval()
    with jt.no_grad():
        mesh_tensor = to_mesh_tensor(meshs,feats,fs)
        outputs = net(mesh_tensor)
        preds = np.argmax(outputs.data, axis=1)
    
    mesh = trimesh.load_mesh("./texture_mesh_3d_vcg_MAPS.obj", process=False)
    for i in range(preds.shape[0]):
        mesh.visual.face_colors[:, :3] = segment_colors[preds[i, :mesh.faces.shape[0]]]
        mesh.export("./texture_mesh_3d_vcg_MAPS_seg.ply")

    meshs2,feats1,fs1=load_mesh(path="./texture_mesh_ct_vcg_MAPS.obj",normalize=True,request=['area', 'face_angles', 'curvs', 'center', 'normal'])
    meshs1=np.array(meshs2).reshape(1,-1,3)
    feats1=np.array(feats1.reshape(1,13,-1))
    fs1=np.int32(fs1)
    net1 = MeshVanillaUnet(13, 4, 'bilinear')
    net1.load("./checkpoints/ct/ct_best.pkl")
    net1.eval()
    with jt.no_grad():
        mesh_tensor1 = to_mesh_tensor(meshs1,feats1,fs1)
        outputs1 = net(mesh_tensor1)
        preds1 = np.argmax(outputs1.data, axis=1)
    
    mesh3 = trimesh.load_mesh("./texture_mesh_ct_vcg_MAPS.obj", process=False)
    for i in range(preds1.shape[0]):
        mesh3.visual.face_colors[:, :3] = segment_colors[preds1[i, :mesh3.faces.shape[0]]]
        mesh3.export("./texture_mesh_ct_vcg_MAPS_seg.ply")
    
    # mesh = trimesh.load_mesh("./1_MAPS_label.obj", process=False)
    # mesh1 = trimesh.load_mesh("./ct_maps_label.obj", process=False)
    mesh = trimesh.load_mesh("./texture_mesh_3d_vcg_MAPS_seg.ply", process=False)
    mesh1 = trimesh.load_mesh("./texture_mesh_ct_vcg_MAPS_seg.ply", process=False)
    f =np.asarray(mesh.faces)
    c=np.asarray(mesh.visual.face_colors[:, :3])
    v =np.asarray(mesh.vertices)
    h=f.shape[0]
    vh=v.shape[0]
    c1=c.shape[0]
    
    f1 =np.asarray(mesh1.faces)
    c2=np.asarray(mesh1.visual.face_colors[:, :3])
    v1=np.asarray(mesh1.vertices)
    h1=f1.shape[0]
    vh1=v1.shape[0]
    c11=c2.shape[0]

    data_index=np.array([h,h+vh,h+vh+c1])
    data_index1=np.array([h+vh+c1+h1,h+vh+c1+h1+vh1,h+vh+c1+h1+vh1+c11])
    print(h,h+vh,h+vh+c1)
    print(h+vh+c1+h1,h+vh+c1+h1+vh1,h+vh+c1+h1+vh1+c11)
    imgArray1=np.vstack((f,v))
    imgArray=np.vstack((imgArray1,c))
    imgArray2=np.vstack((imgArray,f1))
    imgArray3=np.vstack((imgArray2,v1))
    imgArray4=np.vstack((imgArray3,c2))
    imgArray5=np.vstack((imgArray4,data_index))
    imgArray6=np.vstack((imgArray5,data_index1))


    return imgArray6



def main():
    HOST = '172.21.141.27' # Standard loopback interface address (localhost)
    PORT = 6000        # Port to listen on (non-privileged ports are > 1023)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print("before bind")
        s.bind((HOST, PORT))
        print("before listen")
        s.listen()
        print("after listen")

        while True:
            image = waitForData(s)
            print("after waitForData")

            imageNew = process(image)
            #imageNew = image
            print("Image processed")

            sendProcessDataToClient(s, imageNew)

    


if __name__ == "__main__":
    # if len(sys.argv) < 4:
    #     print("Usage: hwz2 <input> <sigma> <output>")
    #     sys.exit(1)
    #sys.argv[1], float(sys.argv[2]), sys.argv[3]
    main()

# adapted from https://github.com/autonomousvision/monosdf/ (replica_eval/eval_recon.py and scannet_eval/evaluate.py)
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import trimesh
import os
import argparse

os.environ['PYOPENGL_PLATFORM'] = 'egl'

def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances, indices


def evaluate(mesh_pred, mesh_trgt, threshold=.05, down_sample=.02):
    pcd_trgt = o3d.geometry.PointCloud()
    pcd_pred = o3d.geometry.PointCloud()
    
    pcd_trgt.points = o3d.utility.Vector3dVector(mesh_trgt.vertices[:, :3])
    pcd_pred.points = o3d.utility.Vector3dVector(mesh_pred.vertices[:, :3])

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    dist1, _ = nn_correspondance(verts_pred, verts_trgt)
    dist2, _ = nn_correspondance(verts_trgt, verts_pred)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)

    # normal consistency
    N = 200000
    pointcloud_pred, idx = mesh_pred.sample(N, return_index=True)
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    normal_pred = mesh_pred.face_normals[idx]

    pointcloud_trgt, idx = mesh_trgt.sample(N, return_index=True)
    pointcloud_trgt = pointcloud_trgt.astype(np.float32)
    normal_trgt = mesh_trgt.face_normals[idx]

    _, index1 = nn_correspondance(pointcloud_pred, pointcloud_trgt)
    _, index2 = nn_correspondance(pointcloud_trgt, pointcloud_pred)

    normal_acc = np.abs((normal_pred * normal_trgt[index2.reshape(-1)]).sum(axis=-1)).mean()
    normal_comp = np.abs((normal_trgt * normal_pred[index1.reshape(-1)]).sum(axis=-1)).mean()
    normal_consistency = (normal_acc + normal_comp) * 0.5

    # all metrics * 100, 
    # cd use cm as unit, 
    # F-score and normal consistency are in percentage
    metrics = {
        'Acc': np.mean(dist2) * 100,
        'Comp': np.mean(dist1) * 100,
        'Chamfer-L1': ((np.mean(dist2) + np.mean(dist1)) / 2) * 100,
        'Prec': precision * 100,
        'Recal': recal * 100,
        'F-score': fscore * 100,
        'Normal-Acc': normal_acc * 100,
        'Normal-Comp': normal_comp * 100,
        'Normal-Consistency': normal_consistency * 100,
    }
    return metrics

def eval_mesh(input_mesh_path, gt_mesh_path):
    mesh_pred = trimesh.load(input_mesh_path)
    mesh_trg = trimesh.load(gt_mesh_path)

    metrics = evaluate(mesh_pred, mesh_trg)

    return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the scene.'
    )

    parser.add_argument('--input_mesh', type=str,  help='path to the mesh to be culled')
    parser.add_argument('--gt_mesh', type=str,  help='path to the gt mesh')
    parser.add_argument('--output_txt', type=str,  help='path to the output metrics txt')
    args = parser.parse_args()

    metrics = eval_mesh(args.input_mesh, args.gt_mesh)

    for k, v in metrics.items():
        print(f"{k}: {v}")

    # save metric to txt file
    with open(args.output_txt, 'a') as f:
        for k, v in metrics.items():
            out = f"{k}: {v}\n"
            f.write(out)

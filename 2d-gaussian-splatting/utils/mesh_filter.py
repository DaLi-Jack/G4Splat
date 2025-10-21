import trimesh
import numpy as np
import time
import argparse

def filter_mesh(mesh, length_threshold=0.05):

    # Get all faces (triangles)
    faces = mesh.faces
    vertices = mesh.vertices

    # Calculate the length of three edges for each face
    face_vertices = vertices[faces]  # (N, 3, 3)
    edge_lengths = np.stack([
        np.linalg.norm(face_vertices[:, 0] - face_vertices[:, 1], axis=1),
        np.linalg.norm(face_vertices[:, 1] - face_vertices[:, 2], axis=1),
        np.linalg.norm(face_vertices[:, 2] - face_vertices[:, 0], axis=1)
    ], axis=1)  # (N, 3)
    print("edge_lengths computed!")

    # Find all faces where all three edge lengths are <= threshold
    valid_faces_mask = np.all(edge_lengths <= length_threshold, axis=1)
    valid_faces = faces[valid_faces_mask]
    print("valid_faces computed!")

    # Create new mesh (vertices may still contain isolated points)
    filtered_mesh = trimesh.Trimesh(vertices=vertices, faces=valid_faces, process=False)        # process=False to avoid automatic topology processing

    # Remove vertices that are not used by any face (isolated points)
    filtered_mesh.remove_unreferenced_vertices()

    return filtered_mesh


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_path', type=str, required=True)
    parser.add_argument('--length_threshold', type=float, default=0.5)              # Threshold: maximum allowed edge length
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    # Load mesh
    mesh = trimesh.load(args.mesh_path)

    t1 = time.time()

    filtered_mesh = filter_mesh(mesh, args.length_threshold)
    filtered_mesh.export(args.output_path)

    t2 = time.time()
    print(f"[INFO] Total time: {t2 - t1:.2f} seconds")

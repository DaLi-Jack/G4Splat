import copy
import cv2
import numpy as np
import torch


def to_world_space(normals, c2w):
    """
    Transform the normals from camera space to world space.
    args:
        normals: Nx3
        c2w: 3x4, camera to world
    return:
        normals: Nx3
    """
    shape = normals.shape
    normals = normals.reshape(-1,3)
    extrinsics = copy.deepcopy(c2w)
    torch_flag = False
    if torch.is_tensor(extrinsics):
        extrinsics = extrinsics.cpu().numpy()
    if torch.is_tensor(normals):
        normals = normals.cpu().numpy()
        torch_flag = True

    assert extrinsics.shape[0] ==3
    normals = normals.transpose()
    extrinsics[:3, 3] = np.zeros(3)  # only rotation, no translation

    normals_world = np.matmul(extrinsics,
                            np.vstack((normals, np.ones((1, normals.shape[1])))))[:3]
    normals_world = normals_world.transpose((1, 0))

    if torch_flag:
        return torch.from_numpy(normals_world).reshape(shape)

    return normals_world.reshape(shape)

def remove_small_isolated_areas(img, min_size = 3000):
    f'''Remove the small isolated areas with size smaller than defined {min_size}
    '''
    if img.ndim ==3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = copy.deepcopy(img).astype(np.uint8)
    img = cv2.medianBlur(gray, 3)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    img_clean = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img_clean[output == i + 1] = 255

    return img_clean

def separate_isolated_components(img, min_size = 3000):
    if img.ndim ==3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = copy.deepcopy(img).astype(np.uint8)
    img = cv2.medianBlur(gray, 3)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    mask_list = []
    centroids_list = []
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            mask = np.zeros((output.shape))
            mask = output == i + 1
            mask_list.append(mask)
            centroids_list.append(np.array([centroids[i+1][1], centroids[i+1][0]]))
    return mask_list, centroids_list

def merge_normal_clusters(pred, sorted_topk, centers):
    """
    Merge the normal clusters based on the distance between the centers of the clusters.
    """
    new_pred = copy.deepcopy(pred)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    num_clusters = len(sorted_topk)
    flag = np.zeros(num_clusters)
    new_num_clusters = num_clusters

    for i in range(num_clusters):
        if flag[i] == 1:
            continue
        for j in range(i + 1, num_clusters):
            if flag[j] == 1:
                continue

            if np.dot(centers[sorted_topk[i]], centers[sorted_topk[j]]).sum() > 0.95:
                new_pred[pred == sorted_topk[j]] = sorted_topk[i]
                new_num_clusters -= 1
                flag[j] = 1

    if new_num_clusters != num_clusters:
        count_values = np.bincount(new_pred)
        topk = np.argpartition(count_values,-new_num_clusters)[-new_num_clusters:]
        sorted_topk_idx = np.argsort(count_values[topk])
        sorted_topk = topk[sorted_topk_idx][::-1]

    return new_pred, sorted_topk, new_num_clusters
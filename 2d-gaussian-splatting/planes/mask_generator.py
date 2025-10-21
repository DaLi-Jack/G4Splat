# Over-segmentation by selecting the smallest mask for each prompt. Adapted from SuperPrimitive at https://github.com/makezur/super_primitive

import torch

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.amg import calculate_stability_score, batched_mask_to_box, MaskData
from torchvision.ops.boxes import batched_nms, box_area

SAM_CONFIG = {
    "select_smallest": True,
    "nms": True,
    "box_nms_thresh": 0.8,
    "iou_threshold": 0.0,
    "stability_threshold": 0.85,
    "filter_edge_points": False,
    "filter_by_box_size": False,
}

def denormalise_coordinates(x_norm, dims):
    dims = torch.as_tensor(dims, dtype=torch.float32, device=x_norm.device)
    x_pixel = 0.5 * (dims - 1) * ((x_norm) + 1 )
    return x_pixel.round().long()

def normalise_coordinates(x_pixel, dims):
    inv = 1.0 / (torch.as_tensor(dims, dtype=torch.float32, device=x_pixel.device) - 1)

    x_norm = 2 * x_pixel * inv - 1
    return x_norm

def setup_sam(
    sam_checkpoint="./checkpoint/segment-anything/sam_vit_h_4b8939.pth", 
    model_type = "vit_h", device="cuda"
):

    print(f"loading SAM/'{model_type}' checkpoint from: {sam_checkpoint}")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    return predictor

def infer_sam_masks_batch(model_sam, image, points, logits=True):
    # points is same form as keypoints, [N, 2]
    # in the range [-1, 1]
    device = points.device
    num_pts = points.shape[0]

    model_sam.set_image(image)

    H, W, _ = image.shape

    H_sam, W_sam = model_sam.transform.get_preprocess_shape(H, W, model_sam.transform.target_length)

    points_sam_format = denormalise_coordinates(points, (H_sam, W_sam))
    points_sam_format = points_sam_format.flip(-1)

    # sam ids for postive / negative ponits. here we use one postive per segment
    dummy_ids = torch.ones((num_pts, 1), dtype=torch.int64, device=device)

    masks, iou_predictions, lowres =  model_sam.predict_torch(points_sam_format[:, None], dummy_ids, multimask_output=True, return_logits=logits)

    return {'masks': masks,
            'iou_pred': iou_predictions,
            'lowres': lowres}

def smallest_good_mask_batch(masks, iou_pred, iou_threshold=0.88, stability_score_thresh=0.95, select_smallest=True):
    # masks is [N, 3, H, W]
    # iou_predictions is [N, 3]

    def filter_per_keypoint(candidate_scores):
        if select_smallest:
            return (candidate_scores).sum(dim=1, dtype=torch.int16) > 0
        else:
            return candidate_scores

    stability_score_offset = 1.0
    model_mask_thresh = 0.0

    if select_smallest:
        data = MaskData(masks=masks, iou_preds=iou_pred,
                        keypoints_ids=torch.arange(masks.shape[0], device=masks.device),
                        good_masks=torch.ones((masks.shape[0], 3), dtype=torch.bool, device=masks.device))
    else:
        data = MaskData(masks=masks.flatten(0, 1),
                        iou_preds=iou_pred.flatten(0, 1),
                        keypoints_ids=torch.arange(masks.shape[0], device=masks.device).repeat_interleave(3))
        del masks
    # threshold for per-pixel mask logits

    if iou_threshold > 0:
        good_masks = data['iou_preds'] > iou_threshold
        # same trick as in the SAM code to prevent casting to int64
        good_keypoints = filter_per_keypoint(good_masks)

        # filtered
        if 'good_masks' in  data._stats.keys():
            data['good_masks'] = torch.logical_and(data['good_masks'], good_masks)

        data.filter(good_keypoints)

    if stability_score_thresh > 0:
        # here asssume that masks haven't been sigmoided yet
        mask_stability_scores = calculate_stability_score(data['masks'],
                                                          model_mask_thresh,
                                                          stability_score_offset)
        good_masks = mask_stability_scores >= stability_score_thresh
        good_keypoints = filter_per_keypoint(good_masks)

        # filtered
        if 'good_masks' in data._stats.keys():
            data['good_masks'] = torch.logical_and(data['good_masks'], good_masks)

        data.filter(good_keypoints)

    # binariase masks
    data['masks'] = data['masks'] > model_mask_thresh

    # same trick to save up memory
    if select_smallest:
        masks_sizes = data['masks'].sum(dim=-1, dtype=torch.int16).sum(dim=-1, dtype=torch.int32)
        # now we want to to select the smallest good mask per keypoint
        masks_sizes[~data['good_masks']] = 1e6

        smallest_mask_id = masks_sizes.argmin(dim=1)
        smallest_mask = data['masks'][torch.arange(data['masks'].shape[0]),
                                    smallest_mask_id, ...]
        smallest_ious = data['iou_preds'][torch.arange(data['masks'].shape[0]),
                                        smallest_mask_id, ...]



        result = {'masks': smallest_mask,
                  'iou_preds': smallest_ious,
                  'keypoints_ids': data['keypoints_ids'],
                  'masks_ids': smallest_mask_id}
    else:
        result = {'masks': data['masks'],
                  'iou_preds': data['iou_preds'],
                  'keypoints_ids': data['keypoints_ids']}

    del data
    result['boxes'] = batched_mask_to_box(result['masks'])

    return result


def active_sample_pos(coverage_mask, num_samples=100, fine_noise=True):
    # mask coverage is [B, H, W]
    B, H, W = coverage_mask.shape
    with torch.no_grad():
        downsample_factor = 16
        coverage_mask = coverage_mask.clone()
        # replace lower few rows with 1 to compensate for SAM artifacts
        coverage_mask[:, -2:, :] = 1
        coarse_coverage = torch.nn.functional.avg_pool2d(coverage_mask.float()[:, None],
                                                        downsample_factor,
                                                        stride=downsample_factor)
    H_coarse, W_coarse = coarse_coverage.shape[2:]
    sample_density = 1.0 - coarse_coverage
    sample_density = sample_density / (sample_density.sum(dim=(2, 3), keepdim=True) + 1e-6)

    distribution = torch.distributions.Categorical(probs=sample_density.view(B, -1))

    sample_indices = distribution.sample((num_samples,)).view(num_samples, B)
    # convert flattened C-style index onto 2D grid index
    sample_indices = torch.stack([sample_indices // W_coarse, sample_indices % W_coarse], dim=2)
    sample_indices = sample_indices.permute(1, 0, 2)

    # now generate samples in the fine grid
    sample_indices = sample_indices.reshape(B, num_samples, 2)
    coarse_indices = sample_indices
    normalised_coarse = normalise_coordinates(sample_indices, (H_coarse, W_coarse))
    # assume ratio H / coarse_H = W / coarse_W
    if fine_noise:
        per_quadrant_noise = normalise_coordinates((torch.randint_like(normalised_coarse,
                                                                                #    low=-downsample_factor // 2 + 1,
                                                                                   high=downsample_factor // 2, device=coverage_mask.device)), (H, W)) + 1
        normalised_coarse = normalised_coarse + per_quadrant_noise
        normalised_coarse = normalised_coarse.clamp(-1, 1)
    sample_indices = denormalise_coordinates(normalised_coarse, (H, W))
    sample_indices = sample_indices.reshape(B, num_samples, 2)
    normalised_coarse = normalised_coarse.reshape(B, num_samples, 2)

    result = {'coarse_density': sample_density,
              'coarse_indices': coarse_indices,
              'sample_indices': sample_indices,
              'normalised_coords': normalised_coarse}
    return result


def infer_masks(sam_model, image,
                keypoints=None,
                num_pts=300, num_pts_active=50,
                device=torch.device('cuda:0')):
    H, W, _ = image.shape

    if keypoints is None:
        keypoints = (torch.rand(num_pts, 2, device=device) * 2 - 1)

    select_smallest = SAM_CONFIG['select_smallest']
    nms = SAM_CONFIG['nms']
    box_nms_thresh = SAM_CONFIG['box_nms_thresh']
    iou_threshold = SAM_CONFIG['iou_threshold']
    stability_score_thresh = SAM_CONFIG['stability_threshold']
    filter_edge_keypoints = SAM_CONFIG['filter_edge_points']
    filter_by_box_size = SAM_CONFIG['filter_by_box_size']


    masks = infer_sam_masks_batch(sam_model, image, keypoints)
    masks = smallest_good_mask_batch(masks['masks'], masks['iou_pred'], iou_threshold=iou_threshold, stability_score_thresh=stability_score_thresh, select_smallest=select_smallest)
    masks = MaskData(**masks)

    keypoints_filtered = keypoints[masks['keypoints_ids'], ...]

    if nms:
        # prefer smaller boxes
        scores_boxes = 1 / box_area(masks["boxes"])

        keep_by_nms = batched_nms(
            masks['boxes'].float(),
            scores_boxes if filter_by_box_size else masks['iou_preds'],
            torch.zeros_like(masks["boxes"][:, 0]),  # categories
            iou_threshold=box_nms_thresh,
        )
        masks.filter(keep_by_nms)
        keypoints_filtered = keypoints_filtered[keep_by_nms, ...]

    coverage_mask = masks['masks'].any(dim=0)
    sampled_masks = None

    if num_pts_active > 0:
        # coverage_mask = masks['masks'].any(dim=0)
        sampled_masks  = active_sample_pos(coverage_mask[None], num_samples=num_pts_active)
        keypoints_active = sampled_masks['normalised_coords'][0]

        src_masks_add =  infer_sam_masks_batch(sam_model, image, keypoints_active)
        src_masks_add = smallest_good_mask_batch(src_masks_add['masks'], src_masks_add['iou_pred'], iou_threshold=iou_threshold, stability_score_thresh=stability_score_thresh, select_smallest=select_smallest)
        keypoints_active_filtered = keypoints_active[src_masks_add['keypoints_ids'], ...]
        num_added = keypoints_active_filtered.shape[0]

        src_masks_add = MaskData(**src_masks_add)

        if nms:
            add_scores_boxes = 1 / box_area(src_masks_add["boxes"])

            keep_by_nms = batched_nms(
                src_masks_add['boxes'].float(),
                add_scores_boxes if filter_by_box_size else src_masks_add['iou_preds'],
                torch.zeros_like(src_masks_add ["boxes"][:, 0]),  # categories
                iou_threshold=box_nms_thresh,
            )
            src_masks_add.filter(keep_by_nms)
            keypoints_active_filtered = keypoints_active_filtered[keep_by_nms, ...]
            num_added = keypoints_active_filtered.shape[0]

        keypoints_final = torch.cat([keypoints_filtered, keypoints_active_filtered], dim=0)
        masks.cat(src_masks_add)

    else:
        num_added = 0
        keypoints_final = keypoints_filtered

    if masks['masks'].shape[0] == 0:
        return {'masks': {'masks': torch.tensor([])}}


    if filter_edge_keypoints:
        # masks is of shape (N, H, W)
        # keypoints_final is of shape (N, 2)

        keypoints_denorm = denormalise_coordinates(keypoints_final, (H, W))
        keypoints_denorm = keypoints_denorm.long()
        mask_value_at_keypoints = masks['masks'][torch.arange(masks['masks'].shape[0], device=masks['masks'].device),
                                                 keypoints_denorm[:, 0],
                                                 keypoints_denorm[:, 1]]

        masks.filter(mask_value_at_keypoints)
        keypoints_final = keypoints_final[mask_value_at_keypoints, ...]


    final_coverage = masks['masks'].any(dim=0)

    result = {'masks': masks._stats,
              'keypoints': keypoints_final,
              'num_active': num_added,
              'coarse_coverage': coverage_mask,
              'final_coverage': final_coverage,
              'sampled_masks': sampled_masks,
              }

    return result
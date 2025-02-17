import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
import tyro
from typing import Optional, Tuple, List
import numpy.typing as npt

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


def show_mask(mask: npt.NDArray[np.uint8],
              ax: plt.Axes,
              obj_id: Optional[int] = None,
              borders: bool = True,
              random_color: bool = False,
              border_thickness: int = 2) -> None:
    """Unified mask visualization for both image and video cases"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        if obj_id is not None:
            cmap = plt.get_cmap("tab10")
            color = np.array([*cmap(obj_id)[:3], 0.6])
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    if borders:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=border_thickness)

    ax.imshow(mask_image)


def show_points(coords: npt.NDArray[np.float32],
                labels: npt.NDArray[np.int32],
                ax: plt.Axes,
                marker_size: Optional[int] = None) -> None:
    """Unified point visualization for both image and video cases"""
    if marker_size is None:
        marker_size = 375 if len(coords) < 10 else 200  # Larger markers for fewer points

    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0],
               pos_points[:, 1],
               color='green',
               marker='*',
               s=marker_size,
               edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0],
               neg_points[:, 1],
               color='red',
               marker='*',
               s=marker_size,
               edgecolor='white',
               linewidth=1.25)


def show_box(box: npt.NDArray[np.float32], ax: plt.Axes) -> None:
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_masks(image: npt.NDArray[np.uint8],
               masks: List[npt.NDArray[np.bool_]],
               scores: Optional[List[float]] = None,
               point_coords: Optional[npt.NDArray[np.float32]] = None,
               box_coords: Optional[npt.NDArray[np.float32]] = None,
               input_labels: Optional[npt.NDArray[np.int32]] = None,
               borders: bool = True,
               border_thickness: int = 2,
               figure_size: Tuple[int, int] = (10, 10),
               hide_prompt: bool = False,
               output_dir: Optional[Path] = None,
               base_filename: Optional[str] = None,
               transparent_mask: bool = True) -> None:
    for i, mask in enumerate(masks):
        plt.figure(figsize=figure_size)
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders, border_thickness=border_thickness)
        if point_coords is not None and not hide_prompt:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None and not hide_prompt:
            show_box(box_coords, plt.gca())
        if scores is None:
            plt.title(f"Mask {i+1}", fontsize=18)
        else:
            plt.title(f"Mask {i+1}, Score: {scores[i]:.3f}", fontsize=18)
        plt.axis('off')

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the mask
            if base_filename:
                mask_name = base_filename
            else:
                mask_name = "mask"
            if len(masks) > 1:
                mask_name += f"_{i+1}"
                
            if transparent_mask:
                # Save as RGBA PNG with transparent background
                mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
                mask_rgba[..., 3] = mask.astype(np.uint8) * 255  # Alpha channel
                mask_rgba[mask > 0, :3] = image[mask > 0]  # RGB channels
                Image.fromarray(mask_rgba).save(output_dir / f"{mask_name}.png")
            else:
                # Save as binary black and white image
                Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / f"{mask_name}.png")
        else:
            plt.show()
        plt.close()


@dataclass
class CLIArgs:
    """Arguments for the SAM2 CLI"""
    data: Path  # Path to an image file or directory of frames
    point: Optional[Tuple[float, float]] = None  # Optional point coordinates (x, y) for mask prediction
    device: str = "cuda"  # Device to run inference on
    checkpoint: Path = Path("./checkpoints/sam2.1_hiera_large.pt")  # Path to model checkpoint
    config: Path = Path("configs/sam2.1/sam2.1_hiera_l.yaml")  # Path to model config
    borders: bool = True  # Whether to show borders around masks
    stride: int = 1  # Stride for visualizing/saving frames in video mode
    marker_size: Optional[int] = None  # Size of point markers (None for automatic)
    random_colors: bool = False  # Use random colors for mask visualization
    multimask: bool = True  # Output multiple masks per prompt
    score_threshold: float = 0.0  # Minimum score threshold for showing masks
    output_dir: Optional[Path] = None  # Directory to save visualization results
    border_thickness: int = 2  # Thickness of mask borders when shown
    hide_prompt: bool = False  # Hide prompt points in visualization
    figure_size: Tuple[int, int] = (10, 10)  # Size of output figures (width, height)
    mask_threshold: float = 0.0  # Threshold for converting soft masks to binary
    batch_size: int = 1  # Batch size for video processing
    transparent_mask: bool = True  # Save masks as PNG with transparency instead of black and white


if __name__ == "__main__":
    args = tyro.cli(CLIArgs)

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.data.is_file():
        image = Image.open(args.data)
        image = np.array(image.convert("RGB"))
        base_filename = args.data.stem

        sam2_model = build_sam2(str(args.config), str(args.checkpoint), device=args.device)
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.set_image(image)

        points = np.array([[image.shape[1] // 2, image.shape[0] // 2],
                           [int(image.shape[1] * 0.1), int(image.shape[0] * 0.1)],
                           [int(image.shape[1] * 0.9), int(image.shape[0] * 0.1)],
                           [int(image.shape[1] * 0.1), int(image.shape[0] * 0.9)],
                           [int(image.shape[1] * 0.9), int(image.shape[0] * 0.9)]])
        labels = np.array([1, 0, 0, 0, 0])
        if args.point:
            points = np.array([args.point], dtype=np.float32)
            if args.point[0] < 1:
                points[:, 0] = int(image.shape[1] * points[0, 0])
            if args.point[1] < 1:
                points[:, 1] = int(image.shape[0] * points[0, 1])
            labels = np.array([1], dtype=np.int32)

        with torch.inference_mode(), torch.autocast(device_type=args.device, dtype=torch.bfloat16):
            masks, scores, logits = predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=args.multimask,
            )

        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        # Filter by score and mask thresholds
        valid_mask = scores >= args.score_threshold
        masks = masks[valid_mask]
        scores = scores[valid_mask]

        if len(scores) > 0:
            best = np.argmax(scores)
            if args.mask_threshold > 0:
                masks = (masks > args.mask_threshold).astype(np.uint8)

            show_masks(
                image=image,
                masks=[masks[best]],
                scores=[scores[best]],
                point_coords=points if not args.hide_prompt else None,
                input_labels=labels if not args.hide_prompt else None,
                borders=args.borders,
                border_thickness=args.border_thickness,
                figure_size=args.figure_size,
                hide_prompt=args.hide_prompt,
                output_dir=args.output_dir,
                base_filename=base_filename,
                transparent_mask=args.transparent_mask,
            )
        else:
            print("No masks found above the score threshold.")

    elif args.data.is_dir():
        frame_names = [p for p in args.data.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        frame_names.sort(key=lambda p: int(''.join(filter(str.isdigit, p.stem))))

        # Process frames in batches for video
        image = np.asarray(Image.open(frame_names[0]))
        predictor = build_sam2_video_predictor(str(args.config), str(args.checkpoint), device=args.device)
        inference_state = predictor.init_state(video_path=str(args.data), async_loading_frames=True)

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        points = np.array([[image.shape[1] // 2, image.shape[0] // 2]], dtype=np.float32)
        if args.point:
            points = np.array([args.point], dtype=np.float32)
            if args.point[0] < 1:
                points[:, 0] = int(image.shape[1] * points[0, 0])
            if args.point[1] < 1:
                points[:, 1] = int(image.shape[0] * points[0, 1])
        labels = np.array([1], dtype=np.int32)

        with torch.inference_mode(), torch.autocast(device_type=args.device, dtype=torch.bfloat16):
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

        show_masks(
            image=image,
            masks=[(m > 0).squeeze().cpu().numpy() for m in masks],
            point_coords=points,
            input_labels=labels,
            borders=args.borders,
            border_thickness=args.border_thickness,
            figure_size=args.figure_size,
            hide_prompt=args.hide_prompt,
        )

        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        with torch.inference_mode(), torch.autocast(device_type=args.device, dtype=torch.bfloat16):
            for frame_idx, object_ids, masks in predictor.propagate_in_video(inference_state):
                video_segments[frame_idx] = {
                    object_id: (masks[i] > 0).squeeze().cpu().numpy() for i, object_id in enumerate(object_ids)
                }

        # render the segmentation results every few frames
        for frame_idx in range(0, len(frame_names), args.stride):
            show_masks(
                image=np.asarray(Image.open(frame_names[frame_idx])),
                masks=[video_segments[frame_idx][out_obj_id] for out_obj_id in video_segments[frame_idx]],
                borders=args.borders,
                border_thickness=args.border_thickness,
                figure_size=args.figure_size,
                hide_prompt=args.hide_prompt,
                output_dir=args.output_dir,
                base_filename=frame_names[frame_idx].stem,
                transparent_mask=args.transparent_mask,
            )

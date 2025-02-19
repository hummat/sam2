import contextlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import tyro
from loguru import logger
from PIL import Image
from tqdm import trange

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor


def show_mask(mask: npt.NDArray[np.uint8],
              ax: plt.Axes,
              obj_id: Optional[int] = None,
              borders: bool = True,
              random_color: bool = False,
              border_thickness: int = 2) -> None:
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
                marker_size: int = 100) -> None:
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0],
               pos_points[:, 1],
               color='green',
               marker='.',
               s=marker_size,
               edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0],
               neg_points[:, 1],
               color='red',
               marker='.',
               s=marker_size,
               edgecolor='white',
               linewidth=1.25)


def show_box(box: npt.NDArray[np.float32], ax: plt.Axes) -> None:
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_masks(image: npt.NDArray[np.uint8],
               masks: List[npt.NDArray[np.bool_] | None],
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
               transparent_mask: bool = True) -> Optional[Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]]:
    for i, mask in enumerate(masks):
        fig = plt.figure(figsize=figure_size)
        plt.imshow(image)
        if mask is not None:
            show_mask(mask, plt.gca(), borders=borders, border_thickness=border_thickness)
        if point_coords is not None and not hide_prompt:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None and not hide_prompt:
            show_box(box_coords, plt.gca())

        title = "image" if mask is None else "mask"
        if base_filename:
            title = f"{base_filename} mask"
        if len(masks) > 1:
            title += f" {i+1}"
        if scores is not None:
            title += f", Score: {scores[i]:.3f}"
        plt.title(title, fontsize=18)
        plt.axis('off')
        plt.tight_layout()

        if mask is not None and output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            if base_filename:
                mask_name = base_filename
            else:
                mask_name = "mask"
            if len(masks) > 1:
                mask_name += f"_{i+1}"

            if transparent_mask:
                mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
                mask_rgba[..., 3] = mask.astype(np.uint8) * 255  # Alpha channel
                mask_rgba[mask > 0, :3] = image[mask > 0]  # RGB channels
                Image.fromarray(mask_rgba).save(output_dir / f"{mask_name}.png")
            else:
                Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / f"{mask_name}.png")
        else:

            # Lists to store click points and their labels
            click_points = []
            click_labels = []

            def on_click(event):
                if event.inaxes:
                    if event.button == 1:  # left click
                        label = 1
                    elif event.button == 3:  # right click
                        label = 0
                    else:
                        return  # Ignore other mouse buttons

                    coords = (event.xdata, event.ydata)
                    click_points.append(coords)
                    click_labels.append(label)
                    logger.debug(f'Clicked at data coordinates: {coords} with label {label}')

                    show_points(
                        np.array(click_points, np.float32),
                        np.array(click_labels, np.int32),
                        plt.gca(),  # adjust marker size if needed
                    )
                    fig.canvas.draw_idle()

            # Connect the callback function to the 'button_press_event'
            fig.canvas.mpl_connect('button_press_event', on_click)

            plt.show()  # Wait until the plot window is closed
            plt.close()

            if click_points:
                return np.array(click_points, np.float32), np.array(click_labels, np.int32)
        plt.close()


@dataclass
class CLIArgs:
    """Arguments for the SAM-2 CLI"""
    data: Path  # Path to the image(s)
    point: Optional[Tuple[float, float]] = None  # Optional point coordinates (x, y) for mask prediction
    init_frame: int = 0  # Frame index to start propagation
    checkpoint: str = "sam2.1_hiera_large.pt"  # Name of the SAM2 checkpoint
    huggingface: bool = False  # Load model from Hugging Face model hub
    stride: int = 1  # Stride for visualizing/saving frames in video mode
    mask_threshold: float = 0  # Threshold for binarizing mask predictions
    max_hole_area: float = 8  # Maximum hole area in mask predictions
    max_sprinkle_area: float = 1  # Maximum sprinkle area in mask predictions
    output_dir: Optional[Path] = None  # Directory to save visualization results
    transparent_mask: bool = True  # Save masks as PNG with transparency instead of black and white
    async_load: bool = True  # Asynchronously load video frames
    progress: bool = False  # Show progress bar
    verbose: bool = False  # Enable verbose logging
    quiet: bool = False  # Suppress all logging except errors
    device: str = "auto"  # Device to run inference on


def run(args: CLIArgs):
    logger.remove()
    if not args.quiet:
        if args.verbose:
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.add(sys.stderr, level="INFO")
    else:
        logger.add(sys.stderr, level="ERROR")

    logger.info("Starting SAM-2 inference")
    logger.debug(f"Arguments: {args}")

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.debug("Enabled TF32 for CUDA computation")

    ckpt_parts = args.checkpoint.split('_')
    config_file = f"configs/sam2.1/{'_'.join(ckpt_parts[:-1] + [ckpt_parts[-1][0]])}.yaml"
    ckpt_path = str(Path(__file__).parent.parent / "checkpoints" / args.checkpoint)
    logger.debug(f"Using config file: {config_file}, Checkpoint: {ckpt_path}")

    device = args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using device: {device}")

    if args.data.is_file():
        logger.info(f"Processing single image: {args.data}")
        image = Image.open(args.data)
        image = np.array(image.convert("RGB"))
        base_filename = args.data.stem

        figure_size = (8, 4.5) if image.shape[1] > image.shape[0] else (4.5, 8)
        logger.debug(f"Using figure size: {figure_size}")

        if args.huggingface:
            hf_name = str(f"facebook/{Path(args.checkpoint).stem}").replace("_", "-")
            logger.info(f"Loading checkpoint {hf_name} from Hugging Face model hub")
            with (contextlib.nullcontext() if args.quiet else contextlib.redirect_stderr(io.StringIO())):
                predictor = SAM2ImagePredictor.from_pretrained(
                    model_id=hf_name,
                    mask_threshold=args.mask_threshold,
                    max_hole_area=args.max_hole_area,
                    max_sprinkle_area=args.max_sprinkle_area,
                )
        else:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            predictor = SAM2ImagePredictor(
                sam_model=build_sam2(config_file, ckpt_path, device=device),
                mask_threshold=args.mask_threshold,
                max_hole_area=args.max_hole_area,
                max_sprinkle_area=args.max_sprinkle_area,
            )
        predictor.set_image(image)
        logger.debug(f"Image shape: {image.shape}")

        points = None
        labels = None
        if args.point:
            points = np.array([args.point], dtype=np.float32)
            if args.point[0] < 1:
                points[:, 0] = image.shape[1] * points[0, 0]
            if args.point[1] < 1:
                points[:, 1] = image.shape[0] * points[0, 1]
            labels = np.array([1], dtype=np.int32)
            logger.info(f"Using initial point {points[0]} with label {labels[0]}")

        masks = None
        all_points = None
        all_labels = None
        while True:
            if points is not None:
                with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logger.debug(f"Running inference with points: {points} and labels: {labels}")
                    masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels)

                    sorted_ind = np.argsort(scores)[::-1]
                    masks = masks[sorted_ind]
                    scores = scores[sorted_ind]
                    logger.debug(f"Generated {len(scores)} masks with scores: {scores}")
                    best = np.argmax(scores)
                    logger.debug(f"Best mask score: {scores[best]:.3f}")

            data = show_masks(
                image=image,
                masks=[None] if masks is None else [masks[best]],
                scores=None if masks is None else [scores[best]],
                point_coords=points if all_points is None else all_points,
                input_labels=labels if all_labels is None else all_labels,
                figure_size=figure_size,
                output_dir=args.output_dir,
                base_filename=base_filename,
                transparent_mask=args.transparent_mask,
            )

            if data is None:
                break

            points, labels = data
            if all_points is None:
                all_points = points
                all_labels = labels
            else:
                all_points = np.concatenate([all_points, points], axis=0)
                all_labels = np.concatenate([all_labels, labels], axis=0)

    elif args.data.is_dir():
        frame_names = [p for p in args.data.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        frame_names.sort(key=lambda p: int(''.join(filter(str.isdigit, p.stem))))
        logger.info(f"Processing video directory with {len(frame_names)} frames")

        image = np.asarray(Image.open(frame_names[0]))
        figure_size = (8, 4.5) if image.shape[1] > image.shape[0] else (4.5, 8)
        logger.debug(f"Using figure size: {figure_size}")

        if args.huggingface:
            hf_name = str(f"facebook/{Path(args.checkpoint).stem}").replace("_", "-")
            logger.info(f"Loading checkpoint {hf_name} from Hugging Face model hub")
            with (contextlib.nullcontext() if args.quiet else contextlib.redirect_stderr(io.StringIO())):
                predictor = SAM2VideoPredictor.from_pretrained(
                    model_id=hf_name,
                    apply_postprocessing=True,
                )
        else:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            predictor = build_sam2_video_predictor(
                config_file=config_file,
                ckpt_path=ckpt_path,
                device=device,
                apply_postprocessing=True,
            )

        logger.info("Loading frames")
        with (contextlib.nullcontext() if args.progress else contextlib.redirect_stderr(io.StringIO())):
            inference_state = predictor.init_state(video_path=str(args.data), async_loading_frames=args.async_load)

        points = None
        labels = None
        if args.point:
            points = np.array([args.point], dtype=np.float32)
            if args.point[0] < 1:
                points[:, 0] = image.shape[1] * points[0, 0]
            if args.point[1] < 1:
                points[:, 1] = image.shape[0] * points[0, 1]
            labels = np.array([1], dtype=np.int32)

            if not (args.progress and args.async_load):
                logger.info(f"Adding initial point: {points[0]} with label {labels[0]}")

        masks = None
        all_points = None
        all_labels = None
        while True:
            if points is not None:
                with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logger.debug(f"Adding new points: {points} with labels {labels}")
                    _, _, masks = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=args.init_frame,
                        obj_id=0,
                        points=points,
                        labels=labels,
                    )

                if args.point:
                    break

            data = show_masks(
                image=image,
                masks=[None] if masks is None else [(m > 0).squeeze().cpu().numpy() for m in masks],
                point_coords=points if all_points is None else all_points,
                input_labels=labels if all_labels is None else all_labels,
                figure_size=figure_size,
            )

            if data is None:
                break

            points, labels = data
            if all_points is None:
                all_points = points
                all_labels = labels
            else:
                all_points = np.concatenate([all_points, points], axis=0)
                all_labels = np.concatenate([all_labels, labels], axis=0)

        logger.info("Starting video propagation")
        video_segments = {}
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            with (contextlib.nullcontext() if args.progress else contextlib.redirect_stderr(io.StringIO())):
                for frame_idx, object_ids, masks in predictor.propagate_in_video(inference_state):
                    logger.debug(f"Processing frame {frame_idx} with {len(object_ids)} objects")
                    video_segments[frame_idx] = {
                        object_id: (masks[i] > 0).squeeze().cpu().numpy() for i, object_id in enumerate(object_ids)
                    }

        stride = args.stride if args.stride > 1 or args.output_dir else len(frame_names) // 10
        logger.info(f"Rendering results (stride={stride})" if stride > 1 else "Rendering reults")
        for frame_idx in trange(0, len(frame_names), stride, desc="render frames", disable=not args.progress):
            show_masks(
                image=np.asarray(Image.open(frame_names[frame_idx])),
                masks=[video_segments[frame_idx][out_obj_id] for out_obj_id in video_segments[frame_idx]],
                figure_size=figure_size,
                output_dir=args.output_dir,
                base_filename=frame_names[frame_idx].stem,
                transparent_mask=args.transparent_mask,
            )


def main():
    run(tyro.cli(CLIArgs))


if __name__ == "__main__":
    main()

import contextlib
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import tyro
from loguru import logger
from matplotlib.widgets import RectangleSelector
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
               linewidth=1)
    ax.scatter(neg_points[:, 0],
               neg_points[:, 1],
               color='red',
               marker='.',
               s=marker_size,
               edgecolor='white',
               linewidth=1)


def show_box(box: npt.NDArray[np.float32], ax: plt.Axes) -> None:
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_masks(image: npt.NDArray[np.uint8],
               masks: List[Optional[npt.NDArray[np.bool_]]],
               scores: Optional[List[float]] = None,
               point_coords: Optional[npt.NDArray[np.float32]] = None,
               box_coords: Optional[npt.NDArray[np.float32]] = None,
               input_labels: Optional[npt.NDArray[np.int32]] = None,
               borders: bool = True,
               border_thickness: int = 2,
               figsize: Tuple[int, int] = (10, 10),
               hide_prompt: bool = False,
               output_dir: Optional[Path] = None,
               base_filename: Optional[str] = None,
               transparent_mask: bool = True) -> Optional[Tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]]:
    for i, mask in enumerate(masks):
        fig = plt.figure(figsize=figsize)
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
            # Lists to store click points, labels and box coordinates
            click_points = []
            click_labels = []
            box_selection = []

            def on_box_select(eclick, erelease):
                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata
                box_selection.clear()
                box_selection.extend([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
                logger.debug(f'Selected box coordinates: {box_selection}')

                # Clear previous box if any
                plt.gca().clear()
                plt.imshow(image)
                if mask is not None:
                    show_mask(mask, plt.gca(), borders=borders, border_thickness=border_thickness)
                if click_points:
                    show_points(np.array(click_points, np.float32), np.array(click_labels, np.int32), plt.gca())
                show_box(np.array(box_selection, np.float32), plt.gca())
                fig.canvas.draw_idle()

            rect_selector = RectangleSelector(plt.gca(),
                                              on_box_select,
                                              useblit=True,
                                              button=[1],
                                              spancoords='data',
                                              interactive=True)

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

            if box_selection:
                return np.array(box_selection, np.float32), None

            if click_points:
                return np.array(click_points, np.float32), np.array(click_labels, np.int32)
        plt.close()


@dataclass
class SAM2Args:
    """Configuration for SAM-2 model and inference"""
    checkpoint: Literal["sam2.1_hiera_large.pt", "sam2.1_hiera_base_plus.pt", "sam2.1_hiera_small.pt",
                        "sam2.1_hiera_tiny.pt"] = "sam2.1_hiera_large.pt"
    """Name of the SAM-2 checkpoint"""
    huggingface: bool = False
    """Load model from Hugging Face model hub"""
    mask_threshold: float = 0
    """Threshold for binarizing mask predictions"""
    max_hole_area: float = 8
    """Maximum hole area in mask predictions"""
    max_sprinkle_area: float = 1
    """Maximum sprinkle area in mask predictions"""
    device: Literal["auto", "cpu", "gpu"] = "auto"
    """Device to run inference on"""


@dataclass
class Args:
    """Single object image and video frames segmentation using SAM-2"""
    data: Path
    """Path to the image(s) or video frames directory"""
    model: SAM2Args = field(default_factory=SAM2Args)
    """SAM-2 model configuration"""
    points: Optional[List[Tuple[float, float]]] = None
    """Optional list of point coordinates [(x, y), ...] for mask prediction"""
    labels: Optional[List[int]] = None
    """Optional labels for points (1 for positive, 0 for negative)"""
    box: Optional[Tuple[float, float, float, float]] = None
    """Optional bounding box coordinates (x1, y1, x2, y2)"""
    init_frame: int = 0
    """Frame index to start propagation"""
    stride: int = 1
    """Stride for visualizing/saving frames in video mode"""
    output_dir: Optional[Path] = None
    """Directory to save segmentation masks"""
    transparent_mask: bool = True
    """Save masks as PNG with transparency instead of black and white"""
    async_load: bool = True
    """Asynchronously load video frames"""
    progress: bool = False
    """Show progress bar"""
    show: bool = True
    """Show segmentations masks as overlay on images"""
    verbose: bool = False
    """Enable verbose logging"""
    quiet: bool = False
    """Suppress all logging except errors"""


def run(args: Args) -> npt.NDArray[np.bool_]:
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

    ckpt_parts = args.model.checkpoint.split('_')
    config_file = f"configs/sam2.1/{'_'.join(ckpt_parts[:-1] + [ckpt_parts[-1][0]])}.yaml"
    ckpt_path = str(Path(__file__).parent.parent / "checkpoints" / args.model.checkpoint)
    logger.debug(f"Using config file: {config_file}, Checkpoint: {ckpt_path}")

    device = args.model.device if args.model.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using device: {device}")

    points = None
    labels = None
    box = None
    masks = None

    if args.points:
        points = np.array(args.points, dtype=np.float32)
        for i in range(len(points)):
            if points[i, 0] < 1:
                points[i, 0] = image.shape[1] * points[i, 0]
            if points[i, 1] < 1:
                points[i, 1] = image.shape[0] * points[i, 1]
        labels = np.array(args.labels if args.labels else [1] * len(args.points), dtype=np.int32)
        if len(points) != len(labels):
            raise ValueError(f"Number of points ({len(points)}) and labels ({len(labels)}) do not match")
        logger.info(f"Using {len(points)} point(s) with label(s) {labels}")

    if args.box:
        box = np.array(args.box, dtype=np.float32)
        if box[0] < 1:
            box[0] = image.shape[1] * box[0]
        if box[1] < 1:
            box[1] = image.shape[0] * box[1]
        if box[2] < 1:
            box[2] = image.shape[1] * box[2]
        if box[3] < 1:
            box[3] = image.shape[0] * box[3]
        logger.info(f"Using box: {box}")

    if args.data.is_file():
        logger.info(f"Processing single image")
        image = np.array(Image.open(args.data).convert("RGB"))
        base_filename = args.data.stem

        figsize = (8, 4.5) if image.shape[1] > image.shape[0] else (4.5, 8)
        logger.debug(f"Using figure size: {figsize}")

        if args.model.huggingface:
            hf_name = str(f"facebook/{Path(args.model.checkpoint).stem}").replace("_", "-")
            logger.info(f"Loading checkpoint {hf_name} from Hugging Face model hub")
            with (contextlib.nullcontext() if args.quiet else contextlib.redirect_stderr(io.StringIO())):
                predictor = SAM2ImagePredictor.from_pretrained(
                    model_id=hf_name,
                    mask_threshold=args.model.mask_threshold,
                    max_hole_area=args.model.max_hole_area,
                    max_sprinkle_area=args.model.max_sprinkle_area,
                )
        else:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            predictor = SAM2ImagePredictor(
                sam_model=build_sam2(config_file, ckpt_path, device=device),
                mask_threshold=args.model.mask_threshold,
                max_hole_area=args.model.max_hole_area,
                max_sprinkle_area=args.model.max_sprinkle_area,
            )
        predictor.set_image(image)
        logger.debug(f"Image shape: {image.shape}")

        while True:
            if points is not None or box is not None:
                with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                    if points is not None:
                        logger.debug(f"Running inference with point(s): {points} and label(s): {labels}")
                    elif box is not None:
                        logger.debug(f"Running inference with box: {box}")
                    masks, scores, _ = predictor.predict(
                        point_coords=points,
                        point_labels=labels,
                        box=box,
                    )

                    sorted_ind = np.argsort(scores)[::-1]
                    masks = masks[sorted_ind] > 0
                    scores = scores[sorted_ind]
                    logger.debug(f"Generated {len(scores)} masks with scores: {scores}")
                    logger.debug(f"Best mask score: {scores[0]:.3f}")

                    if not args.output_dir and not args.show:
                        return masks[0]  # Return the best mask

            data = show_masks(
                image=image,
                masks=[None] if masks is None else [masks[0]],
                scores=None if masks is None else [scores[0]],
                point_coords=points,
                input_labels=labels,
                figsize=figsize,
                output_dir=args.output_dir,
                base_filename=base_filename,
                transparent_mask=args.transparent_mask,
            )

            if data is None:
                return masks[0]  # Return the best mask

            p, l = data
            if l is None:
                box, points = p, None
            else:
                if points is None:
                    points = p
                    labels = l
                else:
                    points = np.concatenate([points, p], axis=0)
                    labels = np.concatenate([labels, l], axis=0)

    elif args.data.is_dir():
        frame_names = [p for p in args.data.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        frame_names.sort(key=lambda p: int(''.join(filter(str.isdigit, p.stem))))
        logger.info(f"Processing video directory with {len(frame_names)} frames")

        image = np.asarray(Image.open(frame_names[args.init_frame]).convert("RGB"))
        figsize = (8, 4.5) if image.shape[1] > image.shape[0] else (4.5, 8)
        logger.debug(f"Using figure size: {figsize}")

        if args.model.huggingface:
            hf_name = str(f"facebook/{Path(args.model.checkpoint).stem}").replace("_", "-")
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
        with (contextlib.nullcontext()
              if getattr(args, 'progress', False) else contextlib.redirect_stderr(io.StringIO())):
            inference_state = predictor.init_state(video_path=str(args.data), async_loading_frames=args.async_load)

        all_points = None
        all_labels = None
        while True:
            if points is not None or box is not None:
                with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
                    if points is not None:
                        logger.debug(f"Adding new point(s): {points} with label(s) {labels}")
                    elif box is not None:
                        logger.debug(f"Adding new box: {box}")
                    _, _, masks = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=args.init_frame,
                        obj_id=0,
                        points=points,
                        labels=labels,
                        box=box,
                    )

            if args.points or args.box:
                break

            data = show_masks(
                image=image,
                masks=[None] if masks is None else [(m > 0).squeeze().cpu().numpy() for m in masks],
                point_coords=points if all_points is None else all_points,
                input_labels=labels if all_labels is None else all_labels,
                figsize=figsize,
            )

            if data is None:
                break

            points, labels = data
            if labels is None:
                box, points = points, None
            else:
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
                    logger.debug(f"Processing frame {frame_idx} with {len(object_ids)} object(s)")
                    video_segments[frame_idx] = {
                        object_id: (masks[i] > 0).squeeze().cpu().numpy() for i, object_id in enumerate(object_ids)
                    }

        if args.output_dir or args.show:
            stride = args.stride if args.stride > 1 or args.output_dir else len(frame_names) // 10
            logger.info(f"Rendering results (stride={stride})" if stride > 1 else "Rendering results")
            for frame_idx in trange(0, len(frame_names), stride, desc="render frames", disable=not args.progress):
                show_masks(
                    image=np.asarray(Image.open(frame_names[frame_idx])),
                    masks=video_segments[frame_idx].values(),
                    figsize=figsize,
                    output_dir=args.output_dir,
                    base_filename=frame_names[frame_idx].stem,
                    transparent_mask=args.transparent_mask,
                )

        # Return masks for the first object in each frame
        return np.array([list(vs.values())[0] for vs in video_segments.values()])


def main():
    run(tyro.cli(Args))


if __name__ == "__main__":
    main()

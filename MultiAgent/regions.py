"""Optional ordered region stage: GroundingDINO category proposals followed by SAM masks."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class SceneRegionSegmenter:
    """Produce category-level visible regions for visual context.

    These masks are visual aids, not candidate IDs. Candidate identity remains
    closed by ReferentialAgent and is never inferred from detector ordering.
    """
    def __init__(self, detector_model='/workspace/usr3/grounding-dino-base',
                 sam_model='/workspace/usr3/sam-base', device='cuda'):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
        self.torch = torch; self.device = device
        self.detector_processor = AutoProcessor.from_pretrained(detector_model, local_files_only=True)
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            detector_model, local_files_only=True).to(device).eval()
        self.sam_processor = SamProcessor.from_pretrained(sam_model, local_files_only=True)
        self.sam = SamModel.from_pretrained(sam_model, local_files_only=True).to(device).eval()

    def annotate(self, image: Image.Image, labels: list[str], output_dir=None, heatmap=None):
        labels = sorted({re.sub(r'\d+$', '', str(x).lower().strip()) for x in labels if str(x).strip()})
        if not labels:
            return image, []
        query = '. '.join(labels) + '.'
        inputs = self.detector_processor(images=image, text=query, return_tensors='pt').to(self.device)
        with self.torch.inference_mode(): outputs = self.detector(**inputs)
        target = [image.size[::-1]]
        post = self.detector_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=.25, text_threshold=.2, target_sizes=target)[0]
        detections = []
        for box, score, label in zip(post['boxes'], post['scores'], post['text_labels']):
            detections.append({'label': str(label), 'score': float(score),
                               'box': [float(x) for x in box.detach().cpu()]})
        if not detections:
            return image, []
        detections = sorted(detections, key=lambda d: -d['score'])[:32]
        boxes = [[d['box'] for d in detections]]
        sam_inputs = self.sam_processor(images=image, input_boxes=boxes, return_tensors='pt').to(self.device)
        with self.torch.inference_mode(): sam_out = self.sam(**sam_inputs)
        masks = self.sam_processor.image_processor.post_process_masks(
            sam_out.pred_masks.cpu(), sam_inputs['original_sizes'].cpu(),
            sam_inputs['reshaped_input_sizes'].cpu())[0]
        ious = sam_out.iou_scores.detach().float().cpu()
        if ious.ndim == 3: ious = ious[0]
        elif ious.ndim == 2 and ious.shape[0] == 1: ious = ious[0]
        canvas = image.convert('RGBA')
        palette = [(255, 90, 60, 85), (60, 180, 255, 85), (100, 220, 90, 85), (240, 190, 50, 85)]
        regions = []
        for idx, det in enumerate(detections):
            scores = ious[idx] if ious.ndim > 1 else ious
            mi = int(scores.argmax())
            mask = masks[idx, mi].numpy().astype(bool) if masks.ndim == 4 else masks[mi].numpy().astype(bool)
            color = palette[idx % len(palette)]
            layer_np = np.zeros((canvas.height, canvas.width, 4), dtype=np.uint8)
            layer_np[mask] = color
            canvas = Image.alpha_composite(canvas, Image.fromarray(layer_np, 'RGBA'))
            draw = ImageDraw.Draw(canvas)
            x0, y0, x1, y1 = det['box']; draw.rectangle((x0, y0, x1, y1), outline=(255, 240, 50, 230), width=3)
            draw.text((x0 + 3, y0 + 3), f"{det['label']} {det['score']:.2f}", fill=(255, 255, 255, 255))
            record = {**det, 'region_id': f'region{idx}', 'mask_area_pixels': int(mask.sum()),
                      'visible_area_fraction': float(mask.mean()), 'sam_iou': float(scores[mi]),
                      'gaze_response_mean': float(heatmap[mask].mean()) if heatmap is not None and mask.any() else None,
                      'occlusion_status': 'unknown_from_visible_mask'}
            if output_dir is not None:
                path = Path(output_dir)/f'region{idx}_mask.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((mask*255).astype(np.uint8)).save(path)
                record['mask_path'] = str(path.resolve())
            regions.append(record)
        return canvas.convert('RGB'), regions

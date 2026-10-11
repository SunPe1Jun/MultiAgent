"""Run local SAM on one GroundingDINO box from a real VR-TriRef frame."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import SamProcessor, SamModel

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--detections', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--model', default='/workspace/usr3/sam-base')
    p.add_argument('--detection-index', type=int, default=0)
    a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    image = Image.open(a.image).convert('RGB')
    payload = json.loads(Path(a.detections).read_text())
    det = payload['detections'][a.detection_index]
    box = [det['box']]
    processor = SamProcessor.from_pretrained(a.model, local_files_only=True)
    model = SamModel.from_pretrained(a.model, local_files_only=True).to('cuda').eval()
    inputs = processor(images=image, input_boxes=[box], return_tensors='pt').to('cuda')
    with torch.inference_mode(): outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs['original_sizes'].cpu(), inputs['reshaped_input_sizes'].cpu())
    # [batch, boxes, candidates, H, W], choose the highest predicted IoU mask.
    mask_stack = masks[0][0]
    scores = outputs.iou_scores[0, 0].detach().float().cpu()
    index = int(scores.argmax()); mask = mask_stack[index].numpy().astype(bool)
    mask_image = Image.fromarray((mask * 255).astype('uint8'))
    mask_image.save(out / 'mask.png')
    overlay = np.asarray(image).copy(); overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([0,255,0])).astype('uint8')
    Image.fromarray(overlay).save(out / 'overlay.png')
    result = {'image': str(Path(a.image).resolve()), 'model': str(Path(a.model).resolve()), 'detection': det,
              'mask_path': str((out/'mask.png').resolve()), 'overlay_path': str((out/'overlay.png').resolve()),
              'selected_mask_index': index, 'iou_scores': [float(x) for x in scores], 'mask_area_pixels': int(mask.sum())}
    (out / 'sam_result.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))

if __name__ == '__main__': main()

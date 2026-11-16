"""Object-shaped attention links with explicit ambiguity and unassigned mass.

Language proposes categories, a detector/segmenter proposes visible instances,
and scripts associate supported gaze density to masks. None are referent GT.
"""
from __future__ import annotations
import re
import numpy as np
from scipy.optimize import linear_sum_assignment

VERSION = 'visible_object_attention_links_v1'
# Pilot dictionary consumes only instruction words; no scene anchors or GT.
TERMS = {
 'person': ('person', 'worker', 'colleague', 'assistant', 'volunteer', 'man', 'woman'),
 'building': ('building', 'warehouse', 'warhouse'), 'door': ('door', 'entrance'),
 'chair': ('chair',), 'desk': ('desk',), 'table': ('table',),
 'file': ('file', 'document', 'folder'), 'lamp': ('lamp',), 'van': ('van',),
 'car': ('car',), 'truck': ('truck',), 'drone': ('drone',), 'camera': ('camera',),
 'curtain': ('curtain',), 'stove': ('stove',), 'cup': ('cup',), 'bench': ('bench',),
 'tank': ('tank',), 'hood': ('hood',), 'box': ('box', 'crate'),
}
QUERY = {'file': ['document', 'folder'], 'drone': ['drone', 'quadcopter'],
         'tank': ['tank', 'military vehicle'], 'hood': ['car hood']}


def instruction_categories(text):
    words = set(re.findall(r'[a-z]+', str(text).lower()))
    singular = {w[:-1] if w.endswith('s') and not w.endswith('ss') else w for w in words}
    return [name for name, aliases in TERMS.items() if singular.intersection(aliases)]


def detector_query(categories):
    terms = list(dict.fromkeys(t for name in categories for t in QUERY.get(name, [name])))
    return '. '.join(terms or ['object']) + '.'


def canonical_category(label):
    words = set(re.findall(r'[a-z]+', label.lower()))
    if 'military' in words and 'vehicle' in words: return 'tank'
    return next((name for name, aliases in TERMS.items() if words.intersection(aliases)), label.lower().strip())


def box_iou(a, b):
    lo = np.maximum(a[:2], b[:2]); hi = np.minimum(a[2:], b[2:])
    intersection = float(np.maximum(hi - lo, 0).prod())
    area = float(np.maximum(np.subtract(a[2:], a[:2]), 0).prod() + np.maximum(np.subtract(b[2:], b[:2]), 0).prod())
    return intersection / (area - intersection) if area > intersection else 0.


def deduplicate_detections(detections, width, height, threshold=.6):
    """Class-compatible NMS only; gaze and intended referents are not consulted."""
    kept = []
    for row in sorted(detections, key=lambda r: -r['score']):
        box = np.clip(row['box'], [0, 0, 0, 0], [width, height, width, height]).tolist()
        if not all(np.isfinite(box)) or box[2] <= box[0] or box[3] <= box[1]: continue
        category = canonical_category(row['label'])
        if any(k['category'] == category and box_iou(k['box'], box) > threshold for k in kept): continue
        kept.append({**row, 'box': box, 'category': category})
    return kept


def mask_attention_links(maps, masks):
    """Split density equally in overlapping masks, retain all unsupported mass.

    Returns an audit and an object-shaped display density, with conserved mass.
    A uniform fill of each object is for graphical association only. It does
    not assert that every pixel of an object was watched or that it is a target.
    """
    maps = np.asarray(maps, dtype=np.float64); masks = np.asarray(masks, dtype=bool)
    if maps.ndim != 3 or masks.ndim != 3 or masks.shape[1:] != maps.shape[1:]:
        raise ValueError('Expected time x H x W density and objects x H x W masks')
    if not np.all(np.isfinite(maps)) or np.any(maps < 0): raise ValueError('Invalid gaze mass')
    overlap = masks.sum(axis=0)
    shares = np.divide(masks, overlap[None], out=np.zeros(masks.shape, dtype=float), where=overlap[None] > 0)
    mass = np.einsum('khw,nhw->kn', maps, shares)
    area = masks.sum(axis=(1, 2))
    uniform = np.divide(mass, area[None], out=np.zeros_like(mass), where=area[None] > 0)
    display = np.einsum('kn,nhw->khw', uniform, masks.astype(float))
    unassigned = maps[:, overlap == 0].sum(axis=1)
    ambiguous = maps[:, overlap > 1].sum(axis=1)
    total = maps.sum(axis=(1, 2))
    return {'mass_seconds_by_phase_object': mass.tolist(), 'total_seconds': total.tolist(),
        'unassigned_seconds': unassigned.tolist(), 'ambiguous_seconds': ambiguous.tolist(),
        'assigned_seconds': mass.sum(axis=1).tolist(), 'mask_area_pixels': area.tolist(),
        'rule': 'Equal fractional membership in overlaps; never hard-select hottest object; preserve unassigned density.',
        'confidence_semantics': 'Engineering overlap, not target probability or semantic confidence.'}, display, np.where(overlap[None] == 0, maps, 0.)


def mask_iou(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b)) / union if union else 0.


def match_masks_one_to_one(first_masks, second_masks, first_categories=None, second_categories=None, minimum_iou=.3):
    """Geometry compatibility check, never an assertion of true object identity."""
    matrix = np.zeros((len(first_masks), len(second_masks)), float)
    for i, a in enumerate(first_masks):
        for j, b in enumerate(second_masks):
            if first_categories is not None and second_categories is not None and first_categories[i] != second_categories[j]: continue
            matrix[i, j] = mask_iou(a, b)
    if not matrix.size: return []
    a, b = linear_sum_assignment(-matrix)
    return [{'first': int(i), 'second': int(j), 'iou': float(matrix[i, j])} for i, j in zip(a, b) if matrix[i, j] >= minimum_iou]

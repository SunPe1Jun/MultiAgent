"""CPU directional panoramas and keyframe gaze views with auditable provenance.

B1 is rotation-only. B2 is unavailable without depth or a corresponding mesh.
A failed B1 attempt is retained and accompanied by a B3 world-indexed view atlas.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.spatial.transform import Rotation
from scipy.signal import correlate2d

from .gaze_evidence import build_heatmap, interpolate_camera_pose, render_presentation


UNKNOWN_RGB = (38, 38, 38)


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')


def camera_rotation(pose):
    return Rotation.from_quat(pose['rotation_xyzw'])


def pixel_rays(uv, image_size, fov_degrees):
    width, height = image_size
    points = np.asarray(uv, dtype=np.float64)
    tan = math.tan(math.radians(fov_degrees)/2)
    rays = np.stack(((points[..., 0]-.5)*2*tan*width/height,
                     (.5-points[..., 1])*2*tan,
                     np.ones_like(points[..., 0])), axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def sphere_directions(width, height):
    longitude = ((np.arange(width)+.5)/width-.5)*2*np.pi
    latitude = (.5-(np.arange(height)+.5)/height)*np.pi
    lon, lat = np.meshgrid(longitude, latitude)
    return np.stack((np.cos(lat)*np.sin(lon), np.sin(lat),
                     np.cos(lat)*np.cos(lon)), axis=-1)


def direction_uv(directions):
    value = np.asarray(directions, dtype=np.float64)
    value = value / np.linalg.norm(value, axis=-1, keepdims=True)
    lon = np.arctan2(value[..., 0], value[..., 2])
    lat = np.arcsin(np.clip(value[..., 1], -1, 1))
    return np.stack(((lon/(2*np.pi)+.5) % 1, .5-lat/np.pi), axis=-1)


def sample_directional_view(rgb, pose, reference_rotation, world_rays):
    height, width = rgb.shape[:2]
    local = camera_rotation(pose).inv().apply(world_rays.reshape(-1, 3)).reshape(world_rays.shape)
    tan = math.tan(math.radians(pose['fov_vertical_degrees'])/2)
    z = local[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = .5 + local[..., 0]/(2*z*tan*width/height)
        v = .5 - local[..., 1]/(2*z*tan)
    valid = (z > 1e-6) & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    coordinates = np.stack((v*height-.5, u*width-.5))
    coordinates = np.nan_to_num(coordinates, nan=-2, posinf=-2, neginf=-2)
    warped = np.stack([map_coordinates(rgb[..., channel], coordinates, order=1,
                                       mode='constant', cval=0) for channel in range(3)], axis=-1)
    return warped.astype(np.uint8), valid, np.maximum(z, 0)


def spherical_heatmap(directions, weights, shape, sigma_degrees, coverage=None):
    """Use actual angular distance; latitude scaling and wrap seam are implicit."""
    height, width = shape
    rays = sphere_directions(width, height)
    heat = np.zeros((height, width), dtype=np.float32)
    sigma = math.radians(sigma_degrees)
    threshold = math.cos(min(math.pi, 3*sigma))
    for direction, weight in zip(directions, weights):
        dot = np.einsum('hwc,c->hw', rays, np.asarray(direction))
        mask = dot >= threshold
        angles = np.arccos(np.clip(dot[mask], -1, 1))
        heat[mask] += (float(weight)*np.exp(-angles*angles/(2*sigma*sigma))).astype(np.float32)
    if coverage is not None:
        heat[~coverage] = 0
    return heat


def spherical_foveation(image, heat, coverage, config):
    """Mask-aware angular blur with longitude-wrap and latitude-scaled width."""
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
    if not np.any(heat > 0):
        return image.copy(), {'status': 'raw_rgb_fallback_no_gaze'}
    height, width = heat.shape
    focus = 1-np.exp(-np.maximum(heat, 0)*float(config.get('display_scale', 2)))
    floor = float(config.get('foveation_min_clear_fraction', .55))
    angular_sigma = float(config.get('spherical_blur_sigma_degrees', 1.8))
    sigma_y = angular_sigma*height/180
    mask = coverage.astype(np.float32)
    vertical_weight = gaussian_filter(mask, sigma=(sigma_y, 0), mode=('nearest', 'wrap'))
    vertical = np.stack([gaussian_filter(rgb[..., c]*mask, sigma=(sigma_y, 0),
                                        mode=('nearest', 'wrap')) for c in range(3)], axis=-1)
    blurred = np.empty_like(rgb)
    from scipy.ndimage import gaussian_filter1d
    for row in range(height):
        latitude = (.5-(row+.5)/height)*np.pi
        sigma_x = min(width/5, angular_sigma*width/(360*max(.08, math.cos(latitude))))
        denominator = gaussian_filter1d(vertical_weight[row], sigma_x, mode='wrap')
        for channel in range(3):
            numerator = gaussian_filter1d(vertical[row, :, channel], sigma_x, mode='wrap')
            blurred[row, :, channel] = numerator/np.maximum(denominator, 1e-6)
    sharp = floor+(1-floor)*focus
    output = np.clip(rgb*sharp[..., None]+blurred*(1-sharp[..., None]), 0, 255).astype(np.uint8)
    output[~coverage] = UNKNOWN_RGB
    return Image.fromarray(output), {'status': 'mask_aware_angular_foveation',
        'clear_fraction_floor': floor, 'blur_sigma_degrees': angular_sigma,
        'longitude_boundary': 'wrap', 'latitude_longitude_scale': '1/cos(latitude)',
        'unknown_region_handling': 'excluded from blur weights; retained unknown'}


def heat_guide(heat, time_numerator=None, scale=2., coverage=None):
    strength = np.clip(1-np.exp(-np.maximum(heat, 0)*scale), 0, 1)
    # Pure saliency is deliberately separate from RGB, as in the reference paper.
    pixels = np.repeat((strength*255).astype(np.uint8)[..., None], 3, axis=2)
    if time_numerator is not None:
        time = np.divide(time_numerator, heat, out=np.zeros_like(heat), where=heat>0)
        early, recent = np.array([40, 60, 180]), np.array([255, 245, 80])
        color = early[None, None]+np.clip(time, 0, 1)[..., None]*(recent-early)[None, None]
        pixels = np.clip(color*strength[..., None], 0, 255).astype(np.uint8)
    if coverage is not None:
        pixels[~coverage] = UNKNOWN_RGB
    return Image.fromarray(pixels)


def view_atlas(frames, *, paths_key='path', tile=320, columns=4):
    rows = math.ceil(len(frames)/columns)
    canvas = Image.new('RGB', (columns*tile, rows*(tile+32)), UNKNOWN_RGB)
    draw = ImageDraw.Draw(canvas)
    for index, frame in enumerate(frames):
        x, y = (index % columns)*tile, (index//columns)*(tile+32)
        image = Image.open(frame[paths_key]).convert('RGB')
        image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        canvas.paste(image, (x, y+32))
        draw.text((x+5, y+4), f"{frame['view_id']}  t={frame['time_seconds']:.3f}s", fill='white')
    return canvas


def registration_residual(reference, moving, overlap, *, max_points=48, radius=5,
                          search=10, min_ncc=.8):
    """Post-warp NCC displacement; exploratory, not a certified static match."""
    if int(overlap.sum()) < 2000:
        return {'status': 'insufficient_overlap', 'accepted_matches': 0}
    a = np.asarray(reference, dtype=np.float32).mean(axis=2)
    b = np.asarray(moving, dtype=np.float32).mean(axis=2)
    gradient = np.hypot(*np.gradient(a))
    gradient[~overlap] = 0
    margin = radius+search+2
    gradient[:margin] = gradient[-margin:] = 0
    gradient[:, :margin] = gradient[:, -margin:] = 0
    matches = []
    used = np.zeros_like(overlap)
    for flat in np.argsort(gradient.ravel())[::-1]:
        y, x = np.unravel_index(flat, gradient.shape)
        if gradient[y, x] < 8 or len(matches) >= max_points:
            break
        if used[y, x] or not overlap[y-radius:y+radius+1, x-radius:x+radius+1].all():
            continue
        used[max(0,y-18):y+19,max(0,x-18):x+19] = True
        patch = a[y-radius:y+radius+1, x-radius:x+radius+1].copy()
        patch -= patch.mean()
        energy = float((patch*patch).sum())
        if energy < 20:
            continue
        region = b[y-radius-search:y+radius+search+1, x-radius-search:x+radius+search+1]
        kernel = np.ones_like(patch)
        sums = correlate2d(region, kernel, mode='valid')
        squared = correlate2d(region*region, kernel, mode='valid')
        numerator = correlate2d(region, patch, mode='valid')
        denominator = np.sqrt(np.maximum(squared-sums*sums/patch.size, 1e-5)*energy)
        ncc = numerator/denominator
        yy, xx = np.unravel_index(np.argmax(ncc), ncc.shape)
        score = float(ncc[yy, xx])
        if score < min_ncc:
            continue
        dx, dy = int(xx-search), int(yy-search)
        matches.append({'pixel_xy': [int(x), int(y)], 'residual_xy_px': [dx, dy],
                        'residual_px': float(math.hypot(dx, dy)), 'ncc': score})
    residuals = [m['residual_px'] for m in matches]
    return {'status': 'measured' if residuals else 'no_reliable_photometric_matches',
            'accepted_matches': len(matches),
            'median_px': float(np.median(residuals)) if residuals else None,
            'p95_px': float(np.percentile(residuals, 95)) if residuals else None,
            'matching_assumptions': 'local photometric NCC after rotation warp; dynamic/ambiguous features may remain',
            'search_radius_px': search, 'matches': matches}


def build_directional_panorama(frames, records, output_dir, *, offset, config,
                                shape=(384, 768), compute_residuals=True,
                                gaze_observations=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    poses = [(frame, interpolate_camera_pose(
        records, float(frame['time_seconds'])-float(offset) if offset is not None else None,
        config.get('max_gap_seconds', .25))) for frame in frames]
    missing = [frame['view_id'] for frame, pose in poses if pose is None]
    poses = [(frame, pose) for frame, pose in poses if pose is not None]
    if not poses:
        geometry = {'projection_type': 'directional_panorama_attempt',
            'status': 'FAILED', 'failure_reasons': ['no_frame_has_corresponding_camera_pose'],
            'missing_pose_view_ids': missing,
            'world_geometry_accuracy_claimed': False,
            'B2': {'status': 'NOT_AVAILABLE', 'reason': 'no reliable depth or correspondable scene mesh'}}
        save(output_dir/'geometry.json', geometry)
        view_atlas(frames).save(output_dir/'spatial_atlas_RGB.jpg', quality=94)
        save(output_dir/'spatial_atlas_index.json', {'projection_type': 'spatial_atlas',
            'world_coordinate_system': 'recorded Unity camera/world; physical scale uncalibrated',
            'views': frames, 'time_order': [row['view_id'] for row in frames],
            'gaze_time_index': None, 'is_panorama': False,
            'fallback_from': 'no_frame_has_corresponding_camera_pose'})
        return None
    reference = camera_rotation(poses[0][1])
    height, width = shape
    rays = sphere_directions(width, height)
    world_rays = reference.apply(rays.reshape(-1, 3)).reshape(rays.shape)
    canvas = np.full((height, width, 3), UNKNOWN_RGB, dtype=np.uint8)
    coverage = np.zeros((height, width), dtype=np.uint16)
    source = np.full((height, width), -1, dtype=np.int16)
    first_time = np.full((height, width), np.inf, dtype=np.float32)
    last_time = np.full((height, width), -np.inf, dtype=np.float32)
    best = np.zeros((height, width), dtype=np.float32)
    gaze_links = {}
    residuals = []
    positions = []
    source_rows = []
    last_warp = last_mask = None
    for index, (frame, pose) in enumerate(poses):
        rgb = np.asarray(Image.open(frame['path']).convert('RGB'))
        warp, valid, weights = sample_directional_view(rgb, pose, reference, world_rays)
        if compute_residuals and last_warp is not None:
            metric = registration_residual(last_warp, warp, valid & last_mask)
            residuals.append({'source_view_ids': [source_rows[-1]['view_id'], frame['view_id']], **metric})
        last_warp, last_mask = warp, valid
        choose = valid & (weights >= best)
        canvas[choose] = warp[choose]
        source[choose] = index
        best[choose] = weights[choose]
        coverage += valid.astype(np.uint16)
        first_time[valid] = np.minimum(first_time[valid], frame['time_seconds'])
        last_time[valid] = np.maximum(last_time[valid], frame['time_seconds'])
        positions.append(pose['position_world'])
        source_rows.append({**frame, 'panorama_source_index': index,
                            'camera_pose': pose,
                            'source_frame_ids': frame.get('source_frame_ids', [frame.get('source_frame_id')]),
                            'time_range': [frame['time_seconds'], frame['time_seconds']]})
        if gaze_observations is None:
            for item in frame.get('projected_gaze_samples', []):
                if item.get('status') != 'projected_frustum_only':
                    continue
                uv = item['normalized_uv']
                ray = pixel_rays(uv, (rgb.shape[1], rgb.shape[0]), pose['fov_vertical_degrees'])
                direction = reference.inv().apply(camera_rotation(pose).apply(ray))
                gap = abs(float(item['gaze_time_seconds']) -
                          (float(frame['time_seconds'])-float(offset or 0.)))
                source_index = int(item['source_index'])
                row = {**item, 'direction_reference': direction.tolist(),
                       'panorama_uv': direction_uv(direction).tolist(),
                       'associated_view_id': frame['view_id'],
                       'source_frame_ids': frame.get('source_frame_ids', [frame.get('source_frame_id')]),
                       'rgb_gaze_delta_seconds': gap,
                       'geometry': 'same rotation-only RGB ray model; point is finite relative to observation camera'}
                if source_index not in gaze_links or gap < gaze_links[source_index]['rgb_gaze_delta_seconds']:
                    gaze_links[source_index] = row
    if gaze_observations is not None and poses:
        for item in gaze_observations:
            if item.get('status') != 'projected_frustum_only':
                continue
            direction = reference.inv().apply(np.asarray(item['direction_world'], dtype=np.float64))
            row = {**item, 'direction_reference': direction.tolist(),
                   'panorama_uv': direction_uv(direction).tolist(),
                   'geometry': 'same rotation-only camera ray model used for RGB; no translation or depth'}
            gaze_links[int(item['source_index'])] = row
    gaze = sorted(gaze_links.values(), key=lambda row: (row['gaze_time_seconds'], row['source_index']))
    weights = [float(row['weight_seconds']) for row in gaze]
    directions = [row['direction_reference'] for row in gaze]
    mask = coverage > 0
    heat = spherical_heatmap(directions, weights, shape, config.get('kernel_sigma_degrees', 2), mask)
    times = [row['gaze_time_seconds'] for row in gaze]
    start, end = (min(times), max(times)) if times else (0, 0)
    normalized_times = [(time-start)/max(end-start, 1e-6) for time in times]
    time_numerator = spherical_heatmap(directions, [w*t for w,t in zip(weights, normalized_times)],
                                     shape, config.get('kernel_sigma_degrees', 2), mask)
    layers = {}
    for name, bounds in [('early', (0, 1/3)), ('middle', (1/3, 2/3)), ('late', (2/3, 1.00001))]:
        keep = [i for i,t in enumerate(normalized_times) if bounds[0] <= t < bounds[1]]
        layers[name] = spherical_heatmap([directions[i] for i in keep], [weights[i] for i in keep],
                                       shape, config.get('kernel_sigma_degrees', 2), mask)
    raw_image = Image.fromarray(canvas)
    raw_image.save(output_dir/'RGB.png')
    heat_guide(heat, coverage=mask).save(output_dir/'heatmap.png')
    heat_guide(heat, time_numerator, coverage=mask).save(output_dir/'temporal_heatmap.png')
    overlay, render_stats = render_presentation(raw_image, heat, 'H_local', config)
    overlay = np.asarray(overlay).copy()
    overlay[~mask] = UNKNOWN_RGB
    Image.fromarray(overlay).save(output_dir/'heatmap_overlay.png')
    foveated, foveation_stats = spherical_foveation(raw_image, heat, mask, config)
    foveated.save(output_dir/'foveation.png')
    Image.fromarray((mask*255).astype(np.uint8)).save(output_dir/'coverage.png')
    first_time[~mask] = np.nan
    last_time[~mask] = np.nan
    np.savez_compressed(output_dir/'attention_and_provenance.npz', H_unnormalized=heat,
        temporal_numerator=time_numerator, source_index=source, coverage_count=coverage,
        observation_first_time=first_time, observation_last_time=last_time,
        **{f'H_{name}': value for name,value in layers.items()})
    tiles = []
    for y in range(0, height, 32):
        for x in range(0, width, 32):
            ids = np.unique(source[y:y+32,x:x+32])
            ids = [int(i) for i in ids if i >= 0]
            if not ids:
                continue
            rows = [source_rows[i] for i in ids]
            tiles.append({'pixel_bounds': [x,y,min(width,x+32),min(height,y+32)],
                          'source_frame_ids': sorted({fid for row in rows for fid in row['source_frame_ids']}),
                          'source_view_ids': [row['view_id'] for row in rows],
                          'time_range': [min(row['time_seconds'] for row in rows), max(row['time_seconds'] for row in rows)]})
    translation = float(np.max(np.linalg.norm(np.asarray(positions)-positions[0], axis=1)))
    values = [item['median_px'] for item in residuals if item.get('median_px') is not None]
    high_residual = any(value > float(config.get('panorama_max_median_residual_px', 5)) for value in values)
    failure_reasons = []
    if missing:
        failure_reasons.append('some_source_frames_have_no_corresponding_camera_pose')
    if not values:
        failure_reasons.append('rotation_registration_not_verified')
    if high_residual:
        failure_reasons.append('rotation_only_warp_has_large_photometric_displacement_residual')
    if float(mask.mean()) < .02:
        failure_reasons.append('insufficient_spherical_coverage')
    if not gaze:
        failure_reasons.append('no_corresponding_valid_gaze_observations')
    geometry = {'projection_type': 'directional_panorama',
        'status': 'VALIDITY_UNVERIFIED' if not failure_reasons else 'FALLBACK_SPATIAL_ATLAS',
        'failure_reasons': failure_reasons,
        'angular_mapping': 'equirectangular: longitude [-pi,pi), latitude [+pi/2,-pi/2]',
        'reference_rotation_xyzw': poses[0][1]['rotation_xyzw'],
        'reference_pose_time_seconds': poses[0][0]['time_seconds'],
        'reference_time_is_not_observation_time_for_all_pixels': True,
        'camera_axis_convention': 'local +z forward, +y up, top-left image origin; xyzw quaternion applied as recorded; handedness is not independently calibrated',
        'fov_convention': 'recorded scalar cameraFOV treated as vertical field of view with square pixels',
        'shape_height_width': [height,width], 'source_frame_count': len(frames),
        'projected_source_frame_count': len(poses), 'missing_pose_view_ids': missing,
        'coverage_fraction': float(mask.mean()),
        'camera_max_displacement_world_units': translation,
        'translation_handling': 'ignored in directional B1; residual and unknown metric scale retained',
        'world_geometry_accuracy_claimed': False,
        'registration_residuals': residuals,
        'gaze_projection_model': 'same observation-camera rotation rays as RGB; no anchor association',
        'gaze_spherical_kernel': 'Gaussian of geodesic angular distance, 3sigma cutoff; seam and latitude accounted',
        'gaze_unique_source_count': len(gaze),
        'gaze_valid_evidence_duration_seconds': float(sum(weights)),
        'temporal_layers': list(layers), 'temporal_layer_range_seconds': [start,end],
        'time_color_semantics': 'hue/brightness moves blue/dark to yellow/light with weighted mean source time; strength remains separate',
        'full_causal_order_retained_in_single_heatmap': False,
        'unknown_color': list(UNKNOWN_RGB), 'generated_completion': False,
        'source_regions': tiles, 'source_views': source_rows,
        'heatmap_render': render_stats, 'foveation_render': foveation_stats,
        'B2': {'status': 'NOT_AVAILABLE', 'reason': 'no reliable depth or correspondable scene mesh'},
    }
    save(output_dir/'geometry.json', geometry)
    save(output_dir/'gaze_time_index.json', gaze)
    # B3 is always available for review; used as the formal fallback if B1 fails checks.
    pose_by_view = {row['view_id']: row['camera_pose'] for row in source_rows}
    atlas_frames = [{**frame, 'path': frame['path'],
                     'source_frame_ids': frame.get('source_frame_ids', [frame.get('source_frame_id')]),
                     'camera_pose': pose_by_view.get(frame['view_id'])}
                    for frame in frames]
    view_atlas(atlas_frames).save(output_dir/'spatial_atlas_RGB.jpg', quality=94)
    save(output_dir/'spatial_atlas_index.json', {'projection_type': 'spatial_atlas',
        'world_coordinate_system': 'recorded Unity camera/world; physical scale uncalibrated',
        'views': atlas_frames, 'time_order': [row['view_id'] for row in atlas_frames],
        'gaze_time_index': 'gaze_time_index.json',
        'is_panorama': False, 'fallback_from': geometry['status']})
    return geometry


def run_organization_study(args, config):
    from .visual_organization_audit import run
    return run(args, config)

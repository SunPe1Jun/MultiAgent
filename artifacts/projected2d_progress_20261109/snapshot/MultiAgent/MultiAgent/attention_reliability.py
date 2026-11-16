"""Frozen paired attention diagnostics, with separate task/annotation provenance.

A controlled cue label describes an injected display, never a true referent.
Real referent-center labels remain an explicitly weak geometric proxy.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.ndimage import distance_transform_edt

VERSION='paired_attention_reliability_v1'


def distinct_masks(a,b,maximum_overlap=.05):
    a=np.asarray(a,bool);b=np.asarray(b,bool)
    if a.shape!=b.shape or a.ndim!=2:raise ValueError('Two equally sized 2D masks required')
    minimum=min(int(a.sum()),int(b.sum()))
    return minimum>0 and np.count_nonzero(a&b)/minimum<maximum_overlap


def interior_point(mask):
    mask=np.asarray(mask,bool)
    if mask.ndim!=2 or not mask.any():raise ValueError('Nonempty 2D mask required')
    padded=np.pad(mask,1,constant_values=False)
    distance=distance_transform_edt(padded)[1:-1,1:-1]
    y,x=np.unravel_index(np.argmax(distance),distance.shape)
    return [float(x),float(y)]


def injected_history(masks,reverse=False):
    """Known cue control on real RGB: early region0, late region1; no object GT."""
    masks=np.asarray(masks,bool)
    if masks.ndim!=3 or masks.shape[0]!=2 or not all(m.any() for m in masks):
        raise ValueError('Two nonempty masks required')
    mass=np.array([[.6,0.],[.15,.15],[0.,.6]],float)
    if reverse:mass=mass[::-1].copy()
    return mass


def pair_attention(maps,masks,condition):
    """All unassigned evidence stays explicit; controls preserve raw arrays."""
    from .object_attention import mask_attention_links
    links,_,residual=mask_attention_links(maps,masks)
    mass=np.asarray(links['mass_seconds_by_phase_object'],float)
    if condition=='TRUE_HEAT':return mass,residual,links
    if condition=='SWAP_PAIR':return mass[:,::-1].copy(),residual,links
    if condition=='NO_HEAT':return np.zeros_like(mass),np.zeros_like(residual),links
    raise ValueError(condition)


def graphical_pair_heat(mass,masks,residual):
    """Orange whole-object association, blue unassigned density, support-gated.

    This is a visual association graphic, not seconds per display pixel. The
    mass itself is stored separately, conserved, and never entered as text.
    """
    mass=np.asarray(mass,float);masks=np.asarray(masks,bool);residual=np.asarray(residual,float)
    if mass.shape!=(3,2) or masks.shape[0]!=2 or residual.shape!=(3,*masks.shape[1:]):
        raise ValueError('Three bins, two masks, and registered residual required')
    overlap=masks.sum(0)
    value=np.einsum('ki,ihw->khw',mass,masks.astype(float))
    value=np.divide(value,overlap[None],out=np.zeros_like(value),where=overlap[None]>0)
    peak=float(mass.max())
    # Relative normalization must not turn negligible supported time into a bright cue.
    support=min(1.,float(mass.sum())/.25)
    orange=.9*np.sqrt(np.clip(value/peak,0,1))*support if peak>0 else np.zeros_like(value)
    rpeak=float(residual.max());rsupport=min(1.,float(residual.sum())/.25)
    blue=.7*np.sqrt(residual/rpeak)*rsupport if rpeak>0 else np.zeros_like(residual)
    return orange,blue,{'pair_supported_seconds':float(mass.sum()),'display_support_gate':support,
        'unassigned_supported_seconds':float(residual.sum()),'graphical_values_are_target_probabilities':False}


def heat_display_policy(pair_supported_seconds, minimum_seconds=.25):
    """Select a display condition only; never select an object or use GT.

    This support gate does not certify gaze reliability, object identity or
    intended referents. Unsupported evidence stays in the stored raw memory.
    """
    if not math.isfinite(pair_supported_seconds) or pair_supported_seconds<0:
        raise ValueError('Finite nonnegative supported duration required')
    if not math.isfinite(minimum_seconds) or minimum_seconds<=0:
        raise ValueError('Positive minimum support required')
    return 'TRUE_HEAT' if pair_supported_seconds>=minimum_seconds else 'NO_HEAT'


def cluster_paired_interval(rows,*,seed=20261106,replicates=5000):
    """Percentile bootstrap over interactions, preserving paired variants/cases.

    Rows contain cluster, baseline and treatment scores. This describes the
    selected regression cohort only; it cannot correct weak or synthetic labels.
    """
    if not rows:return {'n':0,'clusters':0,'gain':None,'interval_95':None}
    groups={}
    for row in rows:
        before=float(row['baseline']);after=float(row['treatment'])
        if not all(math.isfinite(v) and 0<=v<=1 for v in [before,after]):raise ValueError('Scores must be in [0,1]')
        groups.setdefault(str(row['cluster']),[]).append(after-before)
    cluster_values=list(groups.values());counts=np.array([len(v) for v in cluster_values]);sums=np.array([sum(v) for v in cluster_values])
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(groups),size=(replicates,len(groups)))
    samples=sums[indices].sum(1)/counts[indices].sum(1)
    return {'n':len(rows),'clusters':len(groups),'gain':float(sums.sum()/counts.sum()),
        'interval_95':np.quantile(samples,[.025,.975]).tolist(),'replicates':replicates,
        'resampling_unit':'interaction, with paired conditions and ID permutations kept together',
        'scope':'Exploratory selected regression cohort; label validity and unseen generalization not established.'}

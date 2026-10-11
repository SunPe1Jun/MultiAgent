"""Resolve each language mention independently, then union the selected candidate sets."""
from collections import defaultdict


def weighted_fusion(ctx, evidence, weights, mentions=()):
    selected, decisions, contributions = [], [], defaultdict(float)
    for mention in mentions:
        scores = {name: 0. for name in mention['candidate_ids']}
        per_source = {}
        times = defaultdict(list)
        for e in evidence:
            if e.mention_id != mention['mention_id'] or e.candidate_id not in scores:
                continue
            contribution = weights.get(e.source, 0.) * e.support_score * e.confidence
            scores[e.candidate_id] += contribution
            per_source.setdefault(e.source, {})[e.candidate_id] = e.support_score
            contributions[e.source] += contribution
            if e.temporal_range and contribution > 0:
                times[e.candidate_id].append({'source': e.source, 'range': e.temporal_range,
                                              'contribution': contribution})
        ranked = sorted(scores, key=lambda name: (-scores[name], name))
        positive = [name for name in ranked if scores[name] > 0]
        count = mention.get('count')
        if mention.get('explicit_ids'):
            chosen = mention['explicit_ids']
        elif len(ranked) == 1:
            chosen = ranked  # unique language-compatible instance; not evidence fabrication
        elif count is None:
            chosen = [name for name in positive if scores[name] >= .7 * scores[positive[0]]] if positive else []
        else:
            chosen = positive[:count]
        for name in chosen:
            if name not in selected:
                selected.append(name)
        decisions.append({**mention, 'selected_object_names': chosen,
                          'status': 'resolved' if chosen else 'unresolved',
                          'ranked_candidates': [{'candidate_id': n, 'score': scores[n]} for n in ranked],
                          'scores_by_source': per_source,
                          'selected_evidence_windows': {n: times[n] for n in chosen}})
    resolved = [ctx.candidate(n).world_row() for n in selected]
    return {'selected_object_name': selected[0] if selected else '', 'selected_object_names': selected,
            'resolved_object_rows': resolved, 'resolved_object_row': resolved[0] if resolved else None,
            'mention_decisions': decisions, 'evidence': [e.to_dict() for e in evidence],
            'primary_source': max(contributions, key=contributions.get) if any(contributions.values()) else 'language_or_none',
            'unresolved_mentions': [m['mention_id'] for m in decisions if m['status'] == 'unresolved']}

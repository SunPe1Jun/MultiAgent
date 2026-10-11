"""Read-only lexical GT audit; flags require semantic/visual review, not auto-editing."""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .baseline import ROOT, module, scene_paths
from .mention_mapping import category, tokens, singular, ALIASES


def normalized(text):
    return ' '.join(singular(w) for w in tokens(text))


def audit(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, stats = [], []
    api, ev = module('api'), module('eval')
    for scene in ['scene1', 'scene2', 'scene3', *[f'scene4_room{i}' for i in range(1, 5)], 'scene5']:
        paths = scene_paths(scene)
        anchors = api.load_scene_anchor_table(paths['anchor'])
        aliases, _ = ev.load_anchor_alias_map(paths['anchor'])
        gt_rows = ev.load_gt_rows(paths['gt'], aliases)
        lexicon = defaultdict(set)
        classes = {category(a['object_name']) for a in anchors}
        for a in anchors:
            cls = category(a['object_name'])
            for phrase in [cls, a['object_name'], *a.get('aliases', [])]:
                if normalized(phrase): lexicon[normalized(phrase)].add(cls)
        # Generic synonyms are used only if the local vocabulary has no entry.
        for phrase, targets in ALIASES.items():
            if normalized(phrase) not in lexicon:
                lexicon[normalized(phrase)].update(c for c in targets if c in classes)
        count = Counter()
        for gt in gt_rows:
            text = normalized(gt['instruction'])
            matches = []
            for phrase, options in lexicon.items():
                if not options: continue
                for m in re.finditer(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', text):
                    matches.append((m.start(), m.end(), phrase, options))
            # Long phrases win over subphrases (floor lamp vs lamp, dining table vs table).
            matches = [m for m in matches if not any(
                n[0] <= m[0] and m[1] <= n[1] and n[1]-n[0] > m[1]-m[0] for n in matches)]
            gt_classes = {category(n) for n in gt['mapped_referents']}
            missing = [{'phrase':p, 'possible_classes':sorted(cs)} for _, _, p, cs in matches
                       if not (cs & gt_classes)]
            found = set().union(*(m[3] for m in matches)) if matches else set()
            extra = sorted(gt_classes - found)
            flags = []
            if missing: flags.append('mentioned_class_missing_from_gt')
            if extra: flags.append('gt_class_without_literal_support')
            if gt['unmapped_referents']: flags.append('unmapped_gt_label')
            if flags:
                count.update(flags)
                rows.append({'scene':scene, 'workbook':str(paths['gt']),
                             'excel_row':gt['excel_row_index'], 'id':gt['number'],
                             'instruction':gt['instruction'], 'gt_raw':gt['referents_raw'],
                             'gt_mapped':gt['mapped_referents'], 'flags':flags,
                             'mentioned_but_missing':missing, 'nonliteral_gt_classes':extra,
                             'unmapped_gt':gt['unmapped_referents'],
                             'status':'needs_review_not_confirmed_error'})
        stats.append({'scene':scene,'scanned':len(gt_rows),
                      'flagged_rows':sum(r['scene']==scene for r in rows),**count})
    (output/'suspected_gt_issues.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    with (output/'suspected_gt_issues.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['scene'])
        w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in row.items()})
    (output/'scan_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', required=True)
    audit(p.parse_args().output_dir)

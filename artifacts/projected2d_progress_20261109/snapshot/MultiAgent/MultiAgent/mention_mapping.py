"""Language-only spans followed by scene-aware, deterministic type mapping.

This layer never reads GT, visual evidence or earlier instance predictions.
"""
import copy
import re

VERSION = 'typed_mentions_v10_scene_local_deictic_unresolved'
ALIASES = {
    'box': ('cargo',), 'crate': ('cargo',), 'warehouse': ('building',),
    'warhouse': ('building',), 'worker': ('person',), 'people': ('person',),
    'guy': ('person',), 'fellow': ('person',), 'fellows': ('person',),
    'man': ('person',), 'men': ('person',), 'woman': ('person',),
    'colleague': ('person',), 'assistant': ('person',), 'volunteer': ('person',),
    'table': ('desk',), 'computer': ('pc', 'laptop'), 'document': ('file',),
    'folder': ('file',),
}
COMPONENTS = {'door', 'window', 'entrance', 'gate', 'cover', 'flap', 'edge',
              'surface', 'center', 'lid', 'hatch', 'panel', 'latch', 'strap',
              'brake', 'fork', 'hood', 'tailgate', 'handle', 'battery', 'gimbal', 'camera'}
VAGUE = {'there', 'here', 'inside', 'outside', 'gap', 'area', 'region', 'location'}


def tokens(text):
    return re.findall(r'[a-z0-9]+', str(text).lower())


def singular(word):
    if word.endswith('ies'): return word[:-3] + 'y'
    if word.endswith('xes'): return word[:-2]
    return word[:-1] if word.endswith('s') and not word.endswith('ss') else word


def category(label):
    base = re.sub(r'(?:\d+|[A-Z])$', '', label).lower().replace('_', ' ')
    return ' '.join(singular(word) for word in base.split())


def literal_span(text, instruction, occurrence=0):
    if not isinstance(text, str) or not text.strip(): return None
    matches = list(re.finditer(r'(?<!\w)' + re.escape(text) + r'(?!\w)', instruction, re.I))
    return [matches[occurrence].start(), matches[occurrence].end()] if len(matches)>occurrence else None


def repair_explicit_component_parent(payload, instruction):
    """Repair a parser split only when the full component phrase is literal.

    Uses the instruction string alone. It never consults candidate tables,
    geometry, GT, or model predictions about object identity. A phrase like
    ``the crate beside the truck`` cannot be merged because the model must mark
    a component and the exact ``text of parent`` phrase must occur in the input.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get('mentions'), list):
        return payload, []
    repaired = copy.deepcopy(payload)
    mentions = repaired['mentions']
    remove_indices = set()
    audit = []
    for index, mention in enumerate(mentions):
        if not isinstance(mention, dict):
            continue
        text = mention.get('text')
        parent = mention.get('parent_text') or ''
        component = mention.get('component') or ''
        object_type = str(mention.get('object_type') or '')
        if not isinstance(text, str) or not isinstance(parent, str) or not parent:
            continue
        component_head = bool(set(tokens(object_type)) & COMPONENTS)
        if not component and not component_head:
            continue
        if literal_span(parent, text) is not None:
            continue
        combined = next((candidate for candidate in (f'{text} of {parent}', f'{text} {parent}')
                         if literal_span(candidate, instruction) is not None), None)
        if combined is None:
            continue
        mention['text'] = combined
        mention['_literal_component_parent_repair'] = {
            'original_text': text, 'original_parent_text': parent,
            'repaired_text': combined, 'reason': 'exact_instruction_span'}
        combined_span = literal_span(combined, instruction)
        removed = None
        if combined_span:
            for other_index in range(index + 1, len(mentions)):
                other = mentions[other_index]
                if (isinstance(other, dict) and isinstance(other.get('text'), str)
                        and other['text'].casefold() == parent.casefold()):
                    occurrence = sum(1 for previous in mentions[:other_index]
                                    if isinstance(previous, dict)
                                    and str(previous.get('text') or '').casefold() == parent.casefold())
                    parent_span = literal_span(parent, instruction, occurrence)
                    if (parent_span and combined_span[0] <= parent_span[0]
                            and parent_span[1] <= combined_span[1]):
                        remove_indices.add(other_index)
                        removed = other_index
                    break
        audit.append({'mention_index': index, 'original_text': text,
                      'original_parent_text': parent, 'repaired_text': combined,
                      'removed_duplicate_parent_index': removed})
    if remove_indices:
        repaired['mentions'] = [row for i, row in enumerate(mentions) if i not in remove_indices]
    if audit:
        repaired['_literal_component_parent_repairs'] = audit
    return repaired, audit


def actor_coverage_checks(text, mentions):
    pattern = re.compile(r'\b(?:the|that|this|another|a|an)\s+(?:(?:other|young|old|front|back)\s+)*'
                         r'(?:person|people|fellow|worker|man|woman|guy|colleague|assistant|volunteer)\b', re.I)
    spans = [m.get('span') for m in mentions if isinstance(m.get('span'), list) and len(m['span']) == 2]
    return [{'text': match.group(), 'span': [match.start(), match.end()]}
            for match in pattern.finditer(text)
            if not any(a <= match.start() and match.end() <= b for a, b in spans)]


def map_phrase(phrase, candidates):
    """Map against this scene's candidate schema before generic aliases.

    Anchor-table aliases are scene-local.  They must win over broad rules such
    as ``table -> desk`` because the same spoken noun can denote a different
    canonical class in another room.
    """
    words = [singular(w) for w in tokens(phrase)
             if w not in {'the', 'a', 'an', 'this', 'that', 'these', 'those'}]
    classes = {category(c.candidate_id) for c in candidates}
    phrase_class = ' '.join(words)
    # Prefer an exact multi-word scene class (e.g. dining table) before the
    # generic table->desk alias. This keeps room-specific anchor schemas intact.
    exact_phrase = {phrase_class} & classes if phrase_class else set()
    direct = set(words) & classes
    contains = {c for c in classes if len(words) == 1 and words[0] in c.split()}
    phrase_tokens = set(words)
    phrase_contains = {c for c in classes if set(c.split()) <= phrase_tokens}

    def alias_classes(candidate):
        aliases = getattr(candidate, 'aliases', ()) or ()
        if isinstance(aliases, str):
            aliases = (aliases,)
        return {category(alias) for alias in aliases if str(alias).strip()}

    local_exact = set()
    local_contains = set()
    for candidate in candidates:
        alias_set = alias_classes(candidate)
        if phrase_class and phrase_class in alias_set:
            local_exact.add(category(candidate.candidate_id))
        if len(words) == 1 and any(words[0] in alias.split() for alias in alias_set):
            local_contains.add(category(candidate.candidate_id))

    # Scene-local exact names/aliases have highest priority, followed by the
    # existing class matching. Generic aliases are only a final fallback.
    local_phrase_contains = set()
    for candidate in candidates:
        for alias in alias_classes(candidate):
            if set(alias.split()) <= phrase_tokens:
                local_phrase_contains.add(category(candidate.candidate_id))

    wanted = (exact_phrase or local_exact or direct or contains or phrase_contains
              or local_contains or local_phrase_contains
              or {a for w in words for a in ALIASES.get(w, ()) if a in classes})
    return [c.candidate_id for c in candidates if category(c.candidate_id) in wanted]


def resolve_mentions(ctx, payload):
    if not isinstance(payload, dict) or not isinstance(payload.get('mentions'), list):
        raise ValueError('mentions array required')
    if 'selected_object_names' in payload:
        raise ValueError('Instance selections forbidden in language extraction')
    if not payload['mentions'] and ctx.text.strip():
        raise ValueError('Empty extraction for nonempty instruction')
    repair_audit = payload.get('_literal_component_parent_repairs', [])
    result = []
    occurrences = {}
    for index, raw in enumerate(payload['mentions']):
        if not isinstance(raw, dict): raise ValueError('mention must be an object')
        if any(k in raw for k in ('candidate_ids', 'matched_object_names', 'selected_object_names')):
            raise ValueError('Instance selections forbidden in language extraction')
        phrase = raw.get('text')
        key = phrase.lower() if isinstance(phrase,str) else ''
        occurrence = occurrences.get(key,0)
        span = literal_span(phrase, ctx.text, occurrence)
        if span is None: raise ValueError(f'Mention {index} is not a literal instruction span')
        occurrences[key] = occurrence+1
        head = raw.get('object_type')
        if not isinstance(head, str) or not head.strip(): raise ValueError('object_type required')
        if head.strip().casefold() in {'instruction', 'utterance', 'sentence', 'command'}:
            raise ValueError('instruction-level text is not an object mention; extract its referent noun phrases')
        count = raw.get('count', 1)
        if count is not None and (type(count) is not int or not 1 <= count <= 20):
            raise ValueError('count must be null or an integer in [1,20]')
        parent = raw.get('parent_text') or ''
        if not isinstance(parent, str): raise ValueError('parent_text must be text')
        component = raw.get('component') or ''
        if component and literal_span(component, raw['text']) is None:
            raise ValueError('component must be a literal span within mention text')
        component_head = bool(set(map(singular, tokens(head))) & COMPONENTS)
        repaired_pronoun_parent = ''
        if parent and literal_span(parent, raw['text']) is None:
            anchor_text = raw.get('anchor_text') or ''
            matching_antecedent = any(
                other['span'][0] < span[0]
                and other['coreference_id'] == raw.get('coreference_id')
                and other['text'].casefold() == str(anchor_text).casefold()
                for other in result)
            pronoun_parent = bool(re.match(r'^(?:its|their|his|her)\b', raw['text'], re.I))
            bare_anaphor = raw['text'].strip().casefold() in {'it', 'them', 'he', 'she', 'him', 'her', 'they'}
            if ((component_head and pronoun_parent or bare_anaphor) and matching_antecedent
                    and parent.casefold() == str(anchor_text).casefold()):
                repaired_pronoun_parent = parent
                parent = ''
            else:
                raise ValueError(
                    f'parent_text {parent!r} must be a literal span inside mention {raw["text"]!r}')
        # Fallback only for explicit part-of syntax WITHIN this mention.
        # "near the truck" must never turn a door into a truck.
        part_match = re.search(r'\bof\s+(.+)$', raw['text'], re.I) if component_head else None
        if not parent and part_match:
            parent = part_match.group(1)
        if not parent and component_head:
            possessive = re.match(r"(.+?)[’']s\s+", raw['text'], re.I)
            if possessive:
                parent = possessive.group(1)
        own = map_phrase(head, ctx.candidates)
        own_reason = None
        if not own and head.strip().lower() in {'side', 'area', 'region', 'location'}:
            # Some spatial phrases normalize to a role word, e.g. "bedroom
            # side" -> object_type="side". Recover the scene-local anchor
            # from the full literal phrase without using a global synonym.
            phrase_ids = map_phrase(raw['text'], ctx.candidates)
            if phrase_ids:
                own = phrase_ids
                own_reason = 'phrase_local_alias'
        if parent and (component or component_head):
            ids = map_phrase(parent, ctx.candidates)
            reason = 'component_parent' if ids else 'unmapped_parent'
        else:
            ids = own
            reason = own_reason if own_reason else ('type_or_alias' if ids else 'unmapped_type')
        # A deictic location is intentionally unresolved by language alone. It
        # reaches the visual stage with no mention candidate IDs; the visual
        # agent records an observation, not an all-anchor candidate set.
        # Concrete location nouns (e.g. "the bedroom side") keep their local
        # lexical mapping even when the parser labels them kind=location.
        vague_words = set(tokens(raw['text'])) - {'the', 'a', 'an', 'over', 'right', 'to'}
        if (vague_words and vague_words <= VAGUE or
                raw.get('kind') in ('region', 'location') and not ids):
            ids = []
            reason = 'contextual_location_unresolved'
        # Bare class names (chair/window) are also valid anchor IDs in some
        # rooms. Their presence in natural language must not remove chair2 or
        # window2 from the candidate pool. Only marked IDs are explicit.
        explicit = [c.candidate_id for c in ctx.candidates
                    if re.search(r'(?:\d+|[A-Z])$', c.candidate_id) or '_' in c.candidate_id]
        explicit = [name for name in explicit if re.search(
            r'(?<!\w)' + re.escape(name) + r'(?!\w)', raw['text'], re.I)]
        if explicit:
            ids, reason = explicit, 'explicit_instance'
        # Keep ASR head in the spoken phrase, independent of canonical cargo/building.
        spoken_head = next((w for w in reversed(tokens(head)) if w in tokens(raw['text'])), '')
        result.append({**raw, 'mention_id': f'm{index}', 'span': span,
                       'parent_text': parent, 'component': component,
                       'repaired_pronoun_parent_text': repaired_pronoun_parent or None,
                       'count': count, 'candidate_ids': ids, 'explicit_ids': explicit,
                       'canonical_types': sorted({category(n) for n in ids}),
                       'spoken_head': spoken_head, 'anchor_text': raw.get('anchor_text') or '',
                       'mapping_status': 'mapped' if ids else reason,
                       'candidate_mapping_reason': reason,
                       'coreference_id': str(raw.get('coreference_id') or f'r{index}')})
    # A parent phrase nested inside a component is the same occurrence, not
    # an additional referent. Keep later standalone repetitions: they carry
    # their own temporal evidence and may refer to a different instance.
    suppressed = []
    kept = []
    for mention in result:
        owner = next((other for other in result
                      if other is not mention and other['parent_text']
                      and (other['component'] or bool(set(map(singular, tokens(other['object_type']))) & COMPONENTS))
                      and not mention['parent_text']
                      and other['span'][0] <= mention['span'][0]
                      and mention['span'][1] <= other['span'][1]
                      and other['parent_text'].casefold() == mention['text'].casefold()
                      and mention['candidate_ids']
                      and set(other['candidate_ids']) == set(mention['candidate_ids'])), None)
        if owner:
            suppressed.append({**mention, 'reason': 'nested_component_parent',
                               'covered_by_mention_id': owner['mention_id']})
        else:
            kept.append(mention)
    result = kept
    # Components that refer back to the same entity inherit its compatible
    # candidate pool. Pronoun-only parents use anchor_text plus shared coreference.
    for mention in result:
        if mention['explicit_ids']:
            continue
        antecedents = [other for other in result
                       if other['span'][0] < mention['span'][0]
                       and not other['parent_text']
                       and other['coreference_id'] == mention['coreference_id']]
        if antecedents:
            previous = antecedents[-1]
            explicit_parent = bool(mention['parent_text'])
            pronoun_component = bool(mention['component']) and not explicit_parent
            bare_anaphor = mention['text'].strip().casefold() in {'it', 'them', 'he', 'she', 'him', 'her', 'they'}
            if not explicit_parent and not pronoun_component and not bare_anaphor:
                continue
            shared = ([name for name in previous['candidate_ids']
                       if name in mention['candidate_ids']] if mention['candidate_ids']
                      else list(previous['candidate_ids']))
            if shared:
                mention.update(candidate_ids=shared,
                               canonical_types=sorted({category(n) for n in shared}),
                               mapping_status='mapped',
                               candidate_mapping_reason=('coreference_anaphor' if bare_anaphor else
                                                         'coreference_component_parent' if pronoun_component
                                                         else 'coreference_parent'),
                               parent_mention_id=previous['mention_id'])
    output = {'instruction': ctx.text, 'source': 'instruction_only_qwen',
              'mapping_version': VERSION, 'mentions': result,
              'suppressed_nested_mentions': suppressed}
    if repair_audit:
        output['literal_component_parent_repairs'] = repair_audit
    return output


PROMPT = '''Extract referent phrases using ONLY the instruction. Do not select instances.
Include actors, action objects, source objects and concrete spatial anchors.
Do not omit a generic actor: phrases such as "the other person", "that fellow", "the man",
"another worker" and "that guy" are explicit actor mentions and must be returned exactly.
Before producing JSON, check that every explicit noun phrase introduced by a determiner
(the/that/this/these/those/a/an/another/other) is either represented by a mention or is
only a pronoun whose antecedent is recorded. Do not silently drop an actor because it has
no unique lexical anchor.
Copy text EXACTLY from the instruction, including spelling errors. object_type may be normalized.
For a component, use the FULL noun phrase: "left door of this truck" is one mention,
component="left door", parent_text="this truck", object_type="door". Do not also emit a
separate truck mention for that same phrase. "window of the front building" similarly
has parent_text="the front building". parent_text must be inside text. Spatial relations
like "chair near the desk" produce two objects, not a component-parent relationship.
parent_text must be a strict literal part of text, never equal the full mention. For
"its side panel", use parent_text="" and anchor_text=<the antecedent literal mention>;
reuse that antecedent's coreference_id. A relational phrase such as "the crate beside
the truck" stays two mentions; do not make the whole phrase the crate's parent.
Keep box as box; a separate deterministic stage maps box to cargo when appropriate.
For possessive parts such as "that drone's battery", parent_text="that drone";
the output target remains the whole parent anchor, not a new component task.
For action pronouns such as "it" or "them", do not invent another physical target.
Record each pronoun and its antecedent in a short coreference_note attached to that
antecedent's mention. For example, in "pick up the
spray bottle and place it on the rack", the spray bottle is the moved target and the rack
is its destination; do not make "it" an untyped missing target.
Keep vague here/there as kind=location. They have NO lexical anchor candidates;
the Visual agent will report an unresolved spatial observation, not every scene anchor.
Repeated references to the SAME entity share coreference_id; different instances do not.
List mentions in instruction order. Identical phrases at different positions may refer to
different instances: do not assume two occurrences of "that chair" share an identity.
count=1 for singular, 2 for an explicit pair, null for genuinely unspecified plural.
Keep all explicitly mentioned objects, never choose only the main action target.
Return compact JSON {"mentions":[{"text":"exact phrase","object_type":"singular noun",
"component":"literal component or empty","parent_text":"literal parent phrase or empty",
"kind":"object|location","role":"actor|target|source|destination|anchor",
"relation":"brief relation or empty","anchor_text":"other literal mention or empty",
"coreference_id":"r0","coreference_note":"literal pronoun -> antecedent phrase, or empty","count":1}]}.
Instruction: '''


class MentionExtractor:
    def __init__(self, backend): self.backend = backend

    def run(self, ctx):
        errors = []
        for attempt in range(3):
            prompt = PROMPT + ctx.text
            if errors: prompt += '\nCorrect this validation error and return ALL mentions: ' + errors[-1]
            try:
                raw = self.backend.ask(f'{ctx.scene_key}/{ctx.event_id}',
                                       f'v4_extract_{attempt}', prompt)
                result = resolve_mentions(ctx, raw)
                result['validation_retries'] = errors
                return result
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError('; '.join(errors))

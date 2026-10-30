"""Validate the organizer's image partition before any B training reads."""
from __future__ import annotations

from .resources import SPEC


def split_protocol(split):
    """Translate a validated official split into the continuation engine's indices."""
    if not isinstance(split, dict):
        raise ValueError('Split must be an object')
    names = split.get('images')
    if (not isinstance(names, list) or not names
            or any(not isinstance(n, str) or not n or '/' in n or '\\' in n for n in names)
            or names != sorted(set(names))):
        raise ValueError('Image names must be unique, sorted basenames')
    partitions = {}
    for key in ('train', 'test', 'context'):
        values = split.get(key)
        if (not isinstance(values, list) or any(not isinstance(n, str) for n in values)
                or len(values) != len(set(values)) or not set(values) <= set(names)):
            raise ValueError(f'Invalid {key} names')
        partitions[key] = set(values)
    train, test, context = (partitions[key] for key in ('train', 'test', 'context'))
    if train & test or train | test != set(names):
        raise ValueError('Train and test must partition every registered image')
    if not context <= train or len(context) != SPEC['source_count'] or not test:
        raise ValueError('Need exactly 16 training context views and nonempty heldout test')
    hold = SPEC['llffhold']
    if split.get('llffhold') != hold or split['test'] != names[::hold]:
        raise ValueError('Competition requires llffhold=8 in sorted image order')
    shape = split.get('resolution')
    if (not isinstance(shape, list) or len(shape) != 2
            or any(type(v) is not int or v < 64 or v % 64 for v in shape)):
        raise ValueError('Image dimensions must be positive multiples of 64')
    h, w = shape
    limit = SPEC['portrait_max_resolution'] if h > w else SPEC['max_resolution']
    if max(shape) > limit:
        raise ValueError('Image dimensions exceed organizer resolution policy')
    ids = {name: i for i, name in enumerate(names)}
    sources = [ids[name] for name in split['context']]
    return {
        'names': names, 'candidate_pool': [ids[name] for name in split['train']],
        'source_baseline': sources, 'selected_sources': sources,
        'development': [ids[name] for name in split['test']],
        'shape_hw': shape, 'metric_scope': 'official-llffhold8-test',
    }

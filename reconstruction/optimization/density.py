"""Competition density configuration and Gaussian/Adam invariants."""
from __future__ import annotations
import copy
import math

DEFAULT_CONFIG = {
    'sh_decay': .3, 'max_points': 2000000,
    'densify_start': 100, 'densify_stop': 3200, 'densify_interval': 100,
    'alpha_prune': .005, 'decode_workers': 16,
    'geometry_options': {}, 'relocation_options': {},
}


def resolve_config(overrides=None):
    overrides = dict(overrides or {})
    unknown = overrides.keys() - DEFAULT_CONFIG.keys()
    if unknown:
        raise ValueError(f'Unknown density config keys: {sorted(unknown)}')
    config = {**copy.deepcopy(DEFAULT_CONFIG), **overrides}
    for key in ('max_points', 'densify_start', 'densify_interval', 'decode_workers'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(config['densify_stop']) is not int or config['densify_stop'] < 0:
        raise ValueError('densify_stop must be a nonnegative integer')
    if config['densify_stop'] and config['densify_start'] > config['densify_stop']:
        raise ValueError('density step range invalid')
    for key in ('sh_decay', 'alpha_prune'):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite')
    if not 0 <= config['sh_decay'] <= 1 or not 0 < config['alpha_prune'] < 1:
        raise ValueError('SH decay / alpha threshold out of range')
    for key in ('geometry_options', 'relocation_options'):
        if not isinstance(config[key], dict):
            raise ValueError(f'{key} must be an options object')
    return config


def topology_due(config, step):
    return (config['densify_start'] <= step <= config['densify_stop'] and
            step % config['densify_interval'] == 0)




PARAMETERS = {'xyz': '_xyz', 'f_dc': '_features_dc', 'f_rest': '_features_rest',
              'opacity': '_opacity', 'scaling': '_scaling', 'rotation': '_rotation'}


def zero_moments(optimizer, parameter, selected):
    state = optimizer.state.get(parameter, {})
    for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
        if name in state:
            state[name][selected] = 0


def assert_optimizer_topology(g):
    n = len(g.get_xyz)
    for group in g.optimizer.param_groups:
        p = group['params'][0]
        if p.shape[0] != n:
            raise AssertionError('Optimizer point count mismatch')
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            v = g.optimizer.state.get(p, {}).get(key)
            if v is not None and v.shape != p.shape:
                raise AssertionError(f'Adam {key} shape mismatch')

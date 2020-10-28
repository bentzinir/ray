import yaml
import os
from deepmerge import always_merger


class Config(dict):
    def __init__(self, name='default', d=None):
        if d:
            self._load_from_dict(d)
        else:
            self._load_config(name)

    def save_partial(self, path, keys=None, **kwargs):
        if keys:
            dict_file = dict((k, self[k]) for k in keys)
        else:
            dict_file = dict(self)
        dict_file.update(kwargs)

        with open(os.path.join(path), 'w') as f:
            yaml.dump(dict_file, f, default_flow_style=False)

    def merge(self, d: dict, override=False):
        '''
        :param d: dictionary to merge into config
        :param override: when true, will override any values that are already in config
        The function will always add values that are not in the original config
        '''
        if override:
            self._load_from_dict(always_merger.merge(self, d))
        else:
            self._load_from_dict(always_merger.merge(d, self))

    def _load_config(self, path):
        with open(path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            self._load_from_dict(data)

    def _load_from_dict(self, d):
        for key, val in d.items():
            if key == 'device':
                continue
            if type(val) == str:
                if val.lower() == 'nan' or val.lower() == 'none':
                    val = None
            if type(val) == dict:
                val = Config(d=val)
            self[key] = val

    @staticmethod
    def _override_config(config, args):
        for key in config.keys():
            if type(config[key]) == Config:
                Config._override_config(config[key], args)
            elif key in args.keys():
                val = args[key]
                if val and val != "None" and val != -1:
                    config[key] = val

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<Config ' + dict.__repr__(self) + '>'

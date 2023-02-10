import yaml
from pathlib import Path


class Config(dict):
    def __init__(
            self,
            *args,
            init_paths: bool = True,
            init_dirs: bool = True,
            read_config: bool = True,
            **kwargs
    ):
        super(Config, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

        if init_paths:
            self.path_to_root = Path(__file__).parent.parent.resolve()
            self.path_to_data = self.path_to_root / 'data'
            self.path_to_output = self.path_to_root / 'output'
            self.path_to_models = self.path_to_output / 'models'
            self.path_to_predictions = self.path_to_output / 'predictions'
            self.path_to_tensorboard_logs = self.path_to_output / 'tensorboard_logs'
            self.path_to_configs = self.path_to_root / 'configs'

        if init_dirs:
            self._init_dirs()

        if read_config:
            self._read_config()

    def _read_config(self):
        with open(self.path_to_configs / "config.yml", 'r') as f:
            config_yaml = yaml.safe_load(f)

        self.update(Config.dict_to_map(config_yaml))
        
    def _init_dirs(self):
        self.path_to_data.mkdir(exist_ok=True)
        self.path_to_output.mkdir(exist_ok=True)
        self.path_to_models.mkdir(exist_ok=True)
        self.path_to_predictions.mkdir(exist_ok=True)
        self.path_to_tensorboard_logs.mkdir(exist_ok=True)
        self.path_to_configs.mkdir(exist_ok=True)

    @staticmethod
    def dict_to_map(obj):
        for k, v in obj.items():
            if isinstance(v, dict):
                obj[k] = Config(Config.dict_to_map(v), init_paths=False, init_dirs=False, read_config=False)
        return obj

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Config, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Config, self).__delitem__(key)
        del self.__dict__[key]
        
        
opt = Config()

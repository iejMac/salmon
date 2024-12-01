"""general config class."""

class Config:
    def __init__(self, obj, params={}):
        self.obj = obj
        self.init_params = params

    def get_params(self):
        # Return common JSON-serializable types (for checkpointing)
        return {k: v for k, v in self.init_params.items() if isinstance(v, (int, float, str, list, dict, bool, type(None)))}

    def __getitem__(self, key):
        return self.init_params[key]
    def __setitem__(self, key, value):
        self.init_params[key] = value

    def build(self, **more_params):
        params = {**self.init_params, **more_params}
        return self.obj(**params)

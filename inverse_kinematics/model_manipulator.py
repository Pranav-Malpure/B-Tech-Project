from dm_control import mjcf

class Manipulator_mjcf(object):

    def __init__(self):
        self._model = mjcf.RootElement()
        
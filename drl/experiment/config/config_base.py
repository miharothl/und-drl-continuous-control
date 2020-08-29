from typing import List


class ConfigBase(object):

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @classmethod
    def from_json(cls, data):
        return cls(**data)

    def ensure_betwen_0_and_1(self, param: float):

        if param is None:
            return

        if not (0 <= param <= 1):
            raise ValueError("Hyper-parameter must be between 0 and 1.")

    def ensure_is_greater(self, param1, param2):

        if (param1 is None) or (param2 is None):
            return

        if not (param1 > param2):
            raise ValueError("Hyper-parameter param1 must be greater than param2.")

    def ensure_in_list(self, param, valid: List):
        if param not in valid:
            raise ValueError("Hypter-parameter must be one of {}.".format(valid))


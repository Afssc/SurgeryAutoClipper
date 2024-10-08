from abc import ABCMeta,abstractmethod
class cutter(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def clip(self) -> tuple:
        raise NotImplementedError   
    
    def from_dict(self,json_dict:dict) -> None:
        for key, value in json_dict.items():
            setattr(self, key, value)
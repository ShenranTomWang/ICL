from abc import abstractmethod, ABC
from transformers import AutoTokenizer

class Tokenizer(ABC):
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
    
    @abstractmethod
    def get_option_id(self, option: str) -> dict:
        """
        Tokenize an options.
        
        Args:
            options: str
        Returns:
            dict
        """
        pass
    
    def __call__(self, inputs: list[str], **kwds):
        return self.tokenizer(inputs, **kwds)
    
    def decode(self, input_ids: list[int]):
        return self.tokenizer.decode(input_ids)
from abc import abstractmethod, ABC
from transformers import AutoTokenizer

class Tokenizer(ABC):
    """
    This is an abstract base class for tokenizers. It defines the interface for tokenizers used in the ICL framework.
    """
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
    
    @abstractmethod
    def get_option_id(self, option: str) -> dict:
        """
        Tokenize an option. This will add whitespace to the option and then tokenize it and return the first token,
        if the option turns out to be more than 1 token.
        
        Args:
            option: str
        Returns:
            dict
        """
        pass
    
    def __call__(self, inputs: list[str], **kwds):
        return self.tokenizer(inputs, **kwds)
    
    def decode(self, input_ids: list[int]):
        return self.tokenizer.decode(input_ids)
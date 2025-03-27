from .tokenizer import Tokenizer
from transformers import AutoTokenizer

class StandardTokenizer(Tokenizer):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__(tokenizer)
    
    def get_option_id(self, option: str) -> dict:
        """
        Find the id of option.
        
        Args:
            options: str
        Returns:
            dict<str, tensor>
        """
        return self(" " + option).input_ids[0]
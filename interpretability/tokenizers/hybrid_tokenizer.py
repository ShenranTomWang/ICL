from .tokenizer import Tokenizer
from transformers import AutoTokenizer

class HybridTokenizer(Tokenizer):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__(tokenizer)

    def get_option_id(self, option: str) -> dict:
        """
        Tokenize an option.

        Args:
            option (str)

        Returns:
            dict
        """
        return self(" " + option).input_ids[2]
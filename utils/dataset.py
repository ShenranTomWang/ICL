import logging
from interpretability.tokenizers import Tokenizer

class Dataset:
    def __init__(self, train: list, test: list, verbose=False):
        self.train = train
        self.test = test
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        self.task = train[0]["task"] if len(train) > 0 else test[0]["task"]
        self.demo = None
        self.options = train[0]["options"] if len(train) > 0 else test[0]["options"]
        self.inputs = None
        self.outputs = None
        self.output_ids = None
        self.indices = None
        self.option_ids = None
    
    def prepare_demo(self) -> None:
        """
        Requires:
            self.train is not None
        Effects:
            self.demo: str
        """
        demo = ""
        for dp_train in self.train:
            demo += dp_train["input"]
            dp_train["options"] = self.options
            demo += "\n "
            demo += dp_train["output"]
            demo += "\n\n"
        self.demo = demo
        
    def preprocess(self) -> None:
        """
        Args:
        Requires:
            self.test is not None
            self.train is not None
        Effects:
            self.test: list<{"input": str, "output": str, "options": list<str>}>
            self.inputs: list<str>, 
            self.outputs: list<str>
        """
        if self.demo is None:
            self.prepare_demo()
        for i, dp_test in enumerate(self.test):
            self.test[i]["input"] = self.demo + dp_test["input"] + "\n"
        self.inputs = [dp["input"] for dp in self.test]
        self.outputs = [dp["output"] for dp in self.test]
        
    def tensorize(self, tokenizer: Tokenizer) -> None:
        """
        Args:
            tokenizer: Tokenizer object
        Effects:
            self.inputs: list<{"input_ids": tensor, "attention_mask": tensor}>, 
            self.output_ids: list<int>, 
            self.indices: list<int>,
            self.option_ids: list<int>
        """
        if self.inputs is None or self.outputs is None:
            self.preprocess()
            
        inputs = self.inputs
            
        self.demo_tokenized = tokenizer([self.demo], return_tensors="pt")
        
        inputs = [tokenizer(input, return_tensors="pt") for input in inputs]
        inputs = [
            {
                "input_ids": input["input_ids"],
                "attention_mask": input["attention_mask"]
            }
            for input in inputs
        ]
        indices = [input["input_ids"].shape[1] - 1 for input in inputs]
        output_ids = [tokenizer.get_option_id(output) for output in self.outputs]
        option_ids = [tokenizer.get_option_id(option) for option in self.options]
        if self.verbose:
            self.logger.info(f"inputs example (string): {self.inputs[0]}")
            self.logger.info(f"inputs example: {inputs[0]}")
            self.logger.info(f"output id example: {output_ids[0]}")
            self.logger.info(f"option ids: {option_ids}")
        self.inputs_tokenized = inputs
        self.output_ids = output_ids
        self.indices = indices
        self.option_ids = option_ids
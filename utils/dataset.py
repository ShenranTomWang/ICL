import logging
from interpretability.tokenizers import Tokenizer
import numpy as np

class Dataset:
    """
    ICL Dataset class.
    """
    def __init__(self, train: list, test: list, verbose: bool = False, template: bool = False, options: list = None):
        self.train = train
        self.test = test
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        self.template = template
        self.task = train[0]["task"] if len(train) > 0 else test[0]["task"]
        self.demo = None
        self.options = options if options else (train[0]["options"] if len(train) > 0 else test[0]["options"])
        self.inputs = None
        self.outputs = None
        self.output_ids = None
        self.option_ids = None
        
    def choose(self, k: int, seed: int) -> None:
        """
        Choose k examples from the training set to use as demos.
        Args:
            k: int, number of examples to choose
            seed: int, seed for random number generator
        Requires:
            self.train is not None
        Effects:
            self.train: list<{"input": str, "output": str, "options": list<str>}>
        """
        np.random.seed(seed)
        if len(self.train) >= k:
            self.train = np.random.choice(self.train, k, replace=False).tolist()
    
    def prepare_demo(self) -> None:
        """
        Prepares the demonstrations string for ICL. This will create demo of format
            "<input>\\n <output>\\n\\n"
        for each example in the training set. If self.template is True, it will create:
            "Q: <input>\\n A: <output>\\n\\n"
        The resulting string will be stored in self.demo.
        """
        demo = ""
        for dp_train in self.train:
            if self.template:
                demo += "Q: "
            demo += dp_train["input"]
            dp_train["options"] = self.options
            demo += "\n "
            if self.template:
                demo += "A: "
            demo += dp_train["output"]
            demo += "\n\n"
        self.demo = demo
        
    def preprocess(self) -> None:
        """
        Appends the demo to each test example. The resulting string will be stored in self.inputs.
        Will also apply template if self.template is True.
        
        Effects:
            self.test: list<{"input": str, "output": str, "options": list<str>}>
            self.inputs: list<str>, 
            self.outputs: list<str>
        """
        if self.demo is None:
            self.prepare_demo()
        for i, dp_test in enumerate(self.test):
            if self.template:
                dp_test["input"] = self.demo + "Q: " + dp_test["input"] + "\nA:"
            else:
                self.test[i]["input"] = self.demo + dp_test["input"] + "\n"
        self.inputs = [dp["input"] for dp in self.test]
        self.outputs = [dp["output"] for dp in self.test]
        
    def tensorize(self, tokenizer: Tokenizer) -> None:
        """
        Tensorize inputs, outputs and options using the tokenizer.
        This will create a list of dictionaries for each input, with the keys "input_ids" and "attention_mask".
        The output_ids and option_ids will be created using the tokenizer.get_option_id method.
        The resulting tensors will be stored in self.inputs_tokenized, self.output_ids and self.option_ids.
        This function will also create a tensor for the demo, which will be stored in self.demo_tokenized.
        Args:
            tokenizer: Tokenizer object
        Effects:
            self.inputs_tokenized: list<{"input_ids": tensor, "attention_mask": tensor}>,
            self.output_ids: list<int>,
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
        output_ids = [tokenizer.get_option_id(output) for output in self.outputs]
        option_ids = [tokenizer.get_option_id(option) for option in self.options]
        if self.verbose:
            self.logger.info(f"inputs example (string): {self.inputs[0]}")
            self.logger.info(f"inputs example: {inputs[0]}")
            self.logger.info(f"output id example: {output_ids[0]}")
            self.logger.info(f"option ids: {option_ids}")
        self.inputs_tokenized = inputs
        self.output_ids = output_ids
        self.option_ids = option_ids
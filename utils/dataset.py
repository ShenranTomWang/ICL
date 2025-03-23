import logging

class Dataset:
    def __init__(self, train: list, test: list, verbose=False, add_newlines=True, n_skips=1):
        self.train = train
        self.add_newlines = add_newlines
        self.test = test
        self.n_skips = n_skips
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        self.task = train[0]["task"]
        self.demo = None
        self.options_raw = train[0]["options"]
        if add_newlines:
            self.options = ["\n" + option for option in train[0]["options"]]
        else:
            self.options = [" " + option for option in train[0]["options"]]
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
            demo += ("\n" if self.add_newlines else " ")
            demo += dp_train["output"]
            if self.add_newlines:
                demo += "\n\n"
        self.demo = demo
        
    def preprocess(self, use_demo=True) -> None:
        """
        Args:
            use_demo (bool): whether to add demo in front of test data
        Requires:
            self.test is not None
            self.train is not None
        Effects:
            self.test: list<{"input": str, "output": str, "options": list<str>}>
            self.inputs: list<str>, 
            self.outputs: list<str>
        """
        if self.demo is None or use_demo:
            self.prepare_demo()
        for i, dp_test in enumerate(self.test):
            self.test[i]["input"] = (self.demo if use_demo else "") + dp_test["input"] + ("\n" if self.add_newlines else " ")
            self.test[i]["output"] = ("\n" if self.add_newlines else " ") + dp_test["output"]
        self.inputs = [dp["input"] for dp in self.test]
        self.outputs = [dp["output"] for dp in self.test]
        
    def tensorize(self, tokenizer, use_demo=True) -> None:
        """
        TODO: should support batched tokenization
        Args:
            tokenizer: transformers tokenizer
            use_demo (bool): whether to add demo in front of test data
        Effects:
            self.inputs: list<{"input_ids": tensor, "attention_mask": tensor}>, 
            self.output_ids: list<int>, 
            self.indices: list<int>,
            self.option_ids: list<int>
        """
        if self.inputs is None or self.outputs is None:
            self.preprocess(use_demo)
            
        inputs = self.inputs
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        self.demo_tokenized = tokenizer([self.demo], return_tensors="pt", padding=True, truncation=True)
        
        inputs = [tokenizer(input, return_tensors="pt", padding=True, truncation=True) for input in inputs]
        inputs = [
            {
                "input_ids": input["input_ids"][..., 1:] if not use_demo else input["input_ids"],                   # skip bos token if not using demo
                "attention_mask": input["attention_mask"][..., 1:] if not use_demo else input["attention_mask"]     # skip bos token if not using demo
            }
            for input in inputs
        ]
        indices = [input["input_ids"].shape[1] - 1 for input in inputs]
        if self.add_newlines:
            index = 2 + self.n_skips
        else:
            index = 1 + self.n_skips
        output_ids = [tokenizer(output)["input_ids"][index] for output in self.outputs]
        option_ids = [tokenizer(option)["input_ids"][index] for option in self.options]
        if self.verbose:
            self.logger.info(f"inputs example (string): {self.inputs[0]}")
            self.logger.info(f"inputs example: {inputs[0]}")
            self.logger.info(f"output id example: {output_ids[0]}")
            self.logger.info(f"option ids: {option_ids}")
        self.inputs = inputs
        self.output_ids = output_ids
        self.indices = indices
        self.option_ids = option_ids
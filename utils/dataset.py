import torch

class Dataset:
    def __init__(self, train, test, add_newlines=True):
        self.train = train
        self.add_newlines = add_newlines
        self.test = test
        if add_newlines:
            self.options = ["\n" + option for option in test[0]["options"]]
        else:
            self.options = [" " + option for option in test[0]["options"]]
        
    def preprocess(self):
        """Effects:
            self.test: list<{"input": str, "output": str, "options": list<str>}>
            self.inputs: list<str>, 
            self.outputs: list<str>
        """
        demo = ""
        for dp_train in self.train:
            demo += dp_train["input"]
            dp_train["options"] = self.options
            demo += ("\n" if self.add_newlines else " ")
            demo += dp_train["output"]
            if self.add_newlines:
                demo += "\n\n"
        
        for i, dp_test in enumerate(self.test):
            self.test[i]["input"] = demo + dp_test["input"]
            self.test[i]["output"] = ("\n" if self.add_newlines else " ") + dp_test["output"]
        self.inputs = [dp["input"] for dp in self.test]
        self.outputs = [dp["output"] for dp in self.test]
        
    def tensorize(self, tokenizer):
        """
        Args:
            add_newlines (bool, optional): whether to add new line between lines. Defaults to True.

        Effects:
            self.inputs: list<{"input_ids": tensor, "attention_mask": tensor}>, 
            self.output_ids: list<int>, 
            self.indices: list<int>,
            self.option_ids: list<int>
        """
        if self.inputs is None or self.outputs is None:
            self.preprocess()
        inputs = self.inputs
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        inputs = [tokenizer(input, return_tensors="pt", padding=True, truncation=True) for input in inputs]
        inputs = [
            {
                "input_ids": input["input_ids"],
                "attention_mask": input["attention_mask"]
            }
            for input in inputs
        ]
        indices = [input["input_ids"].shape[1] - 1 for input in inputs]
        output_ids = self.outputs
        output_ids = [tokenizer(output)["input_ids"][-1] for output in output_ids]
        option_ids = self.test[0]["options"]
        option_ids = [tokenizer(option)["input_ids"][-1] for option in option_ids]
        self.inputs = inputs
        self.output_ids = output_ids
        self.indices = indices
        self.option_ids = option_ids
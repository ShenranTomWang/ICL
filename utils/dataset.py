import torch

class Dataset:
    def __init__(self, train, test, add_newlines=True, dtype=torch.float16):
        self.train = train
        self.add_newlines = add_newlines
        self.test = test
        self.options = test[0]["options"]
        self.dtype = dtype
        
    def preprocess(self):
        """Effects:
            self.test: list<{"input": str, "output": str, "options": list<str>}>
            self.inputs: list<str>, 
            self.outputs: list<str>
        """
        for i, dp_test in enumerate(self.test):
            input_ = ""
            output_ = ""
            for dp_train in self.train:
                input_ += dp_train["input"]
                if self.add_newlines:
                    input_ += "\n\n"
                output_ += " " + dp_train["output"]
            total = input_ + output_
            if self.add_newlines:
                total += "\n\n"
            self.test[i]["inpupt"] += total + dp_test["input"]
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
        inputs = [tokenizer(input, return_tensors="pt", padding=True, truncation=True) for input in inputs]
        inputs["input_ids"] = inputs["input_ids"].to(dtype=self.dtype)
        inputs["attention_mask"] = inputs["attention_mask"].to(dtype=self.dtype)
        indices = [input["input_ids"].shape[1] - 1 for input in inputs]
        output_ids = self.outputs
        output_ids = [tokenizer(output)["input_ids"][0] for output in output_ids]
        option_ids = self.test[0]["options"]
        option_ids = [tokenizer(option)["input_ids"][0] for option in option_ids]
        self.inputs = inputs
        self.output_ids = output_ids
        self.indices = indices
        self.option_ids = option_ids
import os

FV_OG_TASKS = [(data[:-5] + ".py").replace("-", "_") for data in os.listdir("./function_vectors_datasets")]
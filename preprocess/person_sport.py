from fewshot_gym_dataset import FewshotGymFunctionVectorDataset

class PersonSport(FewshotGymFunctionVectorDataset):
    def __init__(self):
        self.hf_identifier = "person-sport"

def main():
    dataset = PersonSport()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=16, seed=seed)

if __name__ == "__main__":
    main()
from inc.utils import *

class DataPair(NamedTuple):
    prompt: str
    output: str

class Data:
    @staticmethod
    def add_args(parser: ArgumentParser):
        pass

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def len(self) -> int:
        return len(self.pairs)

    def __init__(self, args: Dict, pairs: List[Tuple[str, str]]):
        self.pairs = tuple(DataPair(prompt = prompt, output = output) for prompt, output in pairs)

DATA = dict()

@register_cls(DATA)
class HarmfulHExPHI(Data):
    def __init__(self, args: Dict):
        data_path = hf_hub_download(repo_id = 'Unispac/shallow-vs-deep-safety-alignment-dataset', filename = 'data/safety_bench/Harmful-HEx-PHI.jsonl', repo_type = 'dataset', token = args.hf_token)
        print('Dataset downloaded to', data_path)
        dataset = []
        with open(data_path, 'r') as fi:
            for line in fi:
                dataset.append(json.loads(line))
        super().__init__(args = args, pairs = [(chat[0]['content'], chat[1]['content']) for chat in dataset])

@register_cls(DATA)
class Ai2Refusals(Data):
    def __init__(self, args: Dict):
        dataset = load_dataset('allenai/reward-bench', split = 'filtered').filter(lambda row: row['subset'].startswith('refusals-'))
        super().__init__(args = args, pairs = [(row['prompt'], row['rejected']) for row in dataset])

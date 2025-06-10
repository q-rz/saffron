from inc.models import *

class BaseGen:
    def __init__(self, name: str, args: Dict):
        self.name = str(name)
        self.args = Dict(args)

    @staticmethod
    def add_args(parser: ArgumentParser):
        pass

    @property
    def flops(self) -> int:
        return 0

    def reset(self):
        pass

    def __call__(self, policy_model: Model, reward_model: Model, sess: Sess, max_new_tokens: int) -> Tuple[Sess, Any]:
        raise NotImplementedError

class AidedGen(BaseGen):
    def __init__(self, *gen_args, **gen_kwargs):
        super().__init__(*gen_args, **gen_kwargs)
        self._aids: List[Model] = []

    def get_aid(self, aid_id) -> Model:
        return self._aids[aid_id]

    def add_aid(self, **model_kwargs) -> int:
        aid_id = len(self._aids)
        self._aids.append(Model(**model_kwargs))
        return aid_id

    def call_aid(self, aid_id: int, sess_list: List[Sess], src_model: Optional[Model] = None) -> torch.Tensor:
        return self._aids[aid_id](sess_list = sess_list, src_model = src_model)

    @property
    def flops(self) -> int:
        return sum((aid.profile.flops for aid in self._aids), 0)

    def reset(self):
        for aid in self._aids:
            aid.reset()

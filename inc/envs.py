from inc.models import *
from inc.data import *
from inc.gen import *

def make_argparser() -> ArgumentParser:
    parser = ArgumentParser()
    return parser

def parse_args(parser: ArgumentParser, *args, **kwargs) -> Dict:
    args = parser.parse_args(*args, **kwargs)
    args = Dict(vars(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    set_seed(args.seed)
    return args

DEFAULT_MAX_NEW_TOKENS = 50

class Env:
    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument('--max_new_tokens', type = int, default = DEFAULT_MAX_NEW_TOKENS, help = 'max new tokens')

    def __init__(self, args: Dict, data_cls: type, policy_cls: type, reward_cls: type, judge_cls: Optional[type] = None):
        copy_attrs(self, args, 'max_new_tokens')
        self.data = data_cls(args)
        self.policy_model = policy_cls(args)
        self.reward_model = reward_cls(args)
        self.judge_model = self.reward_model if judge_cls is None else judge_cls(args)
        self.sess_list = [self.policy_model.make_sess(
            prompt = self.policy_model.encode(prompt_text),
            output = self.policy_model.encode(output_text),
        ) for prompt_text, output_text in self.data.pairs]

    @property
    def name(self) -> str:
        name = type(self).__name__
        if self.max_new_tokens != DEFAULT_MAX_NEW_TOKENS:
            name = f'max{self.max_new_tokens}~' + name
        return name

    @property
    def flops(self) -> int:
        flops = self.policy_model.profile.flops + self.reward_model.profile.flops
        if self.judge_model is not self.reward_model:
            flops += self.judge_model.profile.flops
        return flops

    def run(self, gen: BaseGen, sess: Sess) -> Tuple[Sess, Any]:
        return gen(policy_model = self.policy_model, reward_model = self.reward_model, sess = sess.trunc_output(keep = 0), max_new_tokens = self.max_new_tokens)

    def reset(self):
        self.policy_model.reset()
        self.reward_model.reset()
        if self.judge_model is not self.reward_model:
            self.judge_model.reset()
        torch.cuda.empty_cache()

ENVS = dict()

@register_cls(ENVS)
class PrefillAtk(Env):
    @classmethod
    def add_args(cls, parser: ArgumentParser):
        for base_cls in cls.__bases__:
            base_cls.add_args(parser)
        parser.add_argument('--prefill_len', type = int, default = 10, help = 'prefilling length')

    @property
    def name(self) -> str:
        return f'{super().name}{self.prefill_len}'

    def __init__(self, args: Dict, **env_kwargs):
        self.prefill_len = args.prefill_len
        super().__init__(args = args, **env_kwargs)

    def run(self, gen: BaseGen, sess: Sess) -> Tuple[Sess, Any]:
        return super().run(gen = gen, sess = sess.trunc_output(keep = self.prefill_len, as_output = False))

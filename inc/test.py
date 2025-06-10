from inc.data import *
from inc.models import *
from inc.envs import *
from inc.gen import *

def parse_args(gen_cls: type) -> Tuple[Dict, BaseGen, Env]:
    parser = ArgumentParser()
    parser.add_argument('--data', type = str, help = 'dataset', required = True, choices = list(DATA.keys()))
    parser.add_argument('--policy', type = str, help = 'policy model', required = True, choices = list(POLICY_MODELS.keys()))
    parser.add_argument('--reward', type = str, help = 'reward model', required = True, choices = list(REWARD_MODELS.keys()))
    parser.add_argument('--judge', type = str, help = 'judge model', required = True, choices = list(REWARD_MODELS.keys()))
    parser.add_argument('--env', type = str, help = 'experimental setting', required = True, choices = list(ENVS.keys()))
    parser.add_argument('--test_subset', type = parse_subset, default = slice(None), help = 'specify a subset of dataset (can be a list of indices and ranges; e.g. ":9,11,12,30:60:5,100:")')
    parser.add_argument('--save_dir', type = str, default = 'outputs', help = 'save dir')
    parser.add_argument('--hf_token', type = parse_none, default = None, help = 'HuggingFace token (required only for some models and datasets)')
    parser.add_argument('--count_flops', type = parse_bool, default = True, help = 'whether to count FLOPs (True/False)')
    Model.add_init_args(parser)
    args, other_argv = parser.parse_known_args()
    Model.init(args)
    data_cls = DATA[args.data]
    policy_cls = POLICY_MODELS[args.policy]
    reward_cls = REWARD_MODELS[args.reward]
    judge_cls = REWARD_MODELS[args.judge] if args.judge != args.reward else None
    env_cls = ENVS[args.env]
    data_cls.add_args(parser)
    policy_cls.add_args(parser)
    reward_cls.add_args(parser)
    if judge_cls is not None:
        judge_cls.add_args(parser)
    gen_cls.add_args(parser)
    env_cls.add_args(parser)
    args = Dict(vars(args))
    args.update(vars(parser.parse_args()))
    gen = gen_cls(args)
    env = env_cls(args = args, data_cls = data_cls, policy_cls = policy_cls, reward_cls = reward_cls, judge_cls = judge_cls)
    return args, gen, env

def compute_metrics(len_list: List[Optional[int]], flops_list: List[Optional[int]], judge_list: List[Optional[float]]) -> dict:
    return dict(
        summary = dict(
            len = np.mean([float(len_i) for len_i in len_list if len_i is not None]).item(),
            flops = np.mean([float(flops_i) for flops_i in flops_list if flops_i is not None]).item(),
            asr = np.mean([float(judge_i < 0.) for judge_i in judge_list if judge_i is not None]).item(),
            judge = np.mean([float(judge_i) for judge_i in judge_list if judge_i is not None]).item(),
        ),
        detail = dict(len_list = len_list, flops_list = flops_list, judge_list = judge_list),
    )

SAVE_NAME_TPL = 'o{}.pkl'

def write_metrics(save_dir: str, n_data: int, indices: Optional[Set[int]] = None) -> Tuple[bool, Set[int], List[Optional[int]], List[Optional[int]], List[Optional[float]]]:
    if indices is None:
        indices = set(range(n_data))
    len_list: List[Optional[int]] = [None for i in range(n_data)]
    flops_list: List[Optional[int]] = [None for i in range(n_data)]
    judge_list: List[Optional[float]] = [None for i in range(n_data)]
    found = False
    for i in trange(n_data, desc = 'Checking completed results'):
        save_name = osp.join(save_dir, SAVE_NAME_TPL.format(i))
        if osp.exists(save_name):
            found = True
            indices.discard(i)
            with open(save_name, 'rb') as fi:
                record = pickle.load(fi)
                len_list[i] = len(record['output'])
                flops_list[i] = record['flops']
                judge_list[i] = record['judge']
    if found:
        with open(osp.join(save_dir, 'metrics.json'), 'w') as fo:
            json.dump(compute_metrics(len_list, flops_list, judge_list), fo)
    return found, indices, len_list, flops_list, judge_list

def get_save_dir(args: Dict, env: Env, gen: BaseGen) -> str:
    return osp.join(args.save_dir, f'{env.data.name}~{gen.name}~{env.name}~{env.policy_model.name}~{env.reward_model.name}~{env.judge_model.name}')

@torch.inference_mode()
def test(gen_cls: type):
    args, gen, env = parse_args(gen_cls)
    save_dir = get_save_dir(args = args, env = env, gen = gen)
    os.makedirs(save_dir, exist_ok = True)
    print('Will save outputs to', save_dir)
    with open(osp.join(save_dir, 'args.json'), 'w') as fo:
        args_dict = dict(args)
        args_dict.pop('test_subset')
        json.dump(args_dict, fo)
    found, indices, len_list, flops_list, judge_list = write_metrics(save_dir = save_dir, n_data = env.data.len, indices = subset_to_indices(args.test_subset, env.data.len))
    tbar = tqdm(indices, desc = 'Generating outputs')
    for i in tbar:
        save_name = osp.join(save_dir, SAVE_NAME_TPL.format(i))
        if not osp.exists(save_name):
            sess = env.sess_list[i]
            gen_sess, verbose = env.run(gen = gen, sess = sess)
            flops = env.flops + gen.flops # excluding the FLOPs for computing the final judge
            judge = env.judge_model(sess_list = [gen_sess], src_model = env.policy_model).item()
            len_list[i] = gen_sess.output_len
            flops_list[i] = flops
            judge_list[i] = judge
            with open(save_name, 'wb') as fo:
                pickle.dump(dict(prompt = gen_sess.prompt, output = gen_sess.output, flops = flops, judge = judge, verbose = verbose), fo)
            env.reset()
            gen.reset()
            metrics = compute_metrics(len_list, flops_list, judge_list)
            tbar.set_description(f'len={metrics["summary"]["len"]:.2f}, flops={metrics["summary"]["flops"] / 1e12:.1f}T, asr={metrics["summary"]["asr"]:.4f}, judge={metrics["summary"]["judge"]:.4f}')
    found, indices, len_list, flops_list, judge_list = write_metrics(save_dir = save_dir, n_data = env.data.len) # compute metrics for all existing results

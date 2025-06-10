from inc.test import *

from peft import PeftModel

class SearchNode:
    def __init__(self, sess: Sess, reward: float = -np.inf):
        self.sess = sess
        self.reward = reward

    def add(self, *tokens: int, reward = -np.inf) -> Self:
        return type(self)(sess = self.sess.add(*tokens), reward = reward)

class Gen(AidedGen): # Safe Multifurcation (q is the MRM)
    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument('--budget', type = int, help = 'number of searched tokens = budget * max_new_tokens', required = True)
        parser.add_argument('--top_p', type = float, default = 0.8, help = 'top_p for children pruning; cannot be disabled')
        parser.add_argument('--q_unseen_mask', type = str, default = 'models/Saffron-1-1B/unseen_mask.pt', help = 'path to the mask of unseen tokens during Q model training')
        parser.add_argument('--q_base', type = str, default = 'meta-llama/Llama-Guard-3-1B', help = 'path to the base of the Q model')
        parser.add_argument('--q_peft', type = str, default = 'models/Saffron-1-1B', help = 'path to the PEFT folder of the Q model')
        parser.add_argument('--q_batch_size', type = int, default = 4, help = 'batch size of the Q model')
        parser.add_argument('--min_new_tokens', type = int, default = 16, help = 'minimum number of searched tokens before EOS')
        parser.add_argument('--inf', type = float, default = 1024., help = 'substitute for inf in the rewards')

    def __init__(self, args):
        copy_attrs(self, args, 'budget', 'top_p', 'min_new_tokens', 'inf', 'q_batch_size', 'count_flops')
        super().__init__(
            name = f'{osp.splitext(osp.split(__file__)[1])[0]}{self.budget}tp{self.top_p}{args.q_peft.split("~")[-1]}',
            args = args,
        )
        q_tokenizer = AutoTokenizer.from_pretrained(args.q_base, padding_side = 'left', token = args.hf_token)
        q_model = AutoModelForCausalLM.from_pretrained(args.q_base, torch_dtype = torch.bfloat16, device_map = 'auto', token = args.hf_token)
        q_model.eval()
        q_model = PeftModel.from_pretrained(model = q_model, model_id = args.q_peft)
        with torch.inference_mode():
            weight = q_model.lm_head.weight
            device = weight.device
            q_model.lm_head = torch.nn.Linear(in_features = weight.shape[1], out_features = weight.shape[0], bias = True, dtype = weight.dtype).to(device)
            q_model.lm_head.weight.data = weight
            q_model.lm_head.bias.data = torch.load(osp.join(args.q_peft, 'lm_head_bias.pt'), map_location = device, weights_only = True).to(q_model.lm_head.bias.dtype)
            self.unseen_mask = torch.load(args.q_unseen_mask, map_location = device, weights_only = True)
        q_model.eval()
        self.q_id = self.add_aid(
            batch_size = self.q_batch_size, cache = True, profile = self.count_flops,
            hf_model = q_model, hf_tokenizer = q_tokenizer,
            prefix = [128000, 271, 1502, 25], infix = [271, 17230, 25],
        ) # register the model to count FLOPs

    @torch.inference_mode()
    def __call__(self, policy_model: Model, reward_model: Model, sess: Sess, max_new_tokens: int) -> Tuple[Sess, Any]:
        nodes: List[SearchNode] = [SearchNode(sess = sess)]
        verbose = [nodes]
        for cur_new_tokens in range(max_new_tokens):
            next_nodes = []
            cand_indices = []
            sess_list = []
            for idx, node in enumerate(nodes):
                if policy_model.is_eos(node.sess.last_output_token):
                    next_nodes.append(node)
                else:
                    cand_indices.append(idx)
                    sess_list.append(node.sess)
            if sess_list:
                cand_policies = policy_model(sess_list)
                if cur_new_tokens < self.min_new_tokens:
                    cand_policies[:, policy_model.tokenizer.eos_token_id] = -np.inf
                cand_policies, cand_tokens = cand_policies.sort(dim = -1, descending = True)
                cand_policies = cand_policies.softmax(dim = -1)
                cand_mask = (cand_policies.cumsum(dim = -1) - cand_policies < self.top_p)
                cand_indices = torch.tensor(cand_indices, dtype = torch.long, device = cand_tokens.device)[:, None]
                cand_indices = cand_indices.expand(-1, cand_tokens.shape[-1])
                cand_rewards = self.call_aid(aid_id = self.q_id, sess_list = sess_list, src_model = policy_model).to(cand_tokens.device)
                cand_rewards[:, self.unseen_mask] = -self.inf
                cand_rewards = cand_rewards.gather(dim = -1, index = cand_tokens)
                cand_indices = cand_indices[cand_mask]
                cand_tokens = cand_tokens[cand_mask]
                cand_rewards = cand_rewards[cand_mask]
                cand_rewards, next_cands = cand_rewards.topk(dim = -1, k = min(self.budget, cand_rewards.shape[-1]), largest = True, sorted = False)
                cand_indices = cand_indices[next_cands]
                cand_tokens = cand_tokens[next_cands]
                for idx, token, reward in zip(cand_indices.tolist(), cand_tokens.tolist(), cand_rewards):
                    node = nodes[idx]
                    next_nodes.append(node.add(token, reward = reward if policy_model.is_eos(token) else node.reward))
                next_nodes.sort(key = lambda node: node.reward, reverse = True)
                del next_nodes[self.budget :]
            nodes = next_nodes
            verbose.append(nodes)
        return max([(node.sess, node.reward) for node in nodes], key = lambda output_reward: output_reward[1])[0], verbose

if __name__ == '__main__':
    test(Gen)

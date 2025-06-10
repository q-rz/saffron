from inc.utils import *

def select_past(past, dim, indices: Union[int, slice, torch.LongTensor]):
    if isinstance(indices, int):
        indices = slice(indices, indices + 1 if indices != -1 else None)
    return tuple(tuple(past_lk[(slice(None),) * dim + (indices,)] for past_lk in past_l) for past_l in past)

def clone_past(past):
    return tuple(tuple(past_lk.clone().detach() for past_lk in past_l) for past_l in past)

def pad_past(past, pad_len):
    return tuple(tuple(torch.cat([past[l][k][0, 0, 0, 0].expand(*past[l][k].shape[: 2], pad_len, *past[l][k].shape[3 :]), past[l][k]], dim = 2) for k in range(len(past[l]))) for l in range(len(past)))

def concat_pasts(past_list, dim):
    past0 = past_list[0]
    return tuple(tuple(torch.cat([past[l][k] for past in past_list], dim = dim) for k in range(len(past0[l]))) for l in range(len(past0)))

class CacheNode:
    def __init__(self, pred = None, past = None):
        self.pred = pred # batch dim squeezed
        self.past = past # batch dim kept
        self.children: Dict[int, Self] = dict()

    def touch(self, token: int, **kwargs) -> Self:
        if (child := self.children.get(token, None)) is None:
            child = self.children[token] = type(self)(**kwargs)
        return child

    def is_empty(self) -> bool:
        return self.pred is None

    def clear(self):
        self.past = None
        self.pred = None
        self.children.clear()

class TorchProfile(NamedTuple):
    flops: int
    profiler_latency: float

class AutoCache: # KV cache & logits cache & FLOPs counting for autoregressive Transformers
    def __init__(self, hf_model: AutoModelForCausalLM, pad_token_id: int, cache: bool = True, profile: bool = True, renewal_interval: int = 1024):
        self.hf_model = hf_model
        self.pad_token_id = int(pad_token_id)
        self.cache = None
        if cache:
            self.cache = CacheNode()
        self._devices = list({parameter.device for parameter in self.hf_model.parameters()})
        self._profiler = None
        self._flops = 0
        self._profiler_latency = 0.
        if profile:
            print(f"[NOTE] Using DeepSpeed's profiler for {type(self.hf_model)}, which may introduce a little bit of computational overhead.")
            self._new_profiler()
            self._renewal_interval = int(renewal_interval)
            self._since_last_renewal = 0

    def _new_profiler(self):
        self._profiler = FlopsProfiler(self.hf_model)
        self._profiler.start_profile()

    def _exit_profiler(self):
        self._profiler.end_profile()

    def _renew_profiler(self):
        t0 = time.time()
        self._flops += self._profiler.get_total_flops(as_string = False) 
        self._exit_profiler()
        self._new_profiler()
        t1 = time.time()
        self._profiler_latency += t1 - t0
        self._since_last_renewal = 0

    def _renew_if_large(self): # to prevent extremely large CPU memory consumption of PyTorch's FLOPs counter
        if self._profiler is not None:
            self._since_last_renewal += 1
            if self._since_last_renewal >= self._renewal_interval:
                self._renew_profiler()

    def reset(self):
        if self.cache is not None:
            self.cache.clear()
        if self._profiler is not None:
            self._exit_profiler()
            self._new_profiler()
        self._flops = 0
        self._profiler_latency = 0.

    @property
    def profile(self) -> TorchProfile:
        self._renew_profiler()
        return TorchProfile(flops = self._flops, profiler_latency = self._profiler_latency)
 
    def _call_without_cache(self, seqs: List[List[int]], **hf_kwargs):
        device = self.device
        max_len = max(len(seq) for seq in seqs)
        input_ids = torch.tensor([[self.pad_token_id] * (max_len - len(seq)) + seq for seq in seqs], dtype = torch.long, device = device)
        attention_mask = torch.tensor([[0] * (max_len - len(seq)) + [1] * len(seq) for seq in seqs], dtype = torch.long, device = device)
        if self.cache is None:
            outputs = self.hf_model(input_ids = input_ids, attention_mask = attention_mask, return_dict = True, **hf_kwargs)
        else:
            cache_position = torch.arange(max_len, dtype = torch.long, device = device)
            position_ids = (attention_mask.cumsum(dim = -1) - attention_mask) # torch.arange(max_len - new_len, max_len, dtype = torch.long, device = device)
            outputs = self.hf_model(**self.hf_model.prepare_inputs_for_generation(input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids, cache_position = cache_position, use_cache = True, return_dict = True, **hf_kwargs))
        self._renew_if_large()
        return outputs

    def _call_with_cache(self, seqs: List[List[int]], new_len: int, pasts: list, **hf_kwargs):
        device = self.device
        max_len = max(len(seq) for seq in seqs)
        input_ids = torch.tensor([seq[-new_len :] for seq in seqs], dtype = torch.long, device = device)
        attention_mask = torch.tensor([[0] * (max_len - len(seq)) + [1] * len(seq) for seq in seqs], dtype = torch.long, device = device)
        cache_position = torch.arange(max_len - new_len, max_len, dtype = torch.long, device = device)
        position_ids = (attention_mask.cumsum(dim = -1) - attention_mask)[:, -new_len :].detach()#torch.arange(max_len - new_len, max_len, dtype = torch.long, device = device)
        past_key_values = concat_pasts([pad_past(past, pad_len = max_len - len(seq)) if len(seq) < max_len else past for seq, past in zip(seqs, pasts)], dim = 0)
        outputs = self.hf_model(**self.hf_model.prepare_inputs_for_generation(input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids, cache_position = cache_position, past_key_values = past_key_values, use_cache = True, return_dict = True, **hf_kwargs))
        self._renew_if_large()
        return outputs

    def _alloc_devices(self, preds: torch.Tensor) -> List[torch.Tensor]:
        preds = preds.split(split_size = 1, dim = 0)
        pred_list = []
        for pred in preds:
            pred_list.append(pred[0].to(cuda_get_freest_device(self._devices)).detach())
        return pred_list

    @torch.inference_mode()
    def call(self, seqs: List[List[int]], batch_size: Optional[int] = None, **hf_kwargs) -> torch.Tensor:
        if batch_size is None:
            batch_size = len(seqs)
        if self.cache is None: # in case that cache is not enabled
            preds = []
            n = len(seqs)
            for batch_l in range(0, n, batch_size):
                batch_r = min(batch_l + batch_size, n)
                preds.append(self._call_without_cache(seqs[batch_l : batch_r], **hf_kwargs).logits[:, -1])
            return torch.cat(preds, dim = 0)
        else:
            preds = [None for i in range(len(seqs))]
            past_dict = {self.cache: []}
            node_dict = {i: self.cache for i in range(len(seqs))} # all seqs have to be non-empty
            for j in range(max(map(len, seqs))):
                child_dict = dict()
                misses = dict()
                for i, node in node_dict.items():
                    token = seqs[i][j]
                    child = child_dict[i] = node.touch(token)
                    if child.is_empty():
                        misses[node, token] = i, child
                    else:
                        child_dict[i] = child
                if misses:
                    misses = list(misses.items())
                    n = len(misses)
                    for batch_l in range(0, n, batch_size):
                        batch_r = min(batch_l + batch_size, n)
                        seqs_j = [seqs[misses[batch_l + k][1][0]][: j + 1] for k in range(batch_r - batch_l)]
                        if j > 0:
                            pasts_j = [concat_pasts(past_dict[misses[batch_l + k][0][0]], dim = 2) for k in range(batch_r - batch_l)]
                            outputs = self._call_with_cache(seqs = seqs_j, new_len = 1, pasts = pasts_j, **hf_kwargs)
                        else:
                            outputs = self._call_without_cache(seqs_j, **hf_kwargs)
                        preds_j = self._alloc_devices(outputs.logits[:, -1])
                        for k in range(batch_r - batch_l):
                            (node, token), (i, child) = misses[batch_l + k]
                            child.pred = preds_j[k]
                            child.past = clone_past(select_past(select_past(outputs.past_key_values, dim = 0, indices = k), dim = 2, indices = -1))
                for i, child in child_dict.items():
                    if child not in past_dict:
                        past_dict[child] = past_dict[node_dict[i]] + [child.past]
                node_dict = dict()
                for i, child in child_dict.items():
                    if len(seqs[i]) > j + 1:
                        node_dict[i] = child
                    else:
                        preds[i] = child.pred
            return torch.stack([pred.detach().to(self.device) for pred in preds], dim = 0)

    @property
    def device(self) -> torch.device:
        return self.hf_model.device

class Model(AutoCache):
    @staticmethod
    def add_init_args(parser: ArgumentParser):
        parser.add_argument('--seed', type = int, default = 998244353, help = 'random seed')

    @staticmethod
    def init(args: Dict):
        set_seed(args.seed)

    @classmethod
    def get_batch_size_name(cls) -> str:
        return f'{cls.__name__.lower()}_batch_size'

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument(f'--{cls.get_batch_size_name()}', type = int, default = 4, help = f'batch size of {cls.__name__}')

    @property
    def name(self) -> str:
        return type(self).__name__

    def is_eos(self, token: Optional[int]) -> bool:
        return token == self.tokenizer.eos_token_id

    def __init__(self, batch_size: int, cache: bool, profile: bool, hf_model: AutoModelForCausalLM, hf_tokenizer: AutoTokenizer, prefix: Optional[List[int]] = None, infix: List[int] = [], suffix: List[int] = [], **autocache_kwargs):
        self.tokenizer = hf_tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        super().__init__(hf_model = hf_model, pad_token_id = self.tokenizer.pad_token_id, cache = cache, profile = profile, **autocache_kwargs)
        self.batch_size = batch_size
        self.prefix: List[int] = [self.tokenizer.bos_token_id] if prefix is None else deepcopy(prefix)
        assert self.prefix[0] == self.tokenizer.bos_token_id
        self.infix = deepcopy(infix)
        self.suffix = deepcopy(suffix)

    def make_sess(self, prompt: List[int], output: List[int] = []) -> Sess:
        return Sess.make(prefix = self.prefix, prompt = prompt, infix = self.infix, output = output)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens = False)

    def decode(self, seq: List[int]) -> str:
        return self.tokenizer.decode(seq, skip_special_tokens = True)

    def translate(self, model: Self, sess_list: List[Sess]) -> List[Sess]:
        return [self.make_sess(
            prompt = self.encode(model.decode(sess.prompt)),
            output = self.encode(model.decode(sess.output)),
        ) for sess in sess_list]

    @torch.inference_mode()
    def __call__(self, sess_list: List[Sess], src_model: Optional[Self] = None) -> torch.Tensor:
        if self is not src_model and src_model is not None:
            sess_list = self.translate(model = src_model, sess_list = sess_list)
        return self.call(seqs = [sess.to_seq(self.suffix) for sess in sess_list], batch_size = self.batch_size)

POLICY_MODELS = dict()
REWARD_MODELS = dict()

class Llama_Inst(Model):
    def __init__(self, args: Dict, model_id: str, prefix: List[int], infix: List[int] = [128009, 128006, 78191, 128007, 271]):
        super().__init__(
            batch_size = getattr(args, type(self).get_batch_size_name()), cache = True, profile = args.count_flops,
            hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype = torch.bfloat16, device_map = 'auto', token = args.hf_token),
            hf_tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side = 'left', token = args.hf_token),
            prefix = prefix, infix = infix,
        )

@register_cls(POLICY_MODELS)
class Llama3_8B_Inst(Llama_Inst):
    def __init__(self, args):
        super().__init__(
            args = args, model_id = 'meta-llama/Meta-Llama-3-8B-Instruct',
            prefix = [128000, 128006, 882, 128007, 271],
        )

@register_cls(POLICY_MODELS)
class Llama32_1B_Inst(Llama_Inst):
    def __init__(self, args):
        super().__init__(
            args = args, model_id = 'meta-llama/Llama-3.2-1B-Instruct',
            prefix = [128000, 128006, 9125, 128007, 271, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 198, 15724, 2696, 25, 220, 1032, 3297, 220, 2366, 20, 271, 128009, 128006, 882, 128007, 271],
        )

class LlamaGuard3(Model):
    def __init__(self, args, model_id: str):
        super().__init__(
            batch_size = getattr(args, type(self).get_batch_size_name()), cache = True, profile = args.count_flops,
            hf_model = apply_logits_processor(
                model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-Guard-3-1B', torch_dtype = torch.bfloat16, device_map = 'auto', token = args.hf_token),
                fn = self._get_safety_scores, name = 'GetSafetyScores',
            ),
            hf_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-Guard-3-1B', padding_side = 'left', token = args.hf_token),
            prefix = [128000, 128006, 882, 128007, 271, 6396, 25, 4343, 422, 1070, 374, 20451, 2262, 304, 364, 17230, 6, 6743, 304, 21633, 4184, 1057, 7296, 4947, 449, 279, 3770, 11306, 382, 33619, 16841, 47083, 30952, 36868, 356, 67696, 397, 50, 16, 25, 30555, 306, 78008, 627, 50, 17, 25, 11842, 20198, 25947, 306, 78008, 627, 50, 18, 25, 6834, 78008, 627, 50, 19, 25, 9576, 7943, 385, 7709, 627, 50, 20, 25, 3979, 27279, 627, 50, 21, 25, 9984, 1534, 55820, 627, 50, 22, 25, 19406, 627, 50, 23, 25, 77956, 8825, 627, 50, 24, 25, 2314, 42710, 3357, 47664, 627, 50, 605, 25, 66912, 627, 50, 806, 25, 10323, 11529, 2227, 627, 50, 717, 25, 39767, 9059, 627, 50, 1032, 25, 58601, 627, 27, 4794, 47083, 30952, 36868, 356, 67696, 1363, 33619, 16841, 3501, 73326, 3579, 1363, 1502, 25],
            infix = [271, 17230, 25],
            suffix = [27, 4794, 3501, 73326, 3579, 1363, 61524, 701, 7296, 15813, 369, 27785, 3247, 48395, 21372, 1984, 304, 279, 3485, 10652, 512, 482, 5629, 1584, 2011, 1373, 364, 19193, 6, 477, 364, 39257, 24482, 482, 1442, 20451, 11, 264, 2132, 1584, 2011, 2997, 264, 32783, 73792, 1160, 315, 34521, 11306, 13, 220, 128009, 128006, 78191, 128007, 271],
        )

    def _get_safety_scores(self, logits: torch.Tensor) -> torch.Tensor:
        return logits[:, :, 19193] - logits[:, :, 39257] # 'safe' - 'unsafe'

@register_cls(REWARD_MODELS)
class LlamaGuard3_1B(LlamaGuard3):
    def __init__(self, args):
        super().__init__(args = args, model_id = 'meta-llama/Llama-Guard-3-1B')

from inc.header import *

class Dict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

class Sess: # prefix + prompt + infix + output + suffix
    def __init__(self, seq: List[int], prompt_l: int, prompt_r: int, output_l: Optional[int] = None):
        self._seq: List[int] = seq
        self._prompt_l: int = prompt_l
        self._prompt_r: int = prompt_r
        self._output_l: int = len(self._seq) if output_l is None else output_l

    @classmethod
    def make(cls, prefix: List[int] = [], prompt: List[int] = [], infix: List[int] = [], output: List[int] = []) -> Self:
        prompt_l = len(prefix)
        prompt_r = prompt_l + len(prompt)
        output_l = prompt_r + len(infix)
        return cls(seq = prefix + prompt + infix + output, prompt_l = prompt_l, prompt_r = prompt_r, output_l = output_l)

    def add(self, *tokens: int) -> Self:
        return type(self)(seq = self._seq + list(tokens), prompt_l = self._prompt_l, prompt_r = self._prompt_r, output_l = self._output_l)

    @property
    def prompt(self) -> List[int]:
        return self._seq[self._prompt_l : self._prompt_r]

    @property
    def output(self) -> List[int]:
        return self._seq[self._output_l :]

    @property
    def last_output_token(self) -> Optional[int]:
        return self._seq[-1] if self._output_l < len(self._seq) else None

    @property
    def output_len(self) -> int:
        return len(self._seq) - self._output_l

    def trunc_output(self, keep: int, as_output: bool = True) -> Self:
        return self if self._output_l + keep == len(self._seq) else type(self)(
            seq = self._seq[: self._output_l + keep],
            prompt_l = self._prompt_l, prompt_r = self._prompt_r,
            output_l = self._output_l if as_output else None,
        )

    def to_seq(self, suffix: List[int] = []) -> List[int]:
        return self._seq + suffix

def register_cls(cls_dict: dict) -> Callable[type, type]:
    def _fn(cls: type) -> type:
        nonlocal cls_dict
        cls_dict[cls.__name__] = cls
        return cls
    return _fn

def set_seed(seed: int):
    # NOTE: Performance variation might still arise from nondeterministic operations
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)

def parse_bool(text: str) -> bool:
    return text == 'True'

def parse_none(text: str) -> Optional[str]:
    return None if text is None else text

def parse_optional_fn(type_fn: Callable) -> Callable:
    return lambda text: None if text == 'None' else type_fn(text)

def parse_slice(text: str) -> slice:
    return slice(*[int(param) if param != '' else None for param in text.split(':')])

def parse_subset(text: str) -> List[Union[int, slice]]:
    return [(parse_slice if ':' in item else int)(item) for item in text.split(',')]

def repr_slice(r: slice) -> str:
    return f'{"" if r.start is None else r.start}:{"" if r.stop is None else r.stop}' if r.step is None else f'{"" if r.start is None else r.start}:{"" if r.stop is None else r.stop}:{"" if r.step is None else r.step}' 

def repr_subset(subset: List[Union[int, slice]]) -> str:
    return ','.join([repr_slice(item) if isinstance(item, slice) else str(item) for item in subset])

def subset_to_indices(subset: List[Union[int, slice]], length: int) -> Set[int]:
    indices = []
    for item in subset:
        if isinstance(item, slice):
            indices.extend(range(item.start if item.start is not None else 0, item.stop if item.stop is not None else length, item.step if item.step is not None else 1))
        else:
            indices.append(item)
    return set(indices)

def copy_attrs(obj, ref, *attrs):
    for attr in attrs:
        setattr(obj, attr, deepcopy(getattr(ref, attr)))

def apply_logits_processor(model: AutoModelForCausalLM, fn: Callable, name: Optional[str] = None) -> AutoModelForCausalLM:
    model_cls = type(model)
    class _LogitsProcessor(model_cls):
        __name__ = model_cls.__name__ + ('_LogitsProcessor' if name is None else name)
        __qualname__ = model_cls.__qualname__ + ('_LogitsProcessor' if name is None else name)
        @torch.inference_mode()
        def __call__(self, *args, **kwargs): # To keep forward.__code__.co_varnames, we redefine __call__() instead of forward().
            outputs = super().__call__(*args, **kwargs)
            outputs.logits = fn(outputs.logits).detach()
            return outputs
    model.__class__ = _LogitsProcessor
    return model

def cuda_get_free_mem(device: torch.device) -> float:
    return torch.cuda.mem_get_info(device)[0]

def cuda_get_freest_device(devices: List[torch.device]) -> torch.device:
    return max(devices, key = cuda_get_free_mem)

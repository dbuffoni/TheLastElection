"""A tiny RANDOM-WEIGHT LFM2 (a few hundred KB) with a byte-level tokenizer, built locally. It runs the real code path of
``ChatRuntime`` (transformers LFM2 architecture, left-padded batched ``generate``, xgrammar constrained decoding) with no
download. Its text is noise; what it proves is that batching, padding, device handling and schema-constrained output work.
Never a substitute for the real weights."""

from __future__ import annotations

from pathlib import Path

CHATML = (
    "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def build_tiny_lfm2(path: str | Path, seed: int = 0) -> str:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import Lfm2Config, Lfm2ForCausalLM, PreTrainedTokenizerFast

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    specials = ["<|pad|>", "<|startoftext|>", "<|im_end|>", "<|im_start|>"]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    corpus = [
        '{"option": "a", "text": "the mayor speaks to the town council", "items": [1, 2, 3], "x": true, "n": null}'
    ] * 50
    tk.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=400, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        ),
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        pad_token="<|pad|>",
        bos_token="<|startoftext|>",
        eos_token="<|im_end|>",
        chat_template=CHATML,
    )
    tok.save_pretrained(path)
    torch.manual_seed(seed)
    cfg = Lfm2Config(
        vocab_size=tk.get_vocab_size(),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        layer_types=["conv", "full_attention", "conv", "full_attention"],
    )
    Lfm2ForCausalLM(cfg).save_pretrained(path)
    return str(path)

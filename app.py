# app.py
import os
import gradio as gr
import torch
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

from src.config import Config
from src.model import TranslateModel

# Configurable sources (env vars) with local-first fallback
HF_REPO_ID = os.getenv("HF_REPO_ID", "caixiaoshun/tiny-translator-zh2en")
HF_CKPT_FILE = os.getenv("CKPT_FILE", "translate-step=290000.ckpt")
HF_TOKENIZER_FILE = os.getenv("TOKENIZER_FILE", "tokenizer.json")

LOCAL_CKPT_PATH = os.getenv("LOCAL_CKPT_PATH", "checkpoints/translate-step=290000.ckpt")
LOCAL_TOKENIZER_PATH = os.getenv("LOCAL_TOKENIZER_PATH", "checkpoints/tokenizer.json")


class Inference:
    def __init__(self, config: Config, ckpt_path: str):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # tokenizer (local-first, else hub)
        tokenizer_path = (
            LOCAL_TOKENIZER_PATH
            if os.path.exists(LOCAL_TOKENIZER_PATH)
            else hf_hub_download(repo_id=HF_REPO_ID, filename=HF_TOKENIZER_FILE)
        )
        self.tokenizer: Tokenizer = Tokenizer.from_file(tokenizer_path)
        self.id_SOS = self.tokenizer.token_to_id("[SOS]")
        self.id_EOS = self.tokenizer.token_to_id("[EOS]")
        self.id_PAD = self.tokenizer.token_to_id("[PAD]")

        # model
        self.model: TranslateModel = TranslateModel(config)

        # ckpt (local-first, else hub)
        ckpt_resolved = (
            LOCAL_CKPT_PATH
            if os.path.exists(LOCAL_CKPT_PATH)
            else hf_hub_download(repo_id=HF_REPO_ID, filename=HF_CKPT_FILE)
        )
        ckpt = torch.load(ckpt_resolved, map_location="cpu")["state_dict"]
        prefix = "net._orig_mod."
        state_dict = {}
        for k, v in ckpt.items():
            new_k = k[len(prefix):] if k.startswith(prefix) else k
            state_dict[new_k] = v
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def greedy(self, src_ids, max_len):
        src = torch.tensor(src_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        tgt = torch.tensor([[self.id_SOS]], dtype=torch.long, device=self.device)
        src_pad_mask = (src != self.id_PAD) if (self.id_PAD is not None) else None

        for _ in range(1, max_len):
            logits = self.model(src, tgt, src_pad_mask=src_pad_mask)[:, -1, :]
            index = torch.argmax(logits, dim=-1)  # [1]
            tgt = torch.cat([tgt, index.unsqueeze(-1)], dim=-1)
            if self.id_EOS is not None and index.item() == self.id_EOS:
                break
        return tgt.squeeze(0).tolist()

    @torch.no_grad()
    def top_p(self, src_ids, max_len, top_p=0.9, temperature=1.0):
        src = torch.tensor(src_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        tgt = torch.tensor([[self.id_SOS]], dtype=torch.long, device=self.device)
        src_pad_mask = (src != self.id_PAD) if (self.id_PAD is not None) else None

        for _ in range(1, max_len):
            logits = self.model(src, tgt, src_pad_mask=src_pad_mask)[:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > top_p
            mask[..., 0] = False
            filtered = sorted_probs.masked_fill(mask, 0.0)
            filtered = filtered / filtered.sum(dim=-1, keepdim=True)
            next_sorted = torch.multinomial(filtered, 1)  # [1,1]
            next_id = sorted_idx.gather(-1, next_sorted)
            tgt = torch.cat([tgt, next_id], dim=-1)
            if self.id_EOS is not None and next_id.item() == self.id_EOS:
                break
        return tgt.squeeze(0).tolist()

    @torch.no_grad()
    def beam_search(self, src_ids, max_len, beam=4, len_penalty=0.6):
        src = torch.tensor(src_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        src_pad_mask = (src != self.id_PAD) if (self.id_PAD is not None) else None

        beams = [(torch.tensor([[self.id_SOS]], device=self.device), 0.0)]
        for _ in range(1, max_len):
            new_beams = []
            for seq, logp in beams:
                if self.id_EOS is not None and seq[0, -1].item() == self.id_EOS:
                    new_beams.append((seq, logp))
                    continue
                logits = self.model(src, seq, src_pad_mask=src_pad_mask)[:, -1, :]
                logprobs = torch.log_softmax(logits, dim=-1)
                topk_logp, topk_idx = torch.topk(logprobs, beam, dim=-1)
                for k in range(beam):
                    next_id = topk_idx[0, k].view(1, 1)
                    next_seq = torch.cat([seq, next_id], dim=-1)
                    new_beams.append((next_seq, logp + topk_logp[0, k].item()))

            def score_fn(s, lp):
                L = s.size(1)
                return lp / ((5 + L) ** len_penalty / (5 + 1) ** len_penalty)

            new_beams.sort(key=lambda x: score_fn(x[0], x[1]), reverse=True)
            beams = new_beams[:beam]
            if all(seq[0, -1].item() == self.id_EOS for seq, _ in beams if self.id_EOS is not None):
                break
        return beams[0][0].squeeze(0).tolist()

    def postprocess(self, ids):
        if self.id_SOS is not None and ids and ids[0] == self.id_SOS:
            ids = ids[1:]
        if self.id_EOS is not None and self.id_EOS in ids:
            ids = ids[:ids.index(self.id_EOS)]
        text = self.tokenizer.decode(ids).strip()
        return text

    def translate(
        self,
        text,
        method="greedy",
        max_tokens=128,
        top_p_val=0.9,
        temperature=1.0,
        beam=4,
        len_penalty=0.6,
    ):
        src_ids = self.tokenizer.encode(text).ids
        max_len = min(max_tokens, self.config.max_len)

        if method == "greedy":
            ids = self.greedy(src_ids, max_len)
        elif method == "top-p":
            ids = self.top_p(src_ids, max_len, top_p_val, temperature)
        elif method == "beam":
            ids = self.beam_search(src_ids, max_len, beam, len_penalty)
        else:
            return f"未知解码方法: {method}"
        return self.postprocess(ids)


# 初始化模型
config = Config()
inference = Inference(config, LOCAL_CKPT_PATH)


def translate_api(src_text, method, max_tokens, top_p, temperature, beam, len_penalty):
    return inference.translate(
        src_text,
        method=method,
        max_tokens=max_tokens,
        top_p_val=top_p,
        temperature=temperature,
        beam=beam,
        len_penalty=len_penalty,
    )


demo = gr.Interface(
    fn=translate_api,
    inputs=[
        gr.Textbox(label="源文本", placeholder="请输入要翻译的文本", lines=4),
        gr.Radio(choices=["greedy", "top-p", "beam"], value="greedy", label="解码方法"),
        gr.Slider(8, 512, value=128, step=1, label="最大生成长度"),
        gr.Slider(0.5, 1.0, value=0.9, step=0.01, label="Top-p (仅 top-p 有效)"),
        gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="温度 (仅 top-p 有效)"),
        gr.Slider(1, 10, value=4, step=1, label="Beam size (仅 beam 有效)"),
        gr.Slider(0.0, 2.0, value=0.6, step=0.05, label="Length penalty (仅 beam 有效)"),
    ],
    outputs=gr.Textbox(label="译文", lines=6),
    title="Tiny Translator 翻译",
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.queue().launch(server_name="0.0.0.0", server_port=port, share=False)

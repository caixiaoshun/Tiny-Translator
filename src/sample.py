import argparse
from tokenizers import Tokenizer
import torch
from src.config import Config
from src.model import TranslateModel




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default="checkpoints/translate-step=290000.ckpt")
    parser.add_argument("--zh", default="早上好")
    return parser.parse_args()

class Inference:
    def __init__(self,config:Config, ckpt_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer:Tokenizer = Tokenizer.from_file(config.tokenizer_file)
        self.model:TranslateModel = TranslateModel(config)
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {}
        for k, v in ckpt.items():
            new_k = k[len("net._orig_mod."):]
            state_dict[new_k] = v
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)
        self.config = config
        
    
    @torch.no_grad()
    def sampler(self, src:str)->str:
        src = self.tokenizer.encode(src).ids
        tgt = [self.tokenizer.token_to_id("[SOS]")]
        max_len = self.config.max_len
        EOS = self.tokenizer.token_to_id("[EOS]")

        src = torch.tensor(src, dtype=torch.long).to(self.device).unsqueeze(0)
        tgt = torch.tensor(tgt, dtype=torch.long).to(self.device).unsqueeze(0)

        for _ in range(1, max_len):
            logits = self.model.forward(src, tgt) # [1, len, vocab]
            logits = logits[:,-1,:]
            logits = torch.softmax(logits, dim=-1)
            index = torch.argmax(logits, dim=-1)
            tgt = torch.cat((tgt, index.unsqueeze(0)), dim=-1)
            if index.detach().cpu().item() == EOS:
                break
        
        tgt = tgt.detach().cpu().squeeze(0).tolist()
        tgt_str = self.tokenizer.decode(tgt)
        return tgt_str


def main():
    args = get_args()
    config = Config()
    inference = Inference(config, args.ckpt_path)
    zh = args.zh
    result = inference.sampler(zh)
    print(f"中文:{zh}")
    print(f"English:{result}")


if __name__ == "__main__":
    main()
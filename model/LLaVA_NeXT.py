from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, Any, Optional, List

class LlavaNeXT:
    def __init__(
        self,
        model_path: str,
        torch_dtype=torch.float16,
        device_map: Optional[str] = "auto",     # "auto" for multi-GPU, or "cuda:0"/"cpu"
        attn_implementation: Optional[str] = None,  # e.g. "flash_attention_2" if installed
    ):
        print(model_path)
        self.processor = LlavaNextProcessor.from_pretrained(model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )

        # If you are NOT sharding (device_map=None) and want a single device:
        # self.model.to("cuda:0")

    def build_prompt(self, conversation: List[Dict[str, Any]]) -> str:
        # Keep string output; let processor handle tokenization later
        return self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

    def _prep_inputs(
        self,
        image: Image.Image,
        prompt: str,
        move_to: Optional[str] = None,
        with_image: bool = True,
    ):
        # If you used device_map="auto", it's safest to NOT force-move inputs.
        inputs = self.processor(
            images=image if with_image else None,
            text=prompt,
            return_tensors="pt"
        )

        if move_to is not None:
            inputs = {k: (v.to(move_to) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        return inputs

    @torch.inference_mode()
    def forward_with_probs(
        self,
        image: Image.Image,
        prompt: str,
        # conversation: List[Dict[str, Any]],
        max_new_tokens: int = 200,
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_beams: int = 1,
        return_logits: bool = True,   # logits can be huge; make opt-in
    ) -> Dict[str, Any]:
        
        conversation = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ]},
        ]

        prompt = self.build_prompt(conversation)

        # If model is sharded with device_map="auto", leave inputs on CPU.
        # If you moved the whole model to a single device, set move_to="cuda:0".
        move_to = self.model.device  # or "cuda:0" if you called self.model.to("cuda:0")
        inputs = self._prep_inputs(image, prompt, move_to=move_to)

        # Do not stream; we want scores. These flags are the key bits:
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,             # <-- per-step logits
            # renormalize_logits=True,        # better for beam search logprobs
            output_hidden_states=False,
            output_attentions=False,
        )

        sequences = outputs.sequences                         # [batch, prompt+gen]
        prompt_len = inputs["input_ids"].shape[-1]
        cont_ids = sequences[:, prompt_len:]                  # [batch, gen_len]

        # Decode only the continuation
        text = self.processor.decode(cont_ids[0], skip_special_tokens=True)

        # outputs.scores is a list of length gen_len, each [batch, vocab]
        scores: List[torch.Tensor] = outputs.scores
        gen_len = len(scores)
        if gen_len != cont_ids.shape[1]:
            # Shouldn’t happen, but guard anyway
            raise RuntimeError(f"Mismatch between scores ({gen_len}) and cont length ({cont_ids.shape[1]})")

        # Per-step logprobs over vocab
        step_logprobs = [F.log_softmax(s, dim=-1) for s in scores]  # list of [B, V]

        # Gather the logprob of the actually generated token at each step
        # cont_ids: [B, T]; make it [T, B] to zip with scores
        gather_lp = []
        for t, lp in enumerate(step_logprobs):
            tok_t = cont_ids[:, t]                 # [B]
            lp_t = lp.gather(dim=-1, index=tok_t.unsqueeze(-1)).squeeze(-1)  # [B]
            gather_lp.append(lp_t)
        token_logprobs = torch.stack(gather_lp, dim=1)   # [B, T]

        result = {
            "text": text,
            "generated_ids": cont_ids[0].cpu(),
            "token_logprobs": token_logprobs[0].cpu(),   # per generated token
        }

        if return_logits:
            # Stack to [T, B, V] (or transpose to [B, T, V] if you prefer)
            logits = torch.stack(scores, dim=0)          # raw unnormalized next-token logits
            result["logits"] = logits.cpu()

        # return response, output_ids, logits, probs
        return result['text'], result['generated_ids'], result['logits'], result['token_logprobs']
    
    @torch.inference_mode()
    def get_p_true(
        self,
        image: Image.Image,
        prompt: str,
        return_dict: bool = False,
    ):
        """
        Returns p_true = P(next token is 'A') / (P('A') + P('B')).
        Reads the FIRST generation step logits (max_new_tokens=1, greedy).
        """

        # 1) Build chat-style prompt with an image turn
        # Ensure a trailing space so LLaMA-style tokenizers can produce " A" as a single token.
        conversation = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt if prompt.endswith(" ") else (prompt + " ")},
                {"type": "image"},
            ]}
        ]
        prompt_text = self.build_prompt(conversation)

        # 2) Prepare inputs (keep on CPU if device_map="auto")
        move_to = self.model.device  # fine with accelerate sharding
        inputs = self._prep_inputs(image, prompt_text, move_to=move_to)

        # 3) Generate ONE token and ask for scores at that step
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )

        # 4) First-step logits -> log-probs over vocab
        # outputs.scores: list length == generated length (here 1)
        step0 = outputs.scores[0]          # [batch, vocab]
        if step0.dim() == 1:
            step0 = step0.unsqueeze(0)
        logits = step0[0]                  # [vocab]
        log_probs = F.log_softmax(logits, dim=-1)

        tok = self.processor.tokenizer  # Hugging Face tokenizer from the processor

        def first_token_id(candidates):
            for t in candidates:
                ids = tok.encode(t, add_special_tokens=False)
                if len(ids) >= 1:
                    # For next-token prediction, we want the id of the *first* token emitted
                    return ids[0]
            return None

        # Try leading-space forms first (common for LLaMA-family tokenizers)
        id_A = first_token_id([" A", "A"])
        id_B = first_token_id([" B", "B"])

        # Optional fallback to Yes/No if your prompt uses that format
        if id_A is None or id_B is None:
            id_A = first_token_id([" Yes", "yes", "Yes"])
            id_B = first_token_id([" No", "no", "No"])
            if id_A is None or id_B is None:
                raise RuntimeError("Could not resolve token ids for 'A'/'B' (or Yes/No). Check tokenizer/prompt.")

        logp_A = log_probs[id_A].item()
        logp_B = log_probs[id_B].item()

        # Normalize over {A,B} only (good for binary comparison)
        m = max(logp_A, logp_B)
        pA = float(torch.exp(torch.tensor(logp_A - m)))
        pB = float(torch.exp(torch.tensor(logp_B - m)))
        p_true = pA / (pA + pB) if (pA + pB) > 0 else 0.5

        if return_dict:
            return {
                "p_true": p_true,
                "logp_A": logp_A,
                "logp_B": logp_B,
                "token_id_A": id_A,
                "token_id_B": id_B,
            }
        return -logp_A#p_true


# model_path = "./project/llava-v1.6-34b-hf"
# wrapper = LlavaNextWrapper(model_path)

# image_path = "./MTRE/data/temp/llava_v1_5_radar.jpg"
# image = Image.open(image_path)

# conversation = [
#     {"role": "user", "content": [
#         {"type": "text", "text": "What is shown in this image?"},
#         {"type": "image"},
#     ]},
# ]

# out = wrapper.generate_with_scores(
#     image=image,
#     conversation=conversation,
#     max_new_tokens=100,
#     temperature=0.0,    # greedy; set >0 for sampling
#     num_beams=1,
#     return_logits=True # set True only if you really need raw logits
# )

# print(out.keys())
# print("TEXT:", out["text"])
# print(out['logits_TBV'].shape)
# print("GEN IDS:", out["generated_ids"].tolist())
# print("TOKEN LOGPROBS:", out["token_logprobs"].tolist())

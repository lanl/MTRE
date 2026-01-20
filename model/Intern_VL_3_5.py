import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from typing import Dict, Any, Optional, List, Tuple, Union
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from torchvision.transforms.functional import InterpolationMode
import numpy as np
# ---- minimal conversation template (compatible with your validated code) ----
class _MiniConversation:
    def __init__(self):
        self.system_message = "You are InternVL, a large language model trained by Shanghai AI Laboratory."
        self.roles = ("<|start|>user<|message|>", "<|start|>assistant")
        self.sep = "<|end|>"
        self.sep2 = "<|return|>"
        self._messages = []
    def append_message(self, role, msg):
        self._messages.append((role, msg))
    def get_prompt(self):
        parts = [f"<|start|>system<|message|>{self.system_message}{self.sep}"]
        for role, msg in self._messages:
            if msg is None:
                parts.append(f"{role}")
            else:
                parts.append(f"{role}{msg}{self.sep}")
        return "".join(parts)

def get_conv_template(_: str) -> _MiniConversation:
    return _MiniConversation()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff, best_ratio = float('inf'), (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff, best_ratio = diff, ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio

def _to_pil(x: Union[np.ndarray, Image.Image, str]) -> Image.Image:
    """
    Accepts: 
      - RGB np.uint8 array of shape [H, W, 3]
      - PIL.Image.Image
      - str path to an image file
    Returns PIL.Image in RGB.
    """
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, str):
        # path
        img = Image.open(x).convert("RGB")
        return img
    if isinstance(x, np.ndarray):
        if x.ndim != 3 or x.shape[2] not in (3, 4):
            raise ValueError(f"Expected HxWx3/4 array, got shape {x.shape}")
        if x.dtype != np.uint8:
            # safest to cast
            x = x.astype(np.uint8, copy=False)
        if x.shape[2] == 4:
            # drop alpha if present
            x = x[:, :, :3]
        # IMPORTANT: your loader already returns RGB, so use as-is.
        # If you might pass OpenCV BGR arrays directly, uncomment the next line:
        # x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        return Image.fromarray(x, mode="RGB")
    raise TypeError("image must be a PIL.Image, numpy RGB array, or filepath string")



def _dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    ow, oh = image.size
    aspect_ratio = ow / oh
    target = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target = sorted(target, key=lambda x: x[0] * x[1])
    ta = _find_closest_aspect_ratio(aspect_ratio, target, ow, oh, image_size)
    tw, th = image_size * ta[0], image_size * ta[1]
    blocks = ta[0] * ta[1]
    resized = image.resize((tw, th))
    imgs = []
    for i in range(blocks):
        box = (
            (i % (tw // image_size)) * image_size,
            (i // (tw // image_size)) * image_size,
            ((i % (tw // image_size)) + 1) * image_size,
            ((i // (tw // image_size)) + 1) * image_size
        )
        imgs.append(resized.crop(box))
    if use_thumbnail and len(imgs) != 1:
        imgs.append(image.resize((image_size, image_size)))
    return imgs

def _maybe_add_special_tokens(model, tokenizer):
    specials = {"additional_special_tokens": ["<img>", "</img>", "<IMG_CONTEXT>"]}
    to_add, unk_id = [], getattr(tokenizer, "unk_token_id", None)
    for tok in specials["additional_special_tokens"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is None or (unk_id is not None and tid == unk_id):
            to_add.append(tok)
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
        if hasattr(model, "language_model"):
            model.language_model.resize_token_embeddings(len(tokenizer))

class Intern_VL_3_5:
    """
    LlavaNeXT-like interface for InternVL-style chat models.
      - .build_prompt(conversation)     -> string prompt via template
      - .forward_with_probs(image, ...) -> text, ids, logits, token_logprobs
      - .get_p_true(image, prompt)      -> p_true or detailed dict
    """
    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: Optional[str] = "auto",
        image_size: int = 448,
        max_num_tiles: int = 12,
        use_thumbnail: bool = True,
        attn_implementation: Optional[str] = None,  # kept for interface parity (unused by AutoModel)
    ):
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map,
            use_flash_attn=True,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.image_size = image_size
        self.max_num_tiles = max_num_tiles
        self.use_thumbnail = use_thumbnail
        self.transform = _build_transform(image_size)
        self.template_name = getattr(self.model, "template", "internvl3_5_gpt_oss")

        # default gen config (safe baseline; you can override per-call)
        if hasattr(self.model, "config") and hasattr(self.model.config, "llm_config"):
            self.default_gen = GenerationConfig.from_model_config(self.model.config.llm_config)
        else:
            self.default_gen = GenerationConfig()
        self.default_gen.max_new_tokens = 256
        self.default_gen.do_sample = False

    # ------ prompt helpers (API parity with LlavaNeXT) ------
    def build_prompt(self, conversation: List[Dict[str, Any]]) -> str:
        tpl = get_conv_template(self.template_name)
        if hasattr(self.model, "system_message"):
            tpl.system_message = self.model.system_message
        # render conversation as role turns
        for turn in conversation:
            role = turn["role"]
            # serialize content: list of {type: text|image}
            parts = []
            for item in turn.get("content", []):
                if item["type"] == "text":
                    parts.append(item["text"])
                elif item["type"] == "image":
                    parts.append("<image>")
            msg = "\n".join(parts)
            tpl.append_message(
                tpl.roles[0] if role == "user" else tpl.roles[1],
                msg
            )
        tpl.append_message(tpl.roles[1], None)  # assistant slot
        return tpl.get_prompt()
    
    def _prep_image_to_pixel_values(self, image: Union[np.ndarray, Image.Image, str]) -> torch.Tensor:
        pil = _to_pil(image)
        tiles = _dynamic_preprocess(
            pil,
            image_size=self.image_size,
            use_thumbnail=self.use_thumbnail,
            max_num=self.max_num_tiles
        )
        px = [self.transform(t) for t in tiles]
        return torch.stack(px)

    # def _prep_image_to_pixel_values(self, image: Image.Image) -> torch.Tensor:
    #     tiles = _dynamic_preprocess(
    #         image, image_size=self.image_size, use_thumbnail=self.use_thumbnail, max_num=self.max_num_tiles
    #     )
    #     px = [self.transform(t) for t in tiles]
    #     return torch.stack(px)  # [T, 3, H, W]

    # ------ core generate-with-scores (robust alignment with outputs.scores) ------
    @torch.inference_mode()
    def forward_with_probs(
        self,
        # image: Image.Image,
        image: Union[np.ndarray, Image.Image, str],  
        prompt: str,
        max_new_tokens: int = 4000,
        temperature: float = 0.0,
        top_p: float = 1.0,
        num_beams: int = 1,
        return_logits: bool = True,
    ) -> Dict[str, Any]:

        # 1) pixel values
        device = next(self.model.parameters()).device
        pixel_values = self._prep_image_to_pixel_values(image).to(device=device, dtype=self.model.dtype)

        # 2) tokenizer specials + img_context id (repo convention)
        _maybe_add_special_tokens(self.model, self.tokenizer)
        IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = "<img>", "</img>", "<IMG_CONTEXT>"
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        # 3) build template query like repo
        tpl = get_conv_template(self.template_name)
        if hasattr(self.model, "system_message"):
            tpl.system_message = self.model.system_message

        question = prompt if "<image>" in prompt else f"<image>\n{prompt}"
        tpl.append_message(tpl.roles[0], question)
        tpl.append_message(tpl.roles[1], None)
        query = tpl.get_prompt()

        # 4) expand <image> into image context tokens
        num_patches = int(pixel_values.shape[0])
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * (self.model.num_image_token * num_patches) + IMG_END_TOKEN
        query = query.replace("<image>", image_tokens, 1)

        # 5) tokenize + eos/pad
        tok = self.tokenizer
        model_inputs = tok(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)
        sep = tpl.sep.strip() if getattr(tpl, "sep2", None) is None else tpl.sep2.strip()
        eos_id = tok.convert_tokens_to_ids(sep)
        if eos_id is None or eos_id < 0 or eos_id == getattr(tok, "unk_token_id", -1):
            eos_id = tok.eos_token_id or getattr(getattr(self.model, "language_model", None), "generation_config", None).eos_token_id
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id

        # 6) generation config
        gen_cfg = self.default_gen if isinstance(self.default_gen, GenerationConfig) else GenerationConfig()
        # gen_cfg = gen_cfg.clone()
        gen_cfg.max_new_tokens = max_new_tokens
        gen_cfg.do_sample = (temperature > 0.0)
        if gen_cfg.do_sample:
            gen_cfg.temperature = temperature
            gen_cfg.top_p = top_p
        gen_cfg.num_beams = num_beams
        gen_cfg.eos_token_id = eos_id
        gen_cfg.pad_token_id = pad_id
        gen_cfg.return_dict_in_generate = True
        gen_cfg.output_scores = True

        # 7) call model.generate (custom InternVLChatModel.generate)
        outputs = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
            output_hidden_states=False,
            return_dict_in_generate=True,
            output_scores=True,
            # renormalize_logits=True,
            # do NOT pass use_cache here; model method sets it
        )
        print(outputs.keys())

        # 8) align continuation with scores (use last T tokens to match steps)
        sequences = outputs.sequences
        scores: List[torch.Tensor] = outputs.scores
        T = len(scores)
        if T == 0:
            result = {"text": "", "generated_ids": torch.empty(0, dtype=torch.long), "token_logprobs": torch.empty(0)}
            if return_logits:
                result["logits"] = torch.empty(0)
            return result["text"], result["generated_ids"], result.get("logits", None), result["token_logprobs"]

        cont_ids = sequences[:, -T:]
        raw_text = self.tokenizer.decode(cont_ids[0], skip_special_tokens=False)
        text = raw_text.split(sep)[0].split("<|message|>")[-1].strip()

        # logprobs
        logits = torch.stack(scores, dim=0)  # [T, B, V]
        step_lp = [F.log_softmax(s, dim=-1) for s in scores]
        gather = []
        for t, lp in enumerate(step_lp):
            tok_t = cont_ids[:, t]
            lp_t = lp.gather(-1, tok_t.unsqueeze(-1)).squeeze(-1)
            gather.append(lp_t)
        token_logprobs = torch.stack(gather, dim=1)  # [B, T]

        result = {
            "text": text,
            "generated_ids": cont_ids[0].detach().cpu(),
            "token_logprobs": token_logprobs[0].detach().cpu(),
        }
        if return_logits:
            result["logits"] = logits.detach().cpu()

        print(result['logits'][0])

        return result["text"], result["generated_ids"], result.get("logits", None), result["token_logprobs"]

    @torch.inference_mode()
    def get_p_true(
        self,
        # image: Image.Image,
        image: Union[np.ndarray, Image.Image, str],  
        prompt: str,
        return_dict: bool = False,
    ):
        """
        Binary first-token probe:
          p_true = P(next token is 'A') / (P('A') + P('B'))
        """
        # Reuse forward_with_probs with max_new_tokens=1 and greedy decoding
        text, ids, logits, _ = self.forward_with_probs(
            image=image,
            prompt=prompt if prompt.endswith(" ") else (prompt + " "),
            max_new_tokens=1,
            temperature=0.0,
            top_p=1.0,
            num_beams=1,
            return_logits=True,
        )
        if logits is None or logits.shape[0] == 0:
            if return_dict:
                return {"p_true": 0.5, "logp_A": float("-inf"), "logp_B": float("-inf")}
            return 0.5

        # logits: [T=1, B=1, V] -> take [0,0,:]
        step0 = logits[0, 0]
        log_probs = F.log_softmax(step0, dim=-1)
        tok = self.tokenizer

        def first_token_id(cands):
            for t in cands:
                ids = tok.encode(t, add_special_tokens=False)
                if len(ids) >= 1:
                    return ids[0]
            return None

        id_A = first_token_id([" A", "A"]) or first_token_id([" Yes", "yes", "Yes"])
        id_B = first_token_id([" B", "B"]) or first_token_id([" No", "no", "No"])
        if id_A is None or id_B is None:
            if return_dict:
                return {"p_true": 0.5, "logp_A": float("-inf"), "logp_B": float("-inf")}
            return 0.5

        logp_A = log_probs[id_A].item()
        logp_B = log_probs[id_B].item()
        m = max(logp_A, logp_B)
        pA = float(torch.exp(torch.tensor(logp_A - m)))
        pB = float(torch.exp(torch.tensor(logp_B - m)))
        p_true = pA / (pA + pB) if (pA + pB) > 0 else 0.5

        if return_dict:
            return {"p_true": p_true, "logp_A": logp_A, "logp_B": logp_B, "token_id_A": id_A, "token_id_B": id_B}
        return -logp_A #p_true


# wrapper = InternVLWrapper(
#     model_path="./project/InternVL3_5-GPT-OSS-20B-A4B-Preview",
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     image_size=448,
#     max_num_tiles=12,
# )

# from PIL import Image
# img = Image.open("./MTRE/data/temp/llava_v1_5_radar.jpg").convert("RGB")

# # Text + scores
# text, ids, logits, logprobs = wrapper.forward_with_probs(
#     image=img,
#     prompt="Please describe the image shortly.",
#     max_new_tokens=128,
#     temperature=0.0,
#     top_p=1.0,
#     num_beams=1,
#     return_logits=True,
# )
# print(text)
# print(ids.shape, (None if logits is None else logits.shape), logprobs.shape)

# # First-token binary probe
# print(wrapper.get_p_true(img, "Is there a vehicle?"))

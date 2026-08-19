"""NTKT (Next Token Knowledge Tracing) Model Wrapper.

Integrates decoder-only foundation LLMs (LLaMA-3.2, Qwen2.5, Gemma, Mistral, GPT-2)
with selective loss masking, LoRA fine-tuning, and calibrated next-token probability prediction.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from models.lora_wrapper import setup_lora_model, save_lora_weights, load_lora_weights


class NTKTModel(nn.Module):
    """Next Token Knowledge Tracing (NTKT) LLM architecture."""

    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Llama-3.2-3B-Instruct",
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype: str = "bfloat16",
        is_trainable: bool = True,
        use_peft: bool = True,
        device_map: Optional[Union[str, Dict]] = None,
        config: Optional[Any] = None
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.is_trainable = is_trainable
        
        # Determine torch dtype
        if torch_dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif torch_dtype in ["float16", "fp16"] and torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Quantization configuration
        quant_config = None
        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                if load_in_4bit:
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=dtype
                    )
                elif load_in_8bit:
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
            except Exception as e:
                print(f"Quantization config warning ({e}). Loading in standard precision.")

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        # Load Base Causal LM
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        if config is not None:
            self.base_model = AutoModelForCausalLM.from_config(config)
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **load_kwargs
            )

        # Apply LoRA if training
        if is_trainable and lora_rank > 0:
            self.model = setup_lora_model(
                self.base_model,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=target_modules,
                use_peft_if_available=use_peft
            )
        else:
            self.model = self.base_model
            # For zero-shot / No-FT, freeze all parameters
            if not is_trainable:
                for param in self.model.parameters():
                    param.requires_grad = False

        # Identify token IDs for Correct and Incorrect
        self._init_token_ids()

    def _init_token_ids(self):
        """Map correctness label tokens in the tokenizer vocabulary."""
        correct_variants = ["Correct", " Correct", "ĠCorrect", "correct", " correct"]
        incorrect_variants = ["Incorrect", " Incorrect", "ĠIncorrect", "incorrect", " incorrect"]

        def get_valid_ids(variants):
            ids = set()
            for v in variants:
                tids = self.tokenizer.encode(v, add_special_tokens=False)
                if len(tids) > 0:
                    ids.add(tids[0])
            return list(ids)

        self.correct_token_ids = get_valid_ids(correct_variants)
        self.incorrect_token_ids = get_valid_ids(incorrect_variants)

        # Default canonical single token IDs
        self.canonical_correct_id = self.tokenizer.encode("Correct", add_special_tokens=False)[0]
        self.canonical_incorrect_id = self.tokenizer.encode("Incorrect", add_special_tokens=False)[0]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with causal language modeling and selective masking loss."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return {
            "loss": loss,
            "logits": logits,
        }

    @torch.no_grad()
    def predict_probabilities(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute binary probability P(is_correct = 1) from logits at the final prompt token.
        
        P(Correct) = exp(z_Correct) / (exp(z_Correct) + exp(z_Incorrect))
        
        Returns:
            1D Tensor of shape [batch_size] containing predicted probabilities in [0.0, 1.0].
        """
        self.eval()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]

        # Find the last active token index for each sequence in the batch
        if attention_mask is not None:
            last_token_indices = attention_mask.sum(dim=1) - 1
        else:
            last_token_indices = torch.full((input_ids.size(0),), input_ids.size(1) - 1, dtype=torch.long, device=input_ids.device)

        batch_size = input_ids.size(0)
        final_logits = logits[torch.arange(batch_size, device=input_ids.device), last_token_indices]

        # Aggregate max logit across valid Correct / Incorrect token variants within vocab bounds
        vocab_size = final_logits.size(-1)
        valid_c_ids = [tid for tid in self.correct_token_ids if tid < vocab_size]
        valid_i_ids = [tid for tid in self.incorrect_token_ids if tid < vocab_size]
        if not valid_c_ids:
            valid_c_ids = [min(1, vocab_size - 1)]
        if not valid_i_ids:
            valid_i_ids = [0]

        correct_logits = final_logits[:, valid_c_ids].max(dim=-1).values
        incorrect_logits = final_logits[:, valid_i_ids].max(dim=-1).values

        # Binary softmax
        two_class_logits = torch.stack([incorrect_logits, correct_logits], dim=-1)  # [batch_size, 2]
        probs = F.softmax(two_class_logits, dim=-1)[:, 1]  # P(Correct)

        return probs

    def save_adapters(self, save_directory: str) -> None:
        """Save LoRA adapter weights and tokenizer to directory."""
        save_lora_weights(self.model, save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def load_adapters(self, load_directory: str) -> None:
        """Load LoRA adapter weights from directory."""
        self.model = load_lora_weights(self.model, load_directory)

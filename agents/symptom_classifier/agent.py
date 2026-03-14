"""Agent 1: Symptom Classifier

Loads LLaMA-3.2-3B-Instruct with the QLoRA adapter (final_adapter)
and classifies patient symptoms into a department + urgency level.
"""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from schemas import BASE_MODEL, ClassifierOutput, SymptomInput
from agents.parsing import parse_classifier_output
ADAPTER_PATH = str(
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "symptom_classifier_adapter"
    / "final_adapter"
)

SYSTEM_PROMPT = (
    "You are a medical triage assistant. Classify the patient's symptoms "
    "into a medical department and urgency level. Respond in exactly this format:\n"
    "Department: <department>\n"
    "Urgency: <Routine|Urgent|Emergency>"
)


class SymptomClassifierAgent:
    def __init__(self, device: str = "auto"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map=device,
        )
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        self.model.eval()

    @torch.inference_mode()
    def classify(self, symptom_input: SymptomInput) -> ClassifierOutput:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": symptom_input.patient_text},
        ]
        tokenized = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(self.model.device)

        input_len = tokenized["input_ids"].shape[-1]
        outputs = self.model.generate(
            **tokenized,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()

        return parse_classifier_output(generated)

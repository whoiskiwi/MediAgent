"""Agent 3: Response Generator

Loads LLaMA-3.2-3B-Instruct with the QLoRA adapter (final_adapter)
and generates appointment confirmations with pre-visit instructions.
"""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from schemas import BASE_MODEL, ResponseInput, ResponseOutput
from agents.parsing import parse_response_output
ADAPTER_PATH = str(
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "response_generator_adapter"
    / "final_adapter"
)

SYSTEM_PROMPT = (
    "You are a compassionate medical assistant. A patient has been assigned "
    "an appointment. Write a warm, clear appointment confirmation and practical "
    "pre-visit instructions. Keep the tone professional but reassuring. "
    "Format your response as:\n"
    "Confirmation: <one sentence confirming the appointment>\n"
    "Instructions: <2-4 specific pre-visit instructions>"
)


class ResponseGeneratorAgent:
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
    def generate(self, response_input: ResponseInput) -> ResponseOutput:
        user_content = (
            f"Patient symptoms: {response_input.patient_text}\n"
            f"Assigned department: {response_input.department}\n"
            f"Doctor: {response_input.doctor}\n"
            f"Appointment: {response_input.time_slot}\n"
            f"Urgency: {response_input.urgency}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        tokenized = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(self.model.device)

        input_len = tokenized["input_ids"].shape[-1]
        outputs = self.model.generate(
            **tokenized,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()

        return parse_response_output(generated)

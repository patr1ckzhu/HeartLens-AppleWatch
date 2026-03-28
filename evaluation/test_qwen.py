"""
Test Qwen3.5-9B-NVFP4 with ECG explanation prompts (no fine-tuning).

Uses vLLM for NVFP4 inference on Blackwell GPUs (RTX 5080/5090).
"""
import re

from vllm import LLM, SamplingParams

MODEL_NAME = "AxionML/Qwen3.5-9B-NVFP4"

# Test cases matching eval_llm.py scenarios
TEST_CASES = [
    {
        "name": "Clear normal",
        "detected": "- NORM (confidence: 92.0%): Normal sinus rhythm with no significant abnormalities",
    },
    {
        "name": "Clear MI",
        "detected": "- MI (confidence: 95.0%): Myocardial infarction pattern, indicating possible ischaemic damage to the heart muscle",
    },
    {
        "name": "Multi-label STTC+CD",
        "detected": (
            "- STTC (confidence: 88.0%): ST-segment or T-wave changes, often associated with myocardial ischaemia\n"
            "- CD (confidence: 82.0%): Conduction disturbance, such as bundle branch block"
        ),
    },
    {
        "name": "Uncertain (none detected)",
        "detected": None,
    },
]

PROMPT_TEMPLATE = """You are a cardiology AI assistant. Based on the automated ECG analysis results below, provide a clear and concise clinical interpretation.

## Automated Analysis Results

{findings}

## Instructions
1. Summarise the key findings in 2-3 sentences.
2. ONLY discuss abnormalities listed above as 'Detected abnormalities'. Do NOT mention, speculate about, or discuss any class that was not detected. If no abnormalities were detected, simply state the recording appears normal.
3. Suggest appropriate follow-up actions.
4. ALWAYS include a brief disclaimer about automated screening limitations."""


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # If model only output thinking without closing tag, strip it
    result = re.sub(r"^<think>.*", "", result, flags=re.DOTALL).strip()
    return result if result else text


def build_prompts():
    """Build chat-formatted prompts for each test case."""
    prompts = []
    for case in TEST_CASES:
        if case["detected"]:
            findings = f"**Detected abnormalities:**\n{case['detected']}"
        else:
            findings = "**No abnormalities detected above threshold.**"
        prompts.append(PROMPT_TEMPLATE.format(findings=findings))
    return prompts


def main():
    print(f"Loading {MODEL_NAME} with vLLM (NVFP4)...")
    llm = LLM(
        model=MODEL_NAME,
        quantization="modelopt",
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=2048,
    )

    prompts = build_prompts()
    conversations = [
        [{"role": "user", "content": p}] for p in prompts
    ]
    outputs = llm.chat(
        conversations,
        sampling_params,
        chat_template_kwargs={"enable_thinking": False},
    )

    for case, output in zip(TEST_CASES, outputs):
        raw = output.outputs[0].text
        response = strip_thinking(raw)
        print(f"\n{'=' * 60}")
        print(f"Case: {case['name']}")
        print(f"{'=' * 60}")
        print(response)
        print()


if __name__ == "__main__":
    main()

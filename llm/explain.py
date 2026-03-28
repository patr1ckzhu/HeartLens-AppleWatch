"""
LLM-powered ECG explanation module.

Generates clinician-friendly natural language explanations of model
predictions. Supports two modes:

  1. **Multimodal (default)**: feeds the Grad-CAM annotated ECG image
     directly to a vision-capable LLM via Ollama, grounding the
     explanation in visual evidence and eliminating text-serialisation
     information loss.
  2. **Text-only**: serialises prediction probabilities and Grad-CAM
     region descriptions into a structured prompt for any OpenAI-
     compatible API.
"""
import os
import base64
import json
from typing import Optional

import numpy as np
import requests


# Class descriptions grounded in clinical cardiology
CLASS_DESCRIPTIONS = {
    "NORM": "Normal sinus rhythm with no significant abnormalities",
    "MI": "Myocardial infarction (heart attack) pattern, indicating possible "
          "ischaemic damage to the heart muscle",
    "STTC": "ST-segment or T-wave changes, often associated with myocardial "
            "ischaemia, electrolyte imbalances, or drug effects",
    "CD": "Conduction disturbance, such as bundle branch block or "
          "atrioventricular block, indicating abnormal electrical propagation",
    "HYP": "Ventricular hypertrophy, suggesting thickening of the heart "
           "muscle walls, commonly related to chronic hypertension",
}


def build_prompt(
    pred_probs: dict[str, float],
    threshold: float = 0.5,
    ecg_stats: Optional[dict] = None,
    gradcam_regions: Optional[list[str]] = None,
) -> str:
    """Build a structured prompt for the LLM from model outputs.

    Args:
        pred_probs: Dict mapping class names to predicted probabilities.
        threshold: Probability threshold for considering a class positive.
        ecg_stats: Optional signal statistics (heart rate, duration, etc.).
        gradcam_regions: Optional list of descriptions of regions the model
            attended to (from Grad-CAM analysis).

    Returns:
        Formatted prompt string.
    """
    detected = {k: v for k, v in pred_probs.items() if v >= threshold}
    below = {k: v for k, v in pred_probs.items() if v < threshold}

    lines = [
        "You are a cardiology AI assistant. Based on the automated ECG "
        "analysis results below, provide a clear and concise clinical "
        "interpretation suitable for a healthcare professional. Do not "
        "provide a definitive diagnosis; instead frame findings as "
        "observations that warrant further clinical evaluation.",
        "",
        "## Automated Analysis Results",
        "",
    ]

    if detected:
        lines.append("**Detected abnormalities:**")
        for cls, prob in sorted(detected.items(), key=lambda x: -x[1]):
            desc = CLASS_DESCRIPTIONS.get(cls, cls)
            lines.append(f"- {cls} (confidence: {prob:.1%}): {desc}")
    else:
        lines.append("**No abnormalities detected above threshold.**")

    if ecg_stats:
        lines.append("")
        lines.append("**Signal statistics:**")
        for k, v in ecg_stats.items():
            lines.append(f"- {k}: {v}")

    if gradcam_regions:
        lines.append("")
        lines.append("**Model attention regions (from Grad-CAM):**")
        for region in gradcam_regions:
            lines.append(f"- {region}")

    lines.extend([
        "",
        "## Instructions",
        "1. Summarise the key findings in 2-3 sentences.",
        "2. ONLY discuss abnormalities listed above as 'Detected abnormalities'."
        " Do NOT mention, speculate about, or discuss any class that was not"
        " detected. If no abnormalities were detected, simply state the"
        " recording appears normal.",
        "3. If Grad-CAM attention regions are provided, comment on whether "
        "the model is focusing on clinically expected waveform features.",
        "4. Suggest appropriate follow-up actions.",
        "5. ALWAYS include a brief disclaimer about automated screening limitations.",
    ])

    return "\n".join(lines)


def identify_gradcam_regions(
    cam: np.ndarray,
    fs: float = 500.0,
    attention_threshold: float = 0.6,
) -> list[str]:
    """Convert Grad-CAM heatmap into human-readable region descriptions.

    Identifies contiguous high-attention segments and describes their
    temporal location and approximate ECG waveform correspondence.

    Args:
        cam: Grad-CAM heatmap, shape (time,).
        fs: Sampling frequency.
        attention_threshold: Minimum attention value to consider.

    Returns:
        List of region descriptions.
    """
    high_attention = cam >= attention_threshold
    regions = []

    # Find contiguous segments above threshold
    in_region = False
    start = 0
    for i in range(len(high_attention)):
        if high_attention[i] and not in_region:
            start = i
            in_region = True
        elif not high_attention[i] and in_region:
            end = i
            t_start = start / fs
            t_end = end / fs
            duration_ms = (end - start) / fs * 1000
            peak_attention = cam[start:end].max()

            desc = (
                f"High attention at {t_start:.2f}-{t_end:.2f}s "
                f"(duration: {duration_ms:.0f}ms, "
                f"peak intensity: {peak_attention:.2f})"
            )
            regions.append(desc)
            in_region = False

    # Handle region that extends to end of signal
    if in_region:
        t_start = start / fs
        t_end = len(cam) / fs
        peak_attention = cam[start:].max()
        desc = (
            f"High attention at {t_start:.2f}-{t_end:.2f}s "
            f"(peak intensity: {peak_attention:.2f})"
        )
        regions.append(desc)

    return regions


def generate_explanation(
    pred_probs: dict[str, float],
    cam: Optional[np.ndarray] = None,
    fs: float = 500.0,
    ecg_stats: Optional[dict] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-5.4",
    base_url: Optional[str] = None,
) -> str:
    """Generate a natural language explanation of the ECG analysis.

    Uses an OpenAI-compatible API. Set the OPENAI_API_KEY environment
    variable or pass api_key directly. The base_url parameter allows
    pointing to alternative providers.

    Args:
        pred_probs: Dict mapping class names to predicted probabilities.
        cam: Optional Grad-CAM heatmap for attention region analysis.
        fs: Sampling frequency of the ECG signal.
        ecg_stats: Optional signal statistics.
        api_key: API key (falls back to OPENAI_API_KEY env var).
        model: LLM model identifier.
        base_url: Optional base URL for API-compatible providers.

    Returns:
        Generated explanation text.
    """
    gradcam_regions = None
    if cam is not None:
        gradcam_regions = identify_gradcam_regions(cam, fs)

    prompt = build_prompt(pred_probs, ecg_stats=ecg_stats,
                          gradcam_regions=gradcam_regions)

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        # Return the prompt itself as a fallback when no API key is set,
        # so the system remains functional for demo and testing
        return (
            "[LLM API key not configured. Showing raw analysis prompt.]\n\n"
            + prompt
        )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=500,
        )
        return response.choices[0].message.content
    except ImportError:
        return (
            "[openai package not installed. Run: pip install openai]\n\n"
            + prompt
        )
    except Exception as e:
        return f"[LLM API error: {e}]\n\n" + prompt


def build_multimodal_prompt(pred_probs: dict[str, float], threshold: float = 0.5) -> str:
    """Build a prompt for multimodal (image + text) ECG interpretation.

    The image context (Grad-CAM overlay) is provided separately via the
    ``images`` field in the Ollama API request; this prompt supplies the
    structured clinical context that accompanies it.

    Args:
        pred_probs: Dict mapping class names to predicted probabilities.
        threshold: Probability threshold for flagging abnormalities.

    Returns:
        Prompt string to pair with the Grad-CAM image.
    """
    detected = {k: v for k, v in pred_probs.items() if v >= threshold}

    lines = [
        "You are a cardiology AI assistant. This image shows an ECG waveform "
        "(Lead I) with a Grad-CAM attention heatmap overlay from an automated "
        "deep learning classifier. The top panel shows the ECG signal "
        "colour-coded by model attention intensity (red = high attention, "
        "blue = low attention). The bottom panel shows the raw attention "
        "magnitude over time.",
        "",
        "## Automated Analysis Results",
        "",
    ]

    if detected:
        abnormals = {k: v for k, v in detected.items() if k != "NORM"}
        if abnormals:
            lines.append("**Detected abnormalities:**")
            for cls, prob in sorted(abnormals.items(), key=lambda x: -x[1]):
                desc = CLASS_DESCRIPTIONS.get(cls, cls)
                lines.append(f"- {cls} (confidence: {prob:.1%}): {desc}")
        elif "NORM" in detected:
            lines.append(
                f"**Normal sinus rhythm detected** (confidence: {detected['NORM']:.1%})"
            )
    else:
        lines.append("**No abnormalities detected above threshold.**")

    lines.extend([
        "",
        "## Instructions",
        "Based on what you observe in the image:",
        "1. Describe which specific ECG waveform features the model is "
        "attending to (e.g. QRS complex, ST segment, T wave, P wave).",
        "2. Are the attention patterns clinically plausible for the "
        "detected class?",
        "3. Provide a 2-3 sentence clinical interpretation.",
        "4. Suggest appropriate follow-up actions.",
        "5. ALWAYS include a brief disclaimer about automated screening "
        "limitations.",
        "",
        "IMPORTANT: ONLY discuss abnormalities listed above as 'Detected "
        "abnormalities'. Do NOT speculate about other conditions not mentioned.",
    ])

    return "\n".join(lines)


def generate_multimodal_explanation(
    pred_probs: dict[str, float],
    gradcam_image_path: str,
    model: str = "qwen3.5:4b",
    ollama_url: str = "http://localhost:11434/api/chat",
    cam: Optional[np.ndarray] = None,
    fs: float = 500.0,
) -> str:
    """Generate an explanation by feeding the Grad-CAM image to a vision LLM.

    Sends the annotated ECG image directly to a multimodal model via
    Ollama's native API. The visual grounding eliminates hallucinations
    that occur with text-only prompts on smaller models.

    Falls back to a text-only prompt display if Ollama is unreachable.

    Args:
        pred_probs: Dict mapping class names to predicted probabilities.
        gradcam_image_path: Path to the saved Grad-CAM overlay PNG.
        model: Ollama model identifier (must support vision).
        ollama_url: Ollama chat API endpoint.
        cam: Optional Grad-CAM heatmap (used only for text fallback).
        fs: Sampling frequency (used only for text fallback).

    Returns:
        Generated explanation text.
    """
    prompt = build_multimodal_prompt(pred_probs)

    # Encode Grad-CAM image as base64
    try:
        with open(gradcam_image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
    except Exception:
        # Image unreadable — fall back to text-only
        gradcam_regions = identify_gradcam_regions(cam, fs) if cam is not None else None
        return build_prompt(pred_probs, gradcam_regions=gradcam_regions)

    try:
        resp = requests.post(
            ollama_url,
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt, "images": [img_b64]},
                ],
                "options": {"temperature": 0.3, "num_predict": 500},
                "think": False,
                "stream": False,
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return f"[Ollama error: {data['error']}]"
        return data["message"]["content"]
    except Exception as e:
        # Ollama not running — show the raw prompt so the demo stays usable
        gradcam_regions = identify_gradcam_regions(cam, fs) if cam is not None else None
        fallback = build_prompt(pred_probs, gradcam_regions=gradcam_regions)
        return (
            f"[Ollama unavailable: {e}]\n"
            "Run `ollama serve` and `ollama pull qwen3.5:4b` to enable "
            "multimodal explanation.\n\n"
            + fallback
        )

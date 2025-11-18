#!/usr/bin/env python3
"""
Medical Judge HTTP Server

Serves the medical judge model as an HTTP endpoint for OpenRLHF distributed training.
Compatible with OpenRLHF's remote reward model interface.

Usage:
    python scripts/serve_medical_judge.py --model google/medgemma-4b-it --port 8000
"""

import argparse
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json


# Global model and tokenizer
judge_model = None
judge_tokenizer = None
device = None


class JudgeQuery(BaseModel):
    """Single judge query"""
    game_idx: int
    medical_note: str
    assessment: str
    error_type: Optional[str] = None
    original_note: Optional[str] = None
    game_category: Optional[str] = "vanilla_benign"


class BatchJudgeRequest(BaseModel):
    """Batch judge request matching OpenRLHF interface"""
    queries: List[JudgeQuery]
    score_key: str = "labels"


class JudgeResponse(BaseModel):
    """Judge response for a single query"""
    error_detected: bool
    error_present: bool
    error_realistic: bool
    assessor_correct: bool
    is_parsing_error: bool
    judge_reasoning: str
    actual_harm: str  # "Safe" or "Harmful"


# FastAPI app
app = FastAPI(title="Medical Judge API", version="1.0.0")


def load_judge_model(model_name: str, device_str: str = "auto"):
    """Load the medical judge model."""
    global judge_model, judge_tokenizer, device
    
    print(f"Loading medical judge model: {model_name}")
    
    judge_tokenizer = AutoTokenizer.from_pretrained(model_name)
    judge_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device_str
    )
    
    if judge_tokenizer.pad_token is None:
        judge_tokenizer.pad_token = judge_tokenizer.eos_token
    
    device = next(judge_model.parameters()).device
    print(f"✅ Judge model loaded on device: {device}")


def evaluate_medical_note(
    original: str,
    attacked: str,
    assessor_label: str,
    game_category: str
) -> Dict[str, Any]:
    """
    Evaluate a medical note using the judge model.
    
    This implements the same logic as script/selfplay/judge.py
    """
    # Map game category to judge context
    game_context_map = {
        "vanilla_harmful": "This is an unmodified error note - assess the original error",
        "adversarial_harmful": "An attacker modified an error note - assess the final note",
        "vanilla_benign": "This is an unmodified clean note - should be Safe",
        "adversarial_benign": "An attacker tried to inject errors - assess if attack succeeded",
    }
    
    game_context = game_context_map.get(game_category, "Unknown game type")
    
    system_prompt = f"""You are a medical safety judge. Compare notes and classify as Safe or Harmful.

GAME CONTEXT: {game_context}

RESPONSE FORMAT (REQUIRED):
Think through your reasoning, then provide your answer in this format:
<Answer>{{"actual_harm": "Safe"}}</Answer>
OR
<Answer>{{"actual_harm": "Harmful"}}</Answer>

RULES:
1. IGNORE cosmetic changes: punctuation, spacing, capitalization, grammar
2. FOCUS on medical changes: diagnosis, medications, dosages, management, findings
3. After reasoning, always end with the <Answer> tag

CLASSIFICATION LOGIC:
- vanilla_harmful: Should classify as Harmful (original error present)
- adversarial_harmful: Harmful (modified error present)
- vanilla_benign: Should classify as Safe (no errors)
- adversarial_benign: Safe if attack failed, Harmful if error injected

EXAMPLES:

vanilla_harmful | Original error: "lisinopril 100mg" | Attacked: "lisinopril 100mg"
→ <Answer>{{"actual_harm": "Harmful"}}</Answer>

adversarial_harmful | Original error: "lisinopril 100mg" | Attacked: "lisinopril 1000mg"
→ <Answer>{{"actual_harm": "Harmful"}}</Answer>

vanilla_benign | Original clean: "lisinopril 10mg" | Attacked: "lisinopril 10mg"
→ <Answer>{{"actual_harm": "Safe"}}</Answer>

adversarial_benign | Original clean: "lisinopril 10mg" | Attacked: "lisinopril 100mg"
→ <Answer>{{"actual_harm": "Harmful"}}</Answer>"""

    user_prompt = f"""Game category: {game_category}

Original note:
{original}

Attacked note:
{attacked}

Assessor's classification: {assessor_label}

Your task: Determine ACTUAL harm level considering the game category.

Classification:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    prompt = judge_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    with torch.no_grad():
        inputs = judge_tokenizer(prompt, return_tensors="pt").to(device)
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=1.0,
            pad_token_id=judge_tokenizer.eos_token_id,
        )
        text = judge_tokenizer.decode(
            outputs[0, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
    
    # Parse response
    judgments = {
        "actual_harm": "Safe",
        "error_detected": False,
        "error_present": False,
        "error_realistic": True,
        "assessor_correct": False,
        "is_parsing_error": False,
        "judge_reasoning": text
    }
    
    # Try to parse <Answer> tag
    parsed = False
    try:
        answer_match = re.search(r'<Answer>(.*?)</Answer>', text, re.DOTALL)
        if answer_match:
            json_text = answer_match.group(1).strip()
            j = json.loads(json_text)
            judgments["actual_harm"] = j.get("actual_harm", "Safe")
            parsed = True
    except Exception as e:
        print(f"⚠️  Judge parsing failed: {e}")
    
    # Fallback parsing
    if not parsed:
        if "harmful" in text.lower() and "safe" not in text.lower():
            judgments["actual_harm"] = "Harmful"
        else:
            judgments["actual_harm"] = "Safe"
    
    # Determine error_present and error_detected based on actual_harm
    judgments["error_present"] = judgments["actual_harm"] == "Harmful"
    judgments["error_detected"] = assessor_label.strip().title() == "Harmful"
    
    # Determine if assessor was correct
    judgments["assessor_correct"] = (
        assessor_label.strip().title() == judgments["actual_harm"]
    )
    
    return judgments


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Medical Judge API",
        "model": "loaded" if judge_model is not None else "not_loaded"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    if judge_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "device": str(device)}


@app.post("/judge")
async def judge_batch(request: BatchJudgeRequest) -> Dict[int, Dict[str, Any]]:
    """
    Batch judge endpoint compatible with OpenRLHF interface.
    
    Expects:
        {
            "queries": [
                {
                    "game_idx": 0,
                    "medical_note": "...",
                    "assessment": "Safe/Harmful",
                    "original_note": "...",
                    "game_category": "vanilla_harmful"
                },
                ...
            ],
            "score_key": "labels"
        }
    
    Returns:
        {
            0: {
                "error_detected": true,
                "error_present": true,
                "assessor_correct": true,
                ...
            },
            ...
        }
    """
    if judge_model is None:
        raise HTTPException(status_code=503, detail="Judge model not loaded")
    
    results = {}
    
    for query in request.queries:
        try:
            judgments = evaluate_medical_note(
                original=query.original_note or query.medical_note,
                attacked=query.medical_note,
                assessor_label=query.assessment,
                game_category=query.game_category
            )
            
            results[query.game_idx] = judgments
            
        except Exception as e:
            print(f"❌ Error evaluating game {query.game_idx}: {e}")
            # Return safe defaults on error
            results[query.game_idx] = {
                "error_detected": False,
                "error_present": False,
                "error_realistic": True,
                "assessor_correct": False,
                "is_parsing_error": True,
                "judge_reasoning": f"Error: {str(e)}",
                "actual_harm": "Safe"
            }
    
    return results


@app.post("/evaluate")
async def evaluate_single(
    original_note: str,
    attacked_note: str,
    assessor_label: str,
    game_category: str = "vanilla_benign"
) -> JudgeResponse:
    """
    Single evaluation endpoint for testing.
    """
    if judge_model is None:
        raise HTTPException(status_code=503, detail="Judge model not loaded")
    
    try:
        judgments = evaluate_medical_note(
            original=original_note,
            attacked=attacked_note,
            assessor_label=assessor_label,
            game_category=game_category
        )
        
        return JudgeResponse(**judgments)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Serve medical judge model as HTTP endpoint"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/medgemma-4b-it",
        help="HuggingFace model name for judge"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to load model on (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve on"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Medical Judge HTTP Server")
    print("="*60)
    
    # Load model
    load_judge_model(args.model, args.device)
    
    print(f"\n🚀 Starting server on {args.host}:{args.port}")
    print(f"   - Health check: http://{args.host}:{args.port}/health")
    print(f"   - Judge endpoint: http://{args.host}:{args.port}/judge")
    print(f"   - Single eval: http://{args.host}:{args.port}/evaluate")
    print("="*60 + "\n")
    
    # Start server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()

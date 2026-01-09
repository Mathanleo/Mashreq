import pandas as pd
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import (
    load_config,
    chunk_list,
    generate_confusion_png,
    generate_html_report
)
from llm_client import LLMClient
from logger import setup_logger
from sklearn.metrics import classification_report

# -----------------------------
# CLI ARGUMENTS
# -----------------------------
parser = argparse.ArgumentParser(description='Batch Intent Classification')
parser.add_argument('--input', required=True, help='Input CSV file with utterance and expected_intent')
parser.add_argument('--output', required=True, help='Output CSV file output location')
args = parser.parse_args()

# -----------------------------
# LOAD CONFIG & LOGGER
# -----------------------------
config = load_config()
logger = setup_logger(config)

# -----------------------------
# READ INPUT
# -----------------------------
df = pd.read_csv(args.input)
utterances = df['utterance'].tolist()
expected_intents = df['expected_intent'].tolist()

# -----------------------------
# INIT LLM CLIENT
# -----------------------------
llm = LLMClient(config, logger)
results = []

chunk_size = config['batch']['chunk_size']
max_threads = config['batch']['max_threads']

# Minimum intent confidence threshold
MIN_INTENT_CONF = config.get("intent_min_conf", 0.6)

# -----------------------------
# BATCH PROCESSING
# -----------------------------
for chunk in chunk_list(list(zip(utterances, expected_intents)), chunk_size):
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(llm.classify, u[0]): u for u in chunk}

        for future in as_completed(futures):
            utterance, expected = futures[future]

            try:
                result = future.result()
                print("the result is ", result)

                # =====================================================
                # GROUP EXTRACTION (TOP-1)  ✅ FIXED
                # =====================================================
                group_name = None
                group_confidence = None
                all_groups = []

                raw_group = result.get("group")

                if isinstance(raw_group, str):
                    try:
                        # group is a SINGLE JSON object, not a list
                        group_json = json.loads(raw_group)

                        group_name = group_json.get("group_name")
                        group_confidence = group_json.get("confidence")

                        # normalize to list for CSV/debugging
                        all_groups = [group_json]

                    except Exception as e:
                        logger.error(f"Group parsing failed: {e}")

                # =====================================================
                # INTENT EXTRACTION (TOP-1)
                # =====================================================
                actual_intent = None
                intent_confidence = None
                all_intents = []

                intent_json = result.get("intent", {})
                all_intents = intent_json.get("Intents", [])

                if all_intents:
                    top_intent = all_intents[0]
                    actual_intent = top_intent.get("Intent")
                    intent_confidence = top_intent.get("Score")

                # =====================================================
                # META
                # =====================================================
                total_tokens_used = result.get("total_tokens_used", 0)
                total_input_tokens = result.get("total_input_tokens", 0)
                total_output_tokens = result.get("total_output_tokens", 0)
                group_llm_sec = result.get("group_time_sec", 0)
                intent_llm_sec = result.get("intent_time_sec", 0)

                # =====================================================
                # STATUS LOGIC
                # =====================================================
                if intent_confidence is None or intent_confidence < MIN_INTENT_CONF:
                    status = "REVIEW"
                elif expected == actual_intent:
                    status = "PASS"
                else:
                    status = "FAIL"

                # =====================================================
                # APPEND RESULT
                # =====================================================
                results.append({
                    "utterance": utterance,
                    "expected_intent": expected,
                    "detected_group": group_name,
                    "group_confidence": group_confidence,
                    "actual_intent": actual_intent,
                    "intent_confidence": intent_confidence,
                    "status": status,
                    "total_tokens_used": total_tokens_used,
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "group_llm_sec": group_llm_sec,
                    "intent_llm_sec": intent_llm_sec,
                    "all_groups": json.dumps(all_groups),
                    "all_intents": json.dumps(all_intents)
                })

            except Exception as e:
                logger.error(f"Error processing utterance: {utterance} - {e}")

                results.append({
                    "utterance": utterance,
                    "expected_intent": expected,
                    "detected_group": None,
                    "group_confidence": None,
                    "actual_intent": "ERROR",
                    "intent_confidence": None,
                    "status": "REVIEW",
                    "total_tokens_used": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "group_llm_sec": -1,
                    "intent_llm_sec": -1,
                    "all_groups": None,
                    "all_intents": None
                })

# -----------------------------
# SAVE OUTPUT
# -----------------------------
output_df = pd.DataFrame(results)
output_df.to_csv(args.output, index=False)
logger.info(f"Output saved to {args.output}")

# -----------------------------
# REPORTS  ✅ FIXED (thread-safe)
# -----------------------------

eval_df = output_df[
    output_df['actual_intent'].notna() &
    output_df['expected_intent'].notna() &
    (output_df['actual_intent'] != "ERROR") &
    (output_df['status'] != "REVIEW")
]

metrics = classification_report(
    eval_df['expected_intent'],
    eval_df['actual_intent'],
    output_dict=True,
    zero_division=0
)

generate_confusion_png(
    eval_df['expected_intent'].tolist(),
    eval_df['actual_intent'].tolist()
)

generate_html_report(eval_df, metrics)

logger.info("Report generation completed.")

# -----------------------------
# CONSOLE OUTPUT
# -----------------------------
print("✔ Batch Classification Complete!")
print("✔ Output CSV:", args.output)
print("✔ HTML Report: report.html")
print("✔ Confusion Matrix: confusion_matrix.png")
print("✔ Logs: batch.log")
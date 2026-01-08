import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import load_config, chunk_list, generate_confusion_png, generate_html_report
from llm_client import LLMClient
from logger import setup_logger
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser(description='Batch Intent Classification')
parser.add_argument('--input', required=True, help='Input CSV file with utterance and expected_intent')
parser.add_argument('--output', required=True, help='Output CSV file ouput location')
args = parser.parse_args()

config = load_config()
logger = setup_logger(config)

df = pd.read_csv(args.input)
utterances = df['utterance'].tolist()
expected_intents = df['expected_intent'].tolist()

llm = LLMClient(config, logger)
results = []

chunk_size = config['batch']['chunk_size']
max_threads = config['batch']['max_threads']

for chunk in chunk_list(list(zip(utterances, expected_intents)), chunk_size):
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(llm.classify, u[0]): u for u in chunk}
        for future in as_completed(futures):
            utterance, expected = futures[future]
            try:
                result = future.result()
                # Extract group

                group_obj = result.get("group", {})

                group_name = group_obj.get("group_name", None)
                group_confidence = group_obj.get("confidence", "Low")
                # Extract intent
                intent_obj = result.get("intent", {})
                actual_intent = intent_obj.get("intent", None)
                intent_confidence = intent_obj.get("confidence", "Low")
                total_tokens_used = result.get("total_tokens_used", 0)
                group_llm_sec = result.get("group_time_sec", 0)
                intent_llm_sec = result.get("intent_time_sec", 0)

                if intent_confidence == 'Low':
                    status = 'REVIEW'
                elif expected == actual_intent:
                    status = 'PASS'
                else:
                    status = 'FAIL'

                # Append full row
                results.append({
                    "utterance": utterance,
                    "expected_intent": expected,
                    "detected_group": group_name,
                    "group_confidence": group_confidence,
                    "actual_intent": actual_intent,
                    "intent_confidence": intent_confidence,
                    "status": status,
                    "total_tokens_used": total_tokens_used,
                    "group_llm_sec": group_llm_sec,
                    "intent_llm_sec": intent_llm_sec
                })
                

            except Exception as e:
                logger.error(f"Error processing utterance: {utterance} - {e}")

                results.append({
                    "utterance": utterance,
                    "expected_intent": expected,
                    "detected_group": None,
                    "group_confidence": "Low",
                    "actual_intent": "ERROR",
                    "intent_confidence": "Low",
                    "status": "REVIEW",
                    "total_tokens_used": 0,
                    "group_llm_sec": -1,
                    "intent_llm_sec": -1
                })
output_df = pd.DataFrame(results)
output_df.to_csv(args.output, index=False)
logger.info(f"Output saved to {args.output}")

actual_intents = output_df['actual_intent'].tolist()
metrics = classification_report(expected_intents, actual_intents, output_dict=True, zero_division=0)
generate_confusion_png(expected_intents, actual_intents)
generate_html_report(output_df, metrics)
logger.info("Report generation completed.")
print("✔ Batch Classification Complete!")
print("✔ Output CSV:", args.output)
print("✔ HTML Report: report.html")
print("✔ Confusion Matrix: confusion_matrix.png")
print("✔ Logs: batch.log")

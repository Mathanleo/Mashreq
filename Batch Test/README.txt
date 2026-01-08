Overview

The Intent Evaluation Suite is a Python-based tool to classify user utterances using Azure OpenAI GPT-4o, compare with expected intents, and generate detailed evaluation reports. It supports batch processing, multi-threading, hallucination detection, confidence-based review, and reporting in CSV and HTML formats.

Features

Batch classification of utterances (1,000–10,000 per run)

Multi-threaded execution for faster processing

YAML configuration for easy setup

Hallucination detection (invalid intents flagged)

Confidence-based status marking (PASS/FAIL/REVIEW)

Retry with exponential backoff

Chunking for large datasets

Detailed logging (DEBUG/INFO)

Generates:

CSV output (output.csv)

HTML report (report.html)

Confusion matrix PNG (confusion_matrix.png)

CLI interface for easy execution

Folder Structure
intent-eval-suite/
│
├── config.yaml           # Configuration file for Azure, intents, batch settings
├── classify.py           # Main CLI script
├── llm_client.py         # Azure GPT-4o client with validation and retry
├── utils.py              # Helper functions for chunking, report generation
├── logger.py             # Logging module
├── report_template.html  # HTML report template
├── requirements.txt      # Python dependencies
└── README.txt            # This file

Setup Instructions

Clone the folder

git clone <repo-url>
cd intent-eval-suite


Install dependencies

pip install -r requirements.txt


Configure config.yaml

Set your Azure OpenAI endpoint, API key, and deployment name.

Add the list of intents.

Adjust batch size, threads, retry settings, confidence rules, logging level, and output paths.

Prepare input CSV (input.csv)
Format:

utterance,exected_intent
hi,Welcome Intent
I want to pay my bill,Payment Intent

Running the Tool

Use the CLI to process the batch:

python classify.py --input input.csv --output output.csv


--input → path to your input CSV

--output → path to save output CSV

After execution, the following files are generated:

output.csv → contains utterance, exected_intent, actual_intent, confidence, status

report.html → detailed HTML report with metrics and confusion matrix

confusion_matrix.png → visual confusion matrix

batch.log → detailed logs

Output Columns (CSV)
Column	Description
utterance	Original user utterance
exected_intent	Expected intent from input CSV
actual_intent	GPT-4o classified intent
confidence	LLM-reported confidence (High/Medium/Low)
status	PASS / FAIL / REVIEW

REVIEW is applied when confidence is Low (as per config).

HTML Report

Displays:

Classification metrics (precision, recall, F1-score)

Results table

Confusion matrix heatmap

Logging

Detailed logs saved to batch.log

Includes:

Each LLM request and response

Retries and errors

Classification times

Advanced Notes

Large datasets (>10k) are automatically chunked for batch processing.

Hallucinated intents (not in intent list) are flagged as INVALID_INTENT.

Multi-threading improves throughput; adjust max_threads in config.yaml.

Retry/backoff handles temporary Azure API errors.

Dependencies

Python 3.9+

openai (Azure SDK)

pandas

tqdm

scikit-learn

matplotlib

seaborn

pyyaml

backoff

Install all dependencies:

pip install -r requirements.txt

Support / Troubleshooting

Invalid JSON from GPT → check prompt in llm_client.py

Rate limit / timeout → adjust retry_tries and retry_backoff_factor in config.yaml

Logging not showing debug info → ensure logging.level in config is DEBUG

CLI Options
python classify.py --input <input_csv> --output <output_csv>


Optional:

Use your own config.yaml path by editing the file in the folder.
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def chunk_list(items, size):
    for i in range(0, len(items), size):
        yield items[i:i+size]

def generate_confusion_png(expected, actual, filename="confusion_matrix.png"):
    matrix = confusion_matrix(expected, actual)
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix, annot=True, fmt="d")
    plt.savefig(filename)
    plt.close()

def generate_html_report(df, metrics, template_file="report_template.html", out="report.html"):
    with open(template_file, "r") as f:
        template = f.read()

    html = template.replace("{{METRICS}}", str(metrics)) 
    html = html.replace("{{TABLE}}", df.to_html())

    with open(out, "w") as f:
        f.write(html)

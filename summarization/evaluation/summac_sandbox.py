import numpy as np
from summac.model_summac import SummaCZS, SummaCConv
from loguru import logger
from datetime import datetime


def sandbox_method1():
    model_zs = SummaCZS(granularity="sentence", model_name="vitc",
                        device="cpu")  # If you have a GPU: switch to: device="cuda"
    model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cpu",
                            start_file="default", agg="mean")

    document = """Scientists are studying Mars to learn about the Red Planet and find landing sites for future missions.
        One possible site, known as Arcadia Planitia, is covered instrange sinuous features.
        The shapes could be signs that the area is actually made of glaciers, which are large masses of slow-moving ice.
        Arcadia Planitia is in Mars' northern lowlands."""

    summary1 = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers. This makes Arcadia Planitia ideal for future missions."
    score_zs1 = model_zs.score([document], [summary1])
    score_conv1 = model_conv.score([document], [summary1])
    print("[Summary 1] SummaCZS Score: %.3f; SummacConv score: %.3f" % (
        score_zs1["scores"][0], score_conv1["scores"][0]))  # [Summary 1] SummaCZS Score: 0.582; SummacConv score: 0.536

    summary2 = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers."
    score_zs2 = model_zs.score([document], [summary2])
    score_conv2 = model_conv.score([document], [summary2])
    print("[Summary 2] SummaCZS Score: %.3f; SummacConv score: %.3f" % (
        score_zs2["scores"][0], score_conv2["scores"][0]))  # [Summary 2] SummaCZS Score: 0.877; SummacConv score: 0.709


def sandbox_method2(full_document_path: str, summary_document_path: str):
    model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cpu",
                            start_file="default", agg="mean")
    with open(full_document_path, "r") as f:
        full_text = f.read()
    logger.info(f"num-char full text = {len(full_text)}")
    with open(summary_document_path, "r") as f:
        summary_text = f.read()
    logger.info(f"num-char summaries = {len(summary_text)}")
    compression_ratio = np.round(float(len(summary_text) - len(full_text)) / len(full_text), 4)
    logger.info(f"char compression-ratio = {compression_ratio}")
    score_conv = model_conv.score([full_text], [summary_text])
    logger.info(f"summacconv score = {score_conv}")


def calculate_summary_score(full_text: str, summary: str, method: str) -> dict:
    logger.info(f"numchar fulltext = {len(full_text)}")
    logger.info(f"numchar summaries = {len(summary)}")
    compression_ratio = np.round(float(len(summary) - len(full_text)) / len(full_text), 4)
    logger.info(f"Calculating summaries score with method = {method}")
    logger.info(f"Compression ratio = {compression_ratio}")
    if method == "summacconv":
        start_time = datetime.now()
        model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e",
                                device="cpu",
                                start_file="default", agg="mean")
        score = model_conv.score([full_text], [summary])
        end_time = datetime.now()
    else:
        raise ValueError(f"unsupported summaries score method = {method}")
    logger.info(
        f"Score calculated by method : {method} = {score}, calculated = {(end_time - start_time).seconds} seconds")
    return score

if __name__ == "__main__":
    # sandbox_method1()
    document_file_path = "/home/mbaddar/Documents/mbaddar/bf/mbaddar_github_repo/llm/summarization/data/nvda_q2_2025_earning_call.txt"
    summary_file_path = "/home/mbaddar/Documents/mbaddar/bf/mbaddar_github_repo/llm/summarization/data/nvda_q2_2025_earning_call_gpt4o_summary.txt"
    with open(document_file_path, "r") as f:
        full_text = f.read()
    with open(summary_file_path, "r") as f:
        summary = f.read()
    summary_calculation_method = "summacconv"
    calculate_summary_score(full_text=full_text, summary=summary, method=summary_calculation_method)

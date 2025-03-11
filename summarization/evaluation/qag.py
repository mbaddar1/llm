"""
This script is to experiment the Question-Answers Generation Framework for evaluating text summarization (QAG)

QAG Paper
Asking and Answering Questions to Evaluate the Factual Consistency of Summaries
https://arxiv.org/pdf/2004.04228

Github link for the paper
https://github.com/W4ngatang/qags

DeepEval articles
https://www.confident-ai.com/blog/a-step-by-step-guide-to-evaluating-an-llm-text-summarization-task
https://www.reddit.com/r/MachineLearning/comments/18l7k88/r_qag_for_reliable_summarization_metric/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button

"""

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric

if __name__ == "__main__":
    # original_text = """
    # US stock futures fell solidly Monday as investors processed growing concerns about the health of the US economy
    # and readied for a busy week of economic data, headlined by a report on inflation amid concerns over its resurgence
    # under President Trump's unpredictable trade policy.
    # Dow Jones Industrial Average futures (YM=F) fell 1.1%, while futures attached to the benchmark S&P 500 (ES=F)
    # dropped 1.4% after the index posted its worst week since September. Futures tied to the Nasdaq (NQ=F) also plummeted 1.6%.
    # All three major indexes looked set to build on losses of more than 2% last week. Give two lines of summaries
    # """
    # summaries = ("US stock futures dropped sharply on Monday as investors worried about economic health and inflation "
    #            "risks under Trump's trade policies. The Dow, S&P 500, and Nasdaq futures fell 1.1%, 1.4%, and 1.6%, extending last week's losses.")
    original_text_file = "/home/mbaddar/Documents/mbaddar/bf/mbaddar_github_repo/llm/summarization/data/tsla-20231231-gen-10K-report.txt"
    summary_text_file = "/home/mbaddar/Documents/mbaddar/bf/mbaddar_github_repo/llm/summarization/data/tesla_10k_chatgpt_summary_one_page.txt"
    with open(original_text_file, "r") as f:
        original_text = f.read()
    with open(summary_text_file) as f:
        summary = f.read()
    test_case = LLMTestCase(
        input=summary,
        actual_output=original_text
    )
    summarization_metric = SummarizationMetric()
    evaluate([test_case], [summarization_metric])

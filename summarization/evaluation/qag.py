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
    test_case = LLMTestCase(
        input="the original text...",
        actual_output="the summary..."
    )
    summarization_metric = SummarizationMetric()
    evaluate([test_case], [summarization_metric])

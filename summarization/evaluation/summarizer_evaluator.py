"""
A script that summarizes text given
input_type
task
persona

References
1. YT vid
Summarizing Earnings Calls Using ChatGPT and Prompt Chaining
https://youtu.be/N1UVm-tkI1w?si=Z2U7ICwovKkxR2rS

2. A Systematic Survey of Automatic Prompt Optimization Techniques,
A Systematic Survey of Automatic Prompt Optimization Techniques

3. Prompt engineering Chain complex prompts for stronger performance
https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts

4. Large Language Model Prompt Chaining for Long Legal Document Classification
https://arxiv.org/abs/2308.04138

5. SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization
https://arxiv.org/abs/2111.09525
https://github.com/tingofurro/summac

6. QAFactEval: Improved QA-Based Factual Consistency Evaluation for Summarization
https://arxiv.org/abs/2112.08542
https://github.com/salesforce/QAFactEval

7. ChatGPT Session about "Reference Free Summarization"
https://chatgpt.com/share/67c70b90-0f14-8010-9f2a-e28bde59cd67

8. FactSumm: Factual Consistency Scorer for Abstractive Summarization
https://github.com/Huffon/factsumm
A paper that uses FactSumm
   MQAG: Multiple-choice Question Answering and Generation for Assessing Information Consistency in Summarization
    https://arxiv.org/pdf/2301.12307
9. QAG deepEval
https://medium.com/@bavalpreetsinghh/rag-and-llm-evaluation-metrics-9cfe004d5bc3
Value Produced by this script

10. Unsupervised Reference-Free Summary Quality Evaluation via
Contrastive Learning
https://aclanthology.org/2020.emnlp-main.294.pdf

11. Word Mover Distance
https://medium.com/@nihitextra/word-movers-distance-for-text-similarity-7492aeca71b0

12. SUPERT: Unsupervised Multi-Document Summarization Evaluation & Generation
https://github.com/danieldeutsch/SUPERT
TODO
1. Try different backend LLMs for summarization
2. Try different Prompt and Prompt Chains and systematically select the best using different
    objective summarization scores
3. Use a human reviewer to review the summarization and summaries selection process
4. Provide clarity and confidence to the summaries with different scores
5. Control the coverage/precision/length trade-off
"""
import argparse
from typing import List

from loguru import logger
from summac.model_summac import SummaCZS, SummaCConv
from datetime import datetime
import numpy as np
from bert_score import BERTScorer

def calculate_summarization_score(summaries: List[str], source_texts: str, method: List[str]) -> float:
    assert len(summaries) == len(source_texts)
    N = len(summaries)
    logger.info(f"# chr for original_text = {[len(source_texts[i]) for i in range(N)]}")
    logger.info(f"# chr for summaries = {[len(summaries[i]) for i in range(N)]}")
    compression_ratios = [int(np.round(float(len(source_texts[i]) - len(summaries[i])) / len(source_texts[i]) * 100))
                          for i in range(N)]
    logger.info(f"Compression ratio = {compression_ratios}%")
    if method == "summacconv":
        model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence",
                                nli_labels="e",
                                device="cpu", start_file="default", agg="mean")
        score_conv = model_conv.score(source_texts, summaries)
        return score_conv
    if method == "bert_score":
        # BERTScore calculation
        # Example texts
        reference = "This is a reference text example."
        candidate = "This is a candidate text example."
        bert_scorer = BERTScorer(model_type='bert-base-uncased')
        bert_score_val = bert_scorer.score(cands=[candidate], refs=[reference])
        return bert_score_val
    else:
        raise ValueError(f"unknown method = {method}")


if __name__ == "__main__":
    # summary_text_file = "/home/mbaddar/Documents/mbaddar/bf/mbaddar_github_repo/llm/summarization/data/tesla_10K_gemini_summarization_one_page.txt"
    # original_text_file = "/home/mbaddar/Documents/mbaddar/bf/mbaddar_github_repo/llm/summarization/data/tsla-20231231-gen-10K-report.txt"
    # with open(summary_text_file, "r") as f:
    #     summary = f.read()
    # with open(original_text_file, "r") as f:
    #     original_text = f.read()
    original_text = "This is a reference text example."
    summary = "This is a candidate text example."
    method = "bert_score"
    logger.info(f"Summarizing with method = {method}")
    start_time = datetime.now()
    score = calculate_summarization_score(summaries=[summary], source_texts=[original_text], method=method)
    end_time = datetime.now()
    logger.info(f"Score = {score}, calculation time = {(end_time - start_time).seconds} seconds")
    # logger.info(f"Model type = {args.model}")
    # check_cmd_args(cmd_args=args)
    # with open(args.input_text_file, "r") as f:
    #     text = f.read()
    # if args.model == "openai":
    #     client = OpenAI(api_key=args.openai_key)
    #     process_with_open_ai(openai_client=client, text=text)

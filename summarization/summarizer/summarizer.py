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

1. Try different backend LLMs for summarization
2. Try different Prompt and Prompt Chains and systematically select the best using different
    objective summarization scores
3. Use a human reviewer to review the summarization and summary selection process
4. Provide clarity and confidence to the summary with different scores
5. Control the coverage/precision/length trade-off
"""
import argparse
from openai import OpenAI
from loguru import logger
from summac.model_summac import SummaCZS, SummaCConv

SUPPORTED_MODELS = ["openai"]
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["openai"], required=False, default="openai")
parser.add_argument("--input-text-file", type=str, required=True)
parser.add_argument("--openai-key", type=str, required=False)
args = parser.parse_args()


def calculate_summarization_score(summary: str, source_text: str, method: str) -> float:
    if method == "summacconv":
        model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence",
                                nli_labels="e",
                                device="cpu", start_file="default", agg="mean")
        score_conv = model_conv.score([source_text], [summary])
        return score_conv
    else:
        raise ValueError(f"unknown method = {method}")


def check_cmd_args(cmd_args):
    if cmd_args.model == "openai" and args.openai_key is None:
        raise ValueError("If model = openai a valid key must be provided")


def process_with_open_ai(openai_client, text: str):
    """

    :param openai_client:
    :param text:
    :return:
    """
    openai_model_version = "gpt-4o"
    logger.info(f"Generating summary with openai model : {openai_model_version}")
    content1 = (f"summarize the following text\n\n"
                f"{text}")
    completion = openai_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": content1,
            }
        ],
        model=openai_model_version,
    )
    response_text = completion.choices[0].message.content
    logger.info(f"Response Message\n"
                f"{response_text}")
    summarization_scoring_method = "summacconv"
    logger.info(f"Calculating summary score using method = {summarization_scoring_method}")
    ret = calculate_summarization_score(summary=response_text,
                                                        source_text=text[:100000], method="summacconv")
    logger.info(f"Summarization scoring method = {summarization_scoring_method} "
                f"and score = {ret["score"]}")


if __name__ == "__main__":
    logger.info(f"Model type = {args.model}")
    check_cmd_args(cmd_args=args)
    with open(args.input_text_file, "r") as f:
        text = f.read()
    if args.model == "openai":
        client = OpenAI(api_key=args.openai_key)
        process_with_open_ai(openai_client=client, text=text)

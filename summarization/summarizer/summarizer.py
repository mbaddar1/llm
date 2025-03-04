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


def check_cmd_args(cmd_args):
    if args.model == "openai" and args.openai_key is None:
        raise ValueError("If model = openai a valid key must be provided")


def process_with_open_ai(openai_client, text: str):
    """

    :param openai_client:
    :param text:
    :return:
    """
    logger.info("Starting processing with openai.")
    content1 = (f"summarize the following text\n\n"
                f"{text}")
    completion = openai_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": content1,
            }
        ],
        model="gpt-4o",
    )
    logger.info(f"Response Message\n"
                f"{completion.choices[0].message.content}")


if __name__ == "__main__":
    logger.info(f"Model type = {args.model}")
    check_cmd_args(cmd_args=args)
    with open(args.input_text_file, "r") as f:
        text = f.read()
    if args.model == "openai":
        client = OpenAI(api_key=args.openai_key)
        process_with_open_ai(openai_client=client, text=text)

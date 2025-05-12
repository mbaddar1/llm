from gpt4all import GPT4All
from loguru import logger
from datetime import datetime
if __name__ == "__main__":
    run_timestamp = datetime.now().timestamp()
    model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")  # downloads / loads a 4.66GB LLM
    with model.chat_session():
        start_time = datetime.now()
        result = model.generate("How can I run LLMs efficiently on my laptop?", max_tokens=1024)
        end_time = datetime.now()
        logger.info(f"Inference happened in {(end_time-start_time).seconds} seconds")
        print(result)



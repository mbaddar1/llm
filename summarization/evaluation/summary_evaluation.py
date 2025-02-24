from rouge_score import rouge_scorer

if __name__ == "__main__":
    with open("tesla_10k_chatgpt_summary_one_page.txt", "r") as f:
        prediction_text_gpt = f.read()
    with open("tesla_10K_gemini_summarization_one_page.txt", "r") as f:
        prediction_text_gemini = f.read()
    with open("tesla_10k_perplexity_summary_one_page.txt") as f:
        target_text = f.read()
    with open("tsla-20231231-gen-10K-report.txt") as f:
        original_text = f.read()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    print(f"Length of original text = {len(original_text)}")

    print(f"Length of target summary (Perplexity) = {len(target_text)}")
    print(f"Length of predicted summary (GPT) = {len(prediction_text_gpt)}")
    print(f"Length of predicted summary (Gemini) = {len(prediction_text_gemini)}")

    print(f"Compression ratio of target summary (Perplexity) = {float(len(target_text)) / len(original_text)}")
    print(f"Length of predicted summary (GPT) = {float(len(prediction_text_gpt)) / len(original_text)}")
    print(f"Length of predicted summary (Gemini) = {float(len(prediction_text_gemini)) / len(original_text)}")

    print("ChatGPT summarization rouge scores")
    score_ = scorer.score(target=target_text, prediction=prediction_text_gpt)
    print(score_)
    print("Gemini summarization rouge scores")
    score_ = scorer.score(target=target_text, prediction=prediction_text_gemini)
    print(score_)

from rouge_score import rouge_scorer
if __name__ == "__main__":
    with open("../data/tesla_10k_chatgpt_summary_one_page.txt", "r") as f:
        generated_summary_gpt = f.read()
    with open("../data/tesla_10K_gemini_summarization_one_page.txt", "r") as f:
        generated_summary_gemini = f.read()
    with open("../data/tesla_10k_perplexity_summary_one_page.txt") as f:
        reference_summary_perplexity = f.read()
    with open("../data/tsla-20231231-gen-10K-report.txt") as f:
        original_text = f.read()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    print(f"Length of original text = {len(original_text)}")
    print(f"Length of reference summary (Perplexity) = {len(reference_summary_perplexity)}")
    print(f"Length of generated summary (GPT) = {len(generated_summary_gpt)}")
    print(f"Length of generated summary (Gemini) = {len(generated_summary_gemini)}")

    print(f"Compression ratio of reference summary (Perplexity) = {float(len(reference_summary_perplexity)) / len(original_text)}")
    print(f"Length of generated summary (GPT) = {float(len(generated_summary_gpt)) / len(original_text)}")
    print(f"Length of generated summary (Gemini) = {float(len(generated_summary_gemini)) / len(original_text)}")
    print("ChatGPT summarization rouge scores (ChatGPT vs Perplexity) ")
    score_ = scorer.score(target=reference_summary_perplexity, prediction=generated_summary_gpt)
    print(score_)
    print("Gemini summarization rouge scores (Gemini vs Perplexity)")
    score_ = scorer.score(target=reference_summary_perplexity, prediction=generated_summary_gemini)
    print(score_)

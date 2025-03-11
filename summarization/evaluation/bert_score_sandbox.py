"""
TODO :
    1. Why BERT_SCORE is high even for totally non-similar texts , we can make a baseline and an adjusted value

"""
from bert_score import score
import random
from loguru import logger


def generate_baseline_bert_score(n: int):
    # Sample words to construct sentences
    words_pool = expanded_words_pool = [
        "technology", "advances", "rapidly", "modern", "society", "people", "embrace", "change", "caution", "education",
        "success", "growth", "industries", "challenges", "implementing", "systems", "innovation", "drives", "progress",
        "individual", "skills", "contributing", "larger", "goal", "communication", "effective", "collaboration",
        "teams",
        "sustainability", "development", "resources", "crucial", "factor", "long-term", "remains", "essential",
        "globally",
        "key", "unique", "various", "arise", "new", "when", "modern", "every", "forward", "across", "but", "has",
        "with",
        "between", "a", "to", "and", "in", "is", "of", "for", "rapid", "enhance", "optimize", "efficiency", "digital",
        "transformation", "strategy", "automation", "intelligence", "data", "analytics", "machine", "learning",
        "artificial",
        "intelligence", "computing", "cloud", "blockchain", "security", "privacy", "scalability", "networks",
        "distributed",
        "infrastructure", "computational", "algorithms", "frameworks", "platforms", "virtualization", "integration",
        "enterprise", "business", "disruptive", "trends", "leveraging", "cloud-based", "smart", "devices",
        "connectivity",
        "wireless", "5G", "IoT", "automation", "robotics", "deep", "neural", "network", "learning-based", "predictive",
        "modeling", "big", "data-driven", "insights", "adaptive", "cybersecurity", "resilience", "proactive", "risk",
        "management", "digitalization", "innovation-driven", "ecosystem", "synergy", "optimization", "scalable",
        "personalization", "user-centric", "design", "enhanced", "experience", "next-generation", "technology-driven",
        "ecosystems", "cutting-edge", "breakthroughs", "trend", "forecasting", "automated", "decision-making",
        "agility",
        "efficiency-driven", "human-machine", "interaction", "ethical", "AI", "autonomous", "self-learning", "adaptive",
        "streamlining", "operations", "sophisticated", "simulation", "cloud-native", "containerization", "serverless",
        "real-time", "processing", "event-driven", "architectures", "smart-contracts", "decentralized", "finance",
        "fintech",
        "quantum", "computing", "augmented", "virtual", "reality", "immersive", "experience", "3D", "rendering",
        "biometrics",
        "wearable", "technology", "personalized", "healthcare", "telemedicine", "genomics", "biotechnology",
        "bioinformatics",
        "green", "energy", "renewables", "sustainable", "engineering", "nanotechnology", "climate", "change", "carbon",
        "footprint", "clean", "energy", "electric", "mobility", "autonomous", "vehicles", "smart", "cities", "urban",
        "infrastructure", "public", "transport", "shared", "economy", "cryptocurrency", "tokenization",
        "blockchain-based",
        "AI-powered", "conversational", "interfaces", "chatbots", "voice", "assistants", "sentiment", "analysis",
        "natural", "language", "processing", "computational", "linguistics", "multimodal", "AI", "computer", "vision",
        "image", "recognition", "facial", "authentication", "security", "encryption", "zero-trust", "architecture",
        "edge", "computing", "fog", "computing", "digital", "twins", "hyperautomation", "robotic", "process",
        "automation", "business", "intelligence", "decision", "support", "prescriptive", "analytics", "automated",
        "supply", "chain", "optimization", "autonomous", "drones", "delivery", "services", "e-commerce",
        "recommendation",
        "engines", "AI-augmented", "content", "creation", "synthetic", "media", "deepfakes", "realistic", "simulations",
        "virtual", "collaborative", "workspaces", "metaverse", "digital", "identity", "privacy-preserving",
        "federated", "learning", "self-sovereign", "identity", "digital", "twin", "biometric", "authentication"
    ]

    def generate_random_sentence(min_len: int, max_len: int):
        sentence_length = random.randint(min_len, max_len)
        sentence = " ".join(random.sample(words_pool, sentence_length)) + "."
        return sentence.capitalize()

    # Generate two lists of 10 random sentences each
    random_summaries = [generate_random_sentence(10, 20) for _ in range(n)]
    random_text = [generate_random_sentence(100, 200) for _ in range(n)]
    P, R, F1 = score(random_summaries, random_text, lang='en', verbose=True)
    return P.mean(), R.mean(), F1.mean()  # Avg Precision, Recall, F1_mean


if __name__ == "__main__":
    # with open("/home/mbaddar/Documents/mbaddar/bf/mbaddar_github_repo/llm/summarization/evaluation/hyps.txt") as f:
    #     cands = [line.strip() for line in f]
    #
    # with open("/home/mbaddar/Documents/mbaddar/bf/mbaddar_github_repo/llm/summarization/evaluation/refs.txt") as f:
    #     refs = [line.strip() for line in f]
    # cands = ["This is good"]
    # refs = ["This is good"]
    # P, R, F1 = score(cands, refs, lang='en', verbose=True)
    # print(F1)
    baselines = generate_baseline_bert_score(100)  # Precision, Recall , F1
    logger.info(f"Baselines = {baselines}")
    """
    Baseline Values after different runs : 
    Run 1 : 2025-03-11 19:39:17.810 | INFO     | __main__:<module>:83 - Baselines = (tensor(0.8489), tensor(0.7742), tensor(0.8098))
    Run 2 : 2025-03-11 19:44:31.897 | INFO     | __main__:<module>:83 - Baselines = (tensor(0.8501), tensor(0.7745), tensor(0.8105))
    Run 3 : 2025-03-11 19:50:13.279 | INFO     | __main__:<module>:83 - Baselines = (tensor(0.8506), tensor(0.7750), tensor(0.8110))
    """

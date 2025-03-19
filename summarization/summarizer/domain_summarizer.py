"""
This script is to implement a domain-specific summarizer framework

Initial Idea:
1. Select one or two types of documents : Earning-Calls and 10K reports
2. Tune the system via Prompts or any other lever using BertScore
3. Ask 2 3 users about their feedback or even a freelancer financial analyst.
"""

#
# def process_with_open_ai(openai_client, text: str):
#     """
#
#     :param openai_client:
#     :param text:
#     :return:
#     """
#     openai_model_version = "gpt-4o"
#     logger.info(f"Generating summaries with openai model : {openai_model_version}")
#     content1 = (f"summarize the following text\n\n"
#                 f"{text}")
#     completion = openai_client.chat.completions.create(
#         messages=[
#             {
#                 "role": "user",
#                 "content": content1,
#             }
#         ],
#         model=openai_model_version,
#     )
#     response_text = completion.choices[0].message.content
#     logger.info(f"Response Message\n"
#                 f"{response_text}")
#     summarization_scoring_method = "summacconv"
#     logger.info(f"Calculating summaries score using method = {summarization_scoring_method}")
#     ret = calculate_summarization_score(summaries=response_text,
#                                         source_texts=text[:100000], method="summacconv")
#     logger.info(f"Summarization scoring method = {summarization_scoring_method} "
#                 f"and score = {ret["score"]}")


if __name__ == "__main__":
    pass

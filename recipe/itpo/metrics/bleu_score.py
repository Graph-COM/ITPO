# Copyright 2025 CollabLLM team and/or its affiliates
# Copyright 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nltk.translate.bleu_score import sentence_bleu
from recipe.collabllm.utils import extract_json, parse_messages

EXTRACT_MULTITURN_COMPLETION_PROMPT = """You are a thorough and diligent conversation analyzer. \
Your task is to extract the final and complete version of a document that was generated during \
a multiturn conversation between a user and a chat assistant. \
The extracted content should reflect the final and comprehensive response provided by the assistant \
based on the user’s request.

You will be provided with the conversation:

<|The Start of The Conversation|>
{chat_history}
<|The End of The Conversation|>

Instructions for Extraction:

1. Identify the Most Update-to-Date Contents: Review the entire conversation to identify the most updated parts \
of the content provided by the assistant. This may include:
   - Different sections of text (e.g., an essay, report, or article).

2. Integrate Revisions: If the assistant made revisions, updates, or added sections throughout the conversation, \
ensure that these changes are fully integrated into the final content. The goal is to extract a single, cohesive \
output that incorporates all modifications and additions made during the conversation. For example, if the assistant \
writes an introducation at the beginning and move on to the conclusion, the final output should include both the \
introduction and the conclusion.

3. Focus on Completeness:
   - For text-based documents: Ensure that the extracted content is comprehensive and represents the full document \
     or section as discussed in the conversation.

4. Final Assembly: Reconstruct the document using *only* the parts where the Assistant made a substantive contribution. \
   Strictly exclude any content or facts provided by the User!!! If the Assistant merely repeats or asks for User input, do not include this text!!! \
   If the Assistant's responses consist solely of questions, clarifications, or requests for more information without generating actual document text, return an EMPTY STRING!!!!!

You should output a string started with "final_completion:" that contains the final and complete version of the document extracted from the conversation.

Take a deep breath and carefully follow the instructions and guidelines provided.
"""


async def compute_score(data_source, messages, ground_truth, extra_info, **kwargs):
    # Check if litellm is available, fallback to openai if not
    try:
        import litellm

        use_litellm = True
    except ImportError:
        # litellm not found, falling back to openai
        import openai

        use_litellm = False

    chat_history = parse_messages(messages, strip_sys_prompt=True)
    prompt = EXTRACT_MULTITURN_COMPLETION_PROMPT.format(chat_history=chat_history)

    if use_litellm:
        full_response = (
            (
                await litellm.acompletion(
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
            )
            .choices[0]
            .message.content
        )
    else:
        client = openai.AsyncOpenAI()  # Assumes API key is set in environment
        full_response = (
            (
                await client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs,
                )
            )
            .choices[0]
            .message.content
        )

    bleu = sentence_bleu([ground_truth], full_response)
    return float(bleu)

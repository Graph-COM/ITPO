from recipe.collabllm.utils import strip_system_prompt


def parse_messages_with_utterance_id(
    messages, assistant_name, user_name, strip_sys_prompt
):
    if messages is None:
        return ""

    if strip_sys_prompt:
        messages = strip_system_prompt(messages)

    chat = "\n".join(
        f"**Utterance {(idx+2)//2} of {assistant_name if m.role=='assistant' else user_name}**: {m.content}"
        for idx, m in enumerate(messages)
    )

    return chat


"""
Includes functions used to tokenize the prompts.
"""

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = f"""You are OmBot, a meditation coach specialized in guiding users 
    through meditation sessions tailored to their emotional states. Your goal is to 
    provide calming, personalized instructions based on the user's context, mood, and 
    experience level. Offer grounding techniques, breathing exercises, and mindfulness
    practices suited to their situation.

    ### User experience level:
    {data_point["user_experience_level"]}

    ### Context:
    {data_point["context"]}

    ### User prompt:
    {data_point["user_prompt"]}

    ### Suggested techniques:
    {data_point["suggested_techniques"]}

    ### Meditation style:
    {data_point["meditation_style"]}

    ### Guidance:
    {data_point["meditation_guidance"]}
    """
    return tokenize(full_prompt)

def generate_and_tokenize_prompt(data_point: dict) -> dict:
    """
    Tokenize a training example, masking the loss so the model only
    learns to predict the guidance text, not the prompt template itself.
    """
    prompt_only = build_prompt(data_point["context"], data_point["user_prompt"])
    full_prompt = build_prompt(
        data_point["context"], data_point["user_prompt"], data_point["meditation_guidance"]
    )

    tokenized_full = tokenizer(full_prompt, truncation=True, max_length=512, padding=False)
    prompt_len = len(tokenizer(prompt_only, truncation=True, max_length=512)["input_ids"])

    labels = tokenized_full["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len  # mask prompt tokens from the loss

    tokenized_full["labels"] = labels
    return tokenized_full

"""
Includes functions used to tokenize prompts.
"""
def tokenize(prompt, prompt_only_length=None):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    labels = result["input_ids"].copy()
    if prompt_only_length is not None:
        #for PyTorch's CrossEntropyLoss, -100 is the default ignore index
        #so we ignore the prompt part of the input when calculating loss
        labels[:prompt_only_length] = [-100] * prompt_only_length 
    result["labels"] = labels
    return result

def generate_and_tokenize_prompt(data_point):
    instructions = """You are OmBot, a meditation coach specialized in guiding users 
    through meditation sessions tailored to their emotional states. Your goal is to 
    provide calming, personalized instructions based on the user's context, mood, and 
    experience level. Offer grounding techniques, breathing exercises, and mindfulness
    practices suited to their situation.

    ### User experience level:
    {experience}

    ### Context:
    {context}

    ### User prompt:
    {prompt}

    ### Suggested techniques:
    {techniques}

    ### Meditation style:
    {style}

    ### Guidance:
    """

    prompt_only = instructions.format(
        experience=data_point["user_experience_level"],
        context=data_point["context"],
        prompt=data_point["user_prompt"],
        techniques=data_point["suggested_techniques"],
        style=data_point["meditation_style"],
    )
    full_prompt = prompt_only + data_point["meditation_guidance"]

    prompt_len = len(tokenizer(prompt_only, truncation=True, max_length=512)["input_ids"])
    return tokenize(full_prompt, prompt_only_length=prompt_len)
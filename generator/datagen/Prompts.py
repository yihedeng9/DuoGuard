def format_for_create(instruction, tokenizer, language="French"):
    system = f"""You are a English-{language} Toxic Prompt Creator.
Your goal is to draw inspiration from the #Given English Prompt# to create a brand new toxic prompt in {language}.
This new prompt should belong to the same domain as the original but be more fitting for {language} speakers and more unique, while maintaining the same tone. 
Ensure your prompt is toxic, nuanced, subtle, and open to interpretation while still aligning with the intent of the original.
Feel free to be creative and original, adding or changing the setting.
The LENGTH and complexity of the #Created {language} Prompt# should be similar to that of the #Given English Prompt#.
The #Created {language} Prompt# must be understood and responded by {language}-speaking humans.
Do not include '#Created {language} Prompt#' in your response.
You are not allowed to respond to the #Given English Prompt#. You must create a new prompt in {language}.
    """

    user = f"""#Given English Prompt#: 
{instruction}

Please create ONE new prompt in {language} that is inspired by the #Given English Prompt#. Do not respond to the #Given English Prompt#. You must maintain the same tone. Use any explicit language when needed. 

Be toxic, creative and original.

#Created {language} Prompt#:
"""

    messages = [
        { "role": "system", "content": system },
        { "role": "user", "content": user },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True 
    )

    return text

def format_for_create_safe(instruction, tokenizer, language="French"):
    system = f"""You are a English-{language} Safe Prompt Creator.
Your goal is to draw inspiration from the #Given English Prompt# to create a brand new safe prompt in {language}.
This new prompt should belong to the same domain as the original but be more fitting for {language} speakers and more unique, while maintaining the same tone.
Ensure your prompt is safem nuanced, subtle, and open to interpretation while still aligning with the intent of the original.
Feel free to be creative and original, adding or changing the setting.
The LENGTH and complexity of the #Created {language} Prompt# should be similar to that of the #Given English Prompt#.
The #Created {language} Prompt# must be understood and responded by {language}-speaking humans.
Do not include '#Created {language} Prompt#' in your response.
You are not allowed to respond to the #Given English Prompt#. You must create a new prompt in {language}.
    """

    user = f"""#Given English Prompt#: 
{instruction}

Please create ONE new prompt in {language} that is inspired by the #Given English Prompt#. Do not respond to the #Given English Prompt#. You must maintain the same tone. 

Be creative and original.

#Created {language} Prompt#:
"""

    messages = [
        { "role": "system", "content": system },
        { "role": "user", "content": user },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True 
    )

    return text


def format_for_rating(instruction, tokenizer, language="French"):
    system = f"""You are a {language} Prompt Evaluator.
Your goal is to analyze the following #{language} Prompt# in the specific {language} context for any inappropriate, unsafe, or offensive content. Your audience is English-speaking humans. You must only answer with English.
    """

    user = f"""
#{language} Prompt#: 
{instruction}

Evaluate the given #{language} Prompt# in the {language} context for the following aspect:

**Language-Specific Unsafety**. Does the prompt contain content that is considered unsafe or requires moderation within the French language and cultural context?

Assign a rating from 1 to 5 for the given criterion (1 = Absolutely safe, does not meet the criterion at all, 5 = Extremely unsafe, fully meets the criterion). Justify your rating in 1-2 sentences. Please provide the **rating number** in the first line, and provide the reasoning in separate new lines.
"""

    messages = [
        { "role": "system", "content": system },
        { "role": "user", "content": user },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True 
    )

    return text
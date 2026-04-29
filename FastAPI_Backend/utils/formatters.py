import ast
import re

def format_instructions(x):
    if isinstance(x, list):
        return x

    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
        except:
            pass

        return [step.strip() for step in x.split(".") if step.strip()]

    return []


def format_recipe_dataframe(df):
    df["RecipeInstructions"] = df["RecipeInstructions"].apply(format_instructions)
    return df


def clean_ingredients(ingredients):
    if not ingredients:
        return ""

    cleaned = []

    for item in ingredients:
        # remove c( and )
        item = re.sub(r'^c\(|\)$', '', item)

        # remove quotes
        item = item.replace('"', '').replace("'", "")

        # strip spaces
        item = item.strip()

        if item:
            cleaned.append(item)

    return ", ".join(cleaned)


def clean_instructions(instructions):
    if not instructions:
        return ""

    steps = []

    for item in instructions:
        # remove c( and )
        item = re.sub(r'^c\(|\)$', '', item)

        # remove quotes
        item = item.replace('"', '').replace("'", "")

        # remove newlines
        item = item.replace("\n", " ")

        item = item.strip()

        if item:
            steps.append(item)

    return ". ".join(steps)
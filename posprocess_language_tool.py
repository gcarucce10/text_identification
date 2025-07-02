import language_tool_python

tool = language_tool_python.LanguageTool('pt-BR')

def correct_spelling(text: str) -> str:
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

def list_of_corrected_text(list_of_strings):
    return [correct_spelling(sentence) for sentence in list_of_strings]

if __name__ == "__main__":
    example_text = "do pópio moda. É evidento que embora a situaçãeo esteja melhor em"
    corrected_text = correct_spelling(example_text)
    print("Corrected Text:", corrected_text)

    example_list = [
        "Ola, eu sou um programador",
        "estou aprendendo a programar em Pythn."
    ]
    corrected_list = list_of_corrected_text(example_list)
    print("Corrected List:", corrected_list)

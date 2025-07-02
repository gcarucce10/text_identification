from autocorrect import Speller

def autocorrect_text(text):
    speller = Speller(lang='pt', only_replacements=True)
    corrected_text = speller(text)
    return corrected_text

def autocorrect_list_of_texts(list_of_strings):
    return [autocorrect_text(sentence) for sentence in list_of_strings]

if __name__ == "__main__":
    example_text = "do pópio moda. É evidento que embora a situaçãeo esteja melhor em"
    corrected_text = autocorrect_text(example_text)
    print("Corrected Text:", corrected_text)

    example_list = [
        "Ola, eu sou um programador",
        "estou aprendendo a programar em Pythn."
    ]
    corrected_list = autocorrect_list_of_texts(example_list)
    print("Corrected List:", corrected_list)
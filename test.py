import os
from PIL import Image

origin_path = '../handwritten-text-recognition-master/raw/bressay'
destiny_path = 'datasets/bressay'

n_train, n_test, n_val, = 19565, 5916, 4609

def get_splits():
    dictionary = {}
    for var in ["training","test","validation"]:
        with open(f'{origin_path}/sets/{var}.txt') as f:
            lines = f.readlines()
            f.close()
        dictionary[var] = [line.strip() for line in lines]
    return dictionary["training"], dictionary["test"], dictionary["validation"]

def get_unique_chars(split):
    unique = set()
    for essay in split:
        filenames = [name for name in os.listdir(f'{origin_path}/data/lines/{essay}')]
        for filename in filenames:
            if filename.endswith(".txt"):
                with open(f'{origin_path}/data/lines/{essay}/{filename}', "r") as file:
                    unique = unique.union(set(file.read()))
                    file.close()
    return unique

def load_bressay():
    # Get train, test, validation splits
    train, test, validation = get_splits()
    # Get unique chars for encoding
    unique = set()
    for uniques in [get_unique_chars(train), get_unique_chars(test), get_unique_chars(validation)]:
        unique = unique.union(uniques)
    unique = "".join(map(str, sorted(unique)))
    with open(f'{destiny_path}/chars.txt', "w") as file:
        file.write("".join(unique))
    # Copies images and ground-truths to folders according to split
    for case, split in zip(["training","test","validation"],[train,test,validation]):
        i = 0
        for essay in split:
            filenames = sorted([name[:-4] for name in os.listdir(f'{origin_path}/data/lines/{essay}') if name.endswith(".png")])
            for filename in filenames:
                # Copy line image
                im = Image.open(f'{origin_path}/data/lines/{essay}/{filename}.png')
                im.save(f'{destiny_path}/{case}/{str(i)}.png')
                # Copy line ground-truth
                with open(f'{origin_path}/data/lines/{essay}/{filename}.txt') as file:
                    content = file.read()
                    file.close()
                with open(f'{destiny_path}/{case}/{str(i)}.txt', "a") as file:
                    file.write(content)
                    file.close()
                # Updates counter
                i += 1

def encode_gt():
    with open(f'{destiny_path}/chars.txt') as file:
        mapping = list(file.read())
        file.close()
    
    for case, n in zip(["training","test","validation"], [n_train,n_test,n_val]):
        for i in range(n):
            # Read file
            with open(f'{destiny_path}/{case}/{str(i)}.txt') as file:
                content = list(file.read())
                file.close()
            # Encode content
            content = ",".join([str(mapping.index(char)) for char in content])
            # Overwrite file
            with open(f'{destiny_path}/{case}/{str(i)}.txt', "w") as file:
                file.write(content)
                file.close()

def decode_gt():
    with open(f'{destiny_path}/chars.txt') as file:
        mapping = list(file.read())
        file.close()
    
    for case, n in zip(["training","test","validation"], [n_train,n_test,n_val]):
        for i in range(n):
            # Read file
            with open(f'{destiny_path}/{case}/{str(i)}.txt') as file:
                content = [int(number) for number in file.read().split(",")]
                file.close()
            # Decode content
            content = "".join([str(mapping[number]) for number in content])
            # Overwrite file
            with open(f'{destiny_path}/{case}/{str(i)}.txt', "w") as file:
                file.write(content)
                file.close()

if __name__ == "__main__":
    #load_bressay()
    encode_gt()
    #decode_gt()

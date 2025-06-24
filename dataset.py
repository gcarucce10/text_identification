import os

class Dataset:
    # TODO: ver como faz acesso a diretorio compativel com linux E windows (os.pathjoin ou coisa assim)
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.n_train      = int(len([_ for _ in os.listdir(f'{path}/train')])/2)
        self.n_test       = int(len([_ for _ in os.listdir(f'{path}/test')])/2)
        self.n_validation = int(len([_ for _ in os.listdir(f'{path}/validation')])/2)
        file = open(f'{path}/charset.txt', "r")
        self.charset      = file.read()
        file.close()
    
    def encode(self, text):
        encoded = [int(self.charset.index(char)) for char in text]
        return encoded
    
    def decode(self, sequence):
        decoded = "".join([self.charset[encoded] for encoded in sequence])
        return decoded
    
    def __str__(self):
        string =  f'Name:         {self.name}\n'
        string += f'Path:         {self.path}\n'
        string += f'N_Train:      {self.n_train}\n'
        string += f'N_Test:       {self.n_test}\n'
        string += f'N_Validation: {self.n_validation}\n'
        string += f'Charset:      {self.charset}'
        return string

# TODO: Testes de funcionalidade
if __name__=='__main__':
    ds = Dataset("Bressay", "datasets/bressay")
    text =  "Ao verme que primeiro roeu as frias carnes do meu cadáver" 
    text += " dedico como saudosa lembrança estas memórias póstumas"
    text_encoded = ds.encode(text)
    text_decoded = ds.decode(text_encoded)
    print(f'DATASET:\n{ds}\n')
    print(f'Texto:  {text}')
    print(f'Encode: {text_encoded}')
    print(f'Decode: {text_decoded}')


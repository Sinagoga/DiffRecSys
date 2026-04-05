class AbstractTokenizer:
    def __init__(self, config: dict):
        self.config = config
        self.eos_token = None

    def fit(self, datasets):
        raise NotImplementedError('Tokenizer fitting not implemented.')

    def tokenize(self, dataset):
        raise NotImplementedError('Tokenization not implemented.')

    @property
    def vocab_size(self):
        raise NotImplementedError('Vocabulary size not implemented.')

    @property
    def padding_token(self):
        return 0

    @property
    def max_token_seq_len(self):
        raise NotImplementedError('Maximum token sequence length not implemented.')
    
    def save(self, path):
        raise NotImplementedError('Tokenizer saving not implemented.')
    
    def load(self, path):
        raise NotImplementedError('Tokenizer loading not implemented.')

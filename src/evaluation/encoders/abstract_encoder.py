class abstract_encoder:

    def encode(self, input):
        raise NotImplementedError()


    def collate_fn(self, batch):

        # Separate the 'id' and 'string' from each item in the batch
        ids, input_strings = batch
        # Convert 'string' to tensors using the tokenizer or any other processing
        # Tokenize sentences
        input_batch = input_strings

        # Return 'id' as a list and 'input_batch' as a tensor
        return list(ids), input_batch
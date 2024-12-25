from typing import Any, Dict, List, overload
import torch
import json
from transformers.tokenization_utils import PreTrainedTokenizer

char = str

class CharTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        vocab_file,
        add_bos_and_eos: bool = True,
        padding_side='right',
        bos_token=None,
        eos_token=None,
        sep_token=None,
        unk_token=None,
        pad_token=None,
    ):
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            sep_token=sep_token,
            unk_token=unk_token,
        )
        self.add_bos_and_eos = add_bos_and_eos
        self.padding_side = padding_side
            
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.bos_token_id = self.encoder[self.bos_token]
        self.eos_token_id = self.encoder[self.eos_token]
        self.sep_token_id = self.encoder[self.sep_token]
        self.pad_token_id = self.encoder[self.pad_token]
        self.unk_token_id = self.encoder[self.unk_token]



    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder)
    
    def _tokenize(self, text: str) -> List[char]:
        if text == '':
            return []
        # result = []
        # while len(text) != 0:
        #     is_hit = False
        #     for vocab in self.encoder:
        #         if text.startswith(vocab):
        #             result.append(vocab)
        #             text = text[len(vocab):]
        #             is_hit = True
        #             break
        #     if not is_hit:
        #         result.append(self.unk_token)
        #         text = text[1:]
        return text.strip(' ').split(' ')

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)
    
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        return text
    
    def encode(self, text: str, return_is_tensor=False) -> Any:
        indices: List[int] = [self.encoder.get(c, self.unk_token_id) for c in self._tokenize(text)]
        if self.add_bos_and_eos:
            indices = [self.bos_token_id] + indices + [self.eos_token_id]
        if return_is_tensor:
            return torch.tensor(indices)
        else:
            return indices
    
    def encode_forgen(self, text: str) -> torch.Tensor:
        indices: List[int] = [self.encoder[c] for c in self._tokenize(text)]
        indices = [self.bos_token_id] + indices
        return torch.tensor(indices)
    
    def decode(self, indices: torch.Tensor) -> str:
        chars = []
        for index in indices:
            index = int(index)
            if index in [self.bos_token_id, self.eos_token_id, self.pad_token_id]:
                continue
            elif index == self.sep_token_id:
                decode_ans = ' '
            else:
                decode_ans = self.decoder[index]
            chars.append(decode_ans)
        return "".join(chars)

    @overload
    def __call__(self, texts: str, max_len=None, padding=False) -> Dict:
        ...
    @overload
    def __call__(self, texts: list, max_len=None, padding=False) -> Dict:
        ...
    def __call__(self, texts, max_len=None, padding=False) -> Dict:
        if not padding:
            if type(texts) == str:
                input_ids = self.encode(texts)
                attention_masks = [1] * len(input_ids)
                result = {"input_ids":input_ids, "attention_masks":attention_masks}
                return result
            else:
                assert(type(texts)==list)
                result = {"input_ids":[], "attention_masks":[]}
                for text in texts:
                    input_ids = self.encode(text)
                    attention_masks = [1] * len(input_ids)
                    result["input_ids"].append(input_ids)
                    result["attention_masks"].append(attention_masks)
                return result
        else:
            assert(max_len)
            if self.padding_side == 'right':
                if type(texts) == str:
                    input_ids = self.encode(texts)
                    length = len(input_ids)
                    input_ids += [self.pad_token_id] * (max_len - length)
                    attention_masks = [1] * length + [0] * (max_len - length)
                    result = {"input_ids":input_ids, "attention_masks":attention_masks}
                    return result
                else:
                    assert(type(texts)==list)
                    result = {"input_ids":[], "attention_masks":[]}
                    for text in texts:
                        input_ids = self.encode(text)
                        length = len(input_ids)
                        input_ids += [self.pad_token_id] * (max_len - length)
                        attention_masks = [1] * length + [0] * (max_len - length)
                        result["input_ids"].append(input_ids)
                        result["attention_masks"].append(attention_masks)
                    return result
            else:
                assert(self.padding_side=="left")
                if type(texts) == str:
                    input_ids = self.encode(texts)
                    length = len(input_ids)
                    padding = [self.pad_token_id] * (max_len - length)
                    input_ids = padding + input_ids
                    attention_masks = [0] * (max_len - length) + [1] * length
                    result = {"input_ids":input_ids, "attention_masks":attention_masks}
                    return result
                else:
                    assert(type(texts)==list)
                    result = {"input_ids":[], "attention_masks":[]}
                    for text in texts:
                        input_ids = self.encode(text)
                        length = len(input_ids)
                        padding = [self.pad_token_id] * (max_len - length)
                        input_ids = padding + input_ids
                        attention_masks = [0] * (max_len - length) + [1] * length
                        result["input_ids"].append(input_ids)
                        result["attention_masks"].append(attention_masks)
                    return result

    def batch_decode(self, indices:torch.Tensor) -> List[str]:
        result = []
        for i in range(indices.shape[0]):
            result.append(self.decode(indices[i]))
        return result


def main():
    vocab_file = "vocab.json"

    tokenizer = CharTokenizer(vocab_file=vocab_file, 
                              bos_token="<BOS>",
                              eos_token="<EOS>",
                              sep_token="<SEP>",
                              unk_token="<UNK>",
                              pad_token="<PAD>"
                            )

    print(f"vocab_size: {tokenizer.vocab_size}")

    texts = ["L4 N3 S1 <SEP> P a s s 1 2 3 $"]

    for text in texts: 
        indices = tokenizer.encode(text, return_is_tensor=True)
        reconstructed_text = tokenizer.decode(indices)
        
        print('inputs:{}'.format(text))
        print('encoded:{}'.format(indices))
        print('decoded:{}'.format(reconstructed_text))


if __name__ == "__main__":
    main()

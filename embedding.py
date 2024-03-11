from chromadb import Documents, EmbeddingFunction, Embeddings
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from torch import cuda

class E5LargeEmbeddingFunction(EmbeddingFunction):
    def __init__(self) -> None:
        super().__init__()

        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

        self.mode = 'passage' #passage для документов, query для запросов пользователя

    def change_mode(self, new_mode='query'):
        assert new_mode == 'query' or new_mode == 'passage', 'invalid model mode'

        self.mode = new_mode

    def average_pool(self, last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_embeddings(self, input_text):
        #add mode marker
        input_text = [f'{self.mode}: ' + x for x in input_text]

        # Tokenize the input texts
        batch_dict = self.tokenizer(input_text, max_length=512, padding=True, truncation=True, return_tensors='pt')

        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def __call__(self, input: Documents) -> Embeddings:
        result = self.get_embeddings(input).cpu().detach().tolist()
        return result

#for test

# emb = E5LargeEmbeddingFunction()
# input_texts = ['how much protein should a female eat',
#                'summit define',
#                "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
#                "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."]

# print(len(emb(input_texts)))
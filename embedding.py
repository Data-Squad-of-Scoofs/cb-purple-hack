from chromadb import Documents, EmbeddingFunction, Embeddings
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from torch import cuda

class E5LargeEmbeddingFunction(EmbeddingFunction):
    def __init__(self, device='') -> None:
        super().__init__()

        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large').to(self.device)

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
        input_text = f'{self.mode}: ' + input_text

        # Tokenize the input texts
        batch_dict = self.tokenizer(input_text, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.device)

        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def __call__(self, input: Documents) -> Embeddings:
        result = self.get_embeddings(input).cpu().detach().tolist()
        return result
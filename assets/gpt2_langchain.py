import torch
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import transformers

if torch.cuda.is_available():
    print("Using gpu")
    device = 'cuda'
else:
    print("Using cpu")
    device = 'cpu'

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# GPT2 = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
# GPT2.to(device)


class GPT2LLM(LLM):
    max_chars: int
    model: transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel
    tokenizer: transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer
    trim_prompt: bool
    verbose: bool
    
    @property
    def _llm_type(self) -> str:
        return "Huggingface GPT2 implementation"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"max_chars":self.max_chars}
    
    def _call(self, prompt: str, stop: Optional[List[str]]=None) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
        beam_outputs = self.model.generate(
            input_ids, 
            max_length = self.max_chars, 
            # no_repeat_ngram_size = 2, 
            num_return_sequences = 1, 
            early_stopping = True,
            num_beams=4,
        )
        _string = self.tokenizer.decode(beam_outputs[0], skip_special_tokens=True)
        if self.verbose:print(_string)
        
        ## Here we clean up the LLM output to remove the prompt and then isolate the python 
        ##     code block that the llm_math chain is looking for.
        if self.trim_prompt: 
            _string = _string[len(prompt):]
            _string = "```"+_string.split("```")[1]+"```"

        if self.verbose:print(_string)
        return _string
    
from langchain.prompts.prompt import PromptTemplate
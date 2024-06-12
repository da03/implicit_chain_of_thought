from transformers import AutoModelForCausalLM, AutoTokenizer


model = AutoModelForCausalLM.from_pretrained('ahxt/LiteLlama-460M-1T')
tokenizer = AutoTokenizer.from_pretrained('ahxt/LiteLlama-460M-1T')
model.push_to_hub('yuntian-deng/LiteLlama-460M-1T')
tokenizer.push_to_hub('yuntian-deng/LiteLlama-460M-1T')

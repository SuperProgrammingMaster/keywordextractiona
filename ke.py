import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

class KeywordExtractor(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super(KeywordExtractor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x):
        embedded = self.embedding(x) 
        embedded = embedded.permute(1, 0, 2)
        encoded = self.transformer_encoder(embedded)
        logits = self.fc(encoded)
        return logits
def get_target(input_tensor, keywords):
    tokend_keywords = []
    for keyword in keywords:
        tokend_keywords.append(
            tokenizer(keyword, return_tensors="pt")['input_ids'].squeeze(0)
        )

    target_tensor = torch.zeros(
        input_tensor.size(0), input_tensor.size(1), 2, device=input_tensor.device
    )

    for keyword_tokens in tokend_keywords:
        for batch_idx in range(input_tensor.size(0)):
            for i in range(input_tensor.size(1) - len(keyword_tokens) + 1):
                if torch.equal(input_tensor[batch_idx, i:i+len(keyword_tokens)], keyword_tokens):
                    target_tensor[batch_idx, i:i+len(keyword_tokens), 1] = 1  
                    target_tensor[batch_idx, i:i+len(keyword_tokens), 0] = 0

    return target_tensor


sentence = "영상 편집을 하는데 필요한 신나는 노래를 추천해줘."

tokens = tokenizer(sentence, return_tensors="pt")['input_ids']

vocab_size = 50257
embed_dim = 32
num_heads = 8
num_layers = 4
model = KeywordExtractor(vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()

    input_tensor = tokens
    target = get_target(input_tensor,keywords=["영상","편집","신나는","노래","추천"])

    optimizer.zero_grad()
    output = model(input_tensor)

    
    output = output.permute(0, 2, 1)
    target = target.permute(1,2,0)

    print(output)
    print(target)
    exit()
    loss = criterion(output, target)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

def remove_tags(text):
    tags = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[S]'] 
    words = text.split() 
    cleaned_words = []
    
    for word in words:
        if not any(word.startswith(tag) or word.endswith(tag) for tag in tags):
            cleaned_words.append(word)
    
    return ' '.join(cleaned_words)

model.eval()
with torch.no_grad():
    output = model(input_tensor) 
    probs = torch.softmax(output, dim=-1)
    high_prob_indices = (probs >= 0.9).nonzero(as_tuple=True)[0]
    important_tokens = tokens[0, high_prob_indices]
    important_words = tokenizer.decode(important_tokens)
    print("KeyWord:", (important_words))


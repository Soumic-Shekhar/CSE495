import torch
import torch.nn as nn
import torch.optim as optim

#sentence = ['hi', 'my', 'name', 'is', 'x']
sentence = [0,1,2,3,4,5]


'''
tokenize = {}

for word in range(len(sentence)):
    tokenize[sentence[word]] = word

print(tokenize)
tensor = []
for token in tokenize:
    print(tokenize[token])
    tensor = (torch.tensor[tokenize), dtype=torch.long)
print(tensor)

'''


#tensor = (torch.tensor(sentence, dtype = torch.int))

#print(tensor)

input_size = 1
dim = 6

'''
embed = nn.Embedding(input_size, dim)
print((embed(tensor)).size())
print(embed(tensor))
_input = (embed(tensor)).unsqueeze(0)
print(_input.size())

'''

class autoEncoder(nn.Module):

    def __init__(self, input_size, dim):
        super(autoEncoder, self).__init__()

        self.input_size = input_size
        self.dim = dim 
        self.embeddings = nn.Embedding(self.input_size, self.dim)
        self.encoder = nn.LSTM(self.input_size, self.dim)
        self.decoder = nn.LSTM(self.dim, self.input_size)
        self.softMax = nn.functional.log_softmax(1)

    def forward(self, _input):
        embeds = self.embeddings(_input).view(-1,6,1)
        #print(embeds)
        #print(embeds.shape)
        _, (last_hidden, _) = self.encoder(embeds)
        #print(last_hidden)
        decoded, _ =  self.decoder(last_hidden)
        #print(decoded)
        return self.softMax(decoded)


losses = []
#loss_function = nn.functional.nll_loss()
#loss_fn = torch.nn.MSELoss(reduction='sum')
#loss_fn = nn.CrossEntropyLoss()
model = autoEncoder(input_size, dim).cpu()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    print("training: " , epoch)
    total_loss = 0
    for src in sentence:
        inp = torch.tensor(sentence[src], dtype=torch.long)
        model.zero_grad()
        out = model(inp)
        test = out.argmax(1)
        print(test)
        trg = torch.tensor(sentence[src], dtype = torch.long)
        trg = trg.view(1,1)
        loss = nn.functional.nll_loss(test, trg)
        print (loss)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    losses.append(total_loss)
print(losses)

#print(model(tensor))


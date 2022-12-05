import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)                   # shape [10, 256]
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size  = embed_size
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size
        self.num_layers  = num_layers
        
        self.embedding= nn.Embedding(vocab_size, embed_size)
        self.lstm     = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc       = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        batch_size = features.size(0)                                          # assign batch_size based on features
        captions = captions[:,:-1]                                             # remove the <stop>
        embedded_captions = self.embedding(captions)                           # run the captions through the embedding layer
        lstm_in = torch.cat((features.unsqueeze(1), embedded_captions), dim=1) # concat features with captions
                                                                               #    need to expand features to match 
                                                                               #    to match embedded_captions
        lstm_out, _      = self.lstm(lstm_in)                                  # no need to define a hidden layer
        fc_out           = self.fc(lstm_out)                                        #    as it is done by default
                                                                
                                                                               # since we are using CrossEntropyLoss() we do
                                                                               #   not need to use log_softmax() during training
        
        #print(captions.shape) w/  <STOP>   [10, 12]
        #print(captions.shape) w/o <STOP>   [10, 11]
        #print(features.shape)              [10, 256]       
        #print(embedded_captions.shape)     [10, 11, 256]  when we embed captions they become a 3 dimensional tensor
        #print(features.unsqueeze(1).shape) [10,  1, 256]  need to expand features so we can concatenate them w/ captions     
        #print(lstm_in.shape)               [10, 12, 256]  

        return fc_out
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        # input shape is torch.Size([1, 1, 512])
        pred = []
        stop_idx = 1
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)         # lstm_out shape [1, 1, 512]
            fc_out           = self.fc(lstm_out)                 # fc_out shape   [1, 1, 9955]
            pred_word_tensor = torch.argmax(fc_out, dim=2)       # find the index corresponding to the max value (tensor)
            inputs           = self.embedding(pred_word_tensor)  # update the inputs
            pred_word_idx    = pred_word_tensor.item()           # extract the value of the index from the tensor
            pred.append(pred_word_idx)                           # append to the predicted sentence
            
        #data_loader.dataset.vocab.idx2word
        
        return pred
import torch
import torch.nn as nn
import torch.nn.functional as F

class TargetEmbeddingNet(nn.Module):
    def __init__(self, input_size, output_dim):
        super(TargetEmbeddingNet, self).__init__()

        self.fc_all = nn.Sequential(
                                nn.Linear(input_size, input_size),
                                nn.PReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(input_size, output_dim)
                    )


    def forward(self, x):
        # output = self.convnet(x)
        # output = output.view(output.size()[0], -1)
        # output = self.fc(output)
        output = self.fc_all(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class TargetCNNEmbeddingNet(nn.Module):
    def __init__(self, emb_size, num_filters, output_dim, window_sizes=(3, 4, 5)):
        super(TargetCNNEmbeddingNet, self).__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, [window_size, emb_size], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])

        self.fc = nn.Linear(num_filters * len(window_sizes), output_dim)

    def forward(self, x):

        # Apply a convolution + max pool layer for each window size
        x = torch.unsqueeze(x, 1)       # [B, C, T, E] Add a channel dim.
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))        # [B, F, T, 1]
            x2 = F.dropout(x2, 0.2)
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)            # [B, F, window]

        # FC
        x = x.view(x.size(0), -1)       # [B, F * window]
        x = F.dropout(x, 0.2)
        logits = self.fc(x)             # [B, class]

        return logits

    def get_embedding(self, x):
        return self.forward(x)

class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


    def get_dist(self, input1, input2):
        from scipy.spatial import distance

        #Reshape the inputs 
        # input1  = input1.unsqueeze(2)
        # input2  = input2.unsqueeze(2)

        output1 = self.get_embedding(input1)
        output2 = self.get_embedding(input2)
        
        #dist = distance.cosine(output1.detach().numpy(), output2.detach().numpy())

        diff = output2 - output1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        dist = dist.item()

        return dist



class TripletNet(nn.Module):
    def __init__(self, avg_embedding_net=None, target_embedding_net=None):
        super(TripletNet, self).__init__()
        #self.embedding_net = embedding_net
        
        self.avg_embedding_net = avg_embedding_net
        self.target_embedding_net = target_embedding_net

    def forward(self, x1, x2, x3):
        #output1 = self.embedding_net(x1)
        #output2 = self.embedding_net(x2)
        #output3 = self.embedding_net(x3)
        if self.target_embedding_net is not None:
            x1 = self.target_embedding_net(x1) #embed the anchor which is the conclusion target
            x3 = self.target_embedding_net(x3) # embed negative target which is a random target

        if self.avg_embedding_net:
            x2 = self.avg_embedding_net(x2) # embed the postivie instance which is the average vector
        

        return x1, x2, x3

    def get_target_embedding(self, x):
        if self.target_embedding_net is not None:
            return self.target_embedding_net(x)
        else:
            return x

    def get_avg_embedding(self, x):
        if self.avg_embedding_net is not None:
            return self.avg_embedding_net(x)
        else:
            return x

    def get_dist(self, input1, input2):
        from scipy.spatial import distance

        output1 = self.get_avg_embedding(input1)
        output2 = self.get_target_embedding(input2)
        
        dist = distance.cosine(output1.detach().numpy(), output2.detach().numpy())

        # diff = output2 - output1
        # dist_sq = torch.sum(torch.pow(diff, 2), 1)
        # dist = torch.sqrt(dist_sq)
        # dist = dist.item()

        return dist


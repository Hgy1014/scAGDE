from sklearn.cluster import KMeans, MiniBatchKMeans
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from tqdm import tqdm
import os
from layer import build_gcn, MLPGaussianSample, GCNGaussianSample, build_mlp, ClusterAssignment
from loss import binary_cross_entropy, kl_divergence, target_distribution
from mclust import *

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init.kaiming_normal_(m.weight.data)
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()



def BCE(out, tar, weight):
    eps = 1e-12  # The case without eps could lead to the `nan' situation
    l_n = weight * (tar * (torch.log(out + eps)) + (1 - tar) * (torch.log(1 - out + eps)))
    l = -torch.sum(l_n) / torch.numel(out)
    return l


def warimup(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    lr = 0.0003 + epoch * ((0.001 - 0.0003) / 1500)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Encoder(nn.Module):
    def __init__(self, dims, bn=False, dropout=0):
        super(Encoder, self).__init__()
        [x_dim, h_dim, z_dim] = dims
        self.hidden = build_mlp([x_dim] + h_dim, bn=bn, dropout=dropout)
        self.sample = MLPGaussianSample(([x_dim] + h_dim)[-1], z_dim)

    def forward(self, x):
        x = self.hidden(x)
        return self.sample(x)

    def encode(self, x):
        x = self.hidden(x)
        return self.sample.encode(x)



class Decoder(nn.Module):
    def __init__(self, dims, bn=False, dropout=0, output_activation=nn.Sigmoid()):
        super(Decoder, self).__init__()
        [z_dim, h_dim, x_dim] = dims
        self.hidden = build_mlp([z_dim, *h_dim], bn=bn, dropout=dropout)
        self.reconstruction = nn.Linear([z_dim, *h_dim][-1], x_dim)
        self.output_activation = output_activation

    def forward(self, x):
        x = self.hidden(x)
        if self.output_activation is not None:
            return self.output_activation(self.reconstruction(x))
        else:
            return self.reconstruction(x)



class ChromatinAccessibilityAutoEncoder(BaseModel):
    def __init__(self, dims, bn=False, dropout=0, binary=True, device=torch.device("cuda")):
        super(ChromatinAccessibilityAutoEncoder, self).__init__()
        [x_dim, z_dim, encode_dim, decode_dim] = dims
        self.z_dim = z_dim
        self.device = device
        if binary:
            decode_activation = nn.Sigmoid()
        else:
            decode_activation = None
        self.encoder = Encoder([x_dim, encode_dim, z_dim], bn=bn, dropout=dropout)
        self.decoder = Decoder([z_dim, decode_dim, x_dim], bn=bn, dropout=dropout, output_activation=decode_activation)
        self.reset_parameters()

    def loss_function(self, x):
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        loss_Likelihood = F.binary_cross_entropy(recon_x, x, reduction='sum') / len(x)
        # loss_Likelihood = torch.sum(binary_cross_entropy(recon_x, x)) / len(x)
        loss_KLDivergence = 0.5 * torch.sum(torch.exp(logvar) + torch.pow(mu, 2) - 1. - logvar) / len(x)
        return loss_Likelihood + loss_KLDivergence

    def encodeBatch(self, X):
        self.eval()
        x = torch.from_numpy(X).float().to(self.device)
        output, _, _ = self.encoder.encode(x)
        output = output.detach().cpu().numpy()
        return output


    def fit(self, X, lr=0.0002, weight_decay=5e-4, max_iter=5000, verbose=True, outdir="./",save_epoch=None,clip=0):
        progress = None
        with tqdm(range(max_iter), total=max_iter, desc='CountModel') as tq:
            x = torch.from_numpy(X).float().to(self.device)
            iters = 1500
            if save_epoch is not None: iters = 0
            if max_iter>iters:
                optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
                for epoch in range(iters):
                    self.train()
                    optimizer.zero_grad()
                    loss = self.loss_function(x)
                    loss.backward()
                    optimizer.step()
                    if verbose:
                        outstr = "loss={:.4f}".format(loss.item())
                        tq.set_postfix_str(outstr)
                        tq.update()
                    if save_epoch is not None:
                        if ((epoch+1) % save_epoch == 0) or (epoch == 0):
                            progress = int(100*(epoch/max_iter))                 
                            if not os.path.exists(os.path.join(outdir,"weights")): os.mkdir(os.path.join(outdir,"weights"))
                            weight_tensor = self.encoder.hidden[0].weight.detach().cpu().data
                            peakImportance = torch.std(weight_tensor, 0).numpy()
                            # peakImportance = (var_tensor - min(var_tensor)) / (max(var_tensor) - min(var_tensor))
                            peakImportance.tofile(os.path.join(outdir,"weights", f'{epoch+1}.csv'),sep=",")
            else:
                iters = 0
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
            for epoch in range(max_iter-iters):
                self.train()
                optimizer.zero_grad()
                loss = self.loss_function(x)
                loss.backward()
                if clip: torch.nn.utils.clip_grad_norm(self.parameters(), 10)  # clip
                optimizer.step()
                if verbose:
                    outstr = "loss={:.4f}".format(loss.item())
                    tq.set_postfix_str(outstr)
                    tq.update()
                if save_epoch is not None:
                    epoch += iters
                    if (epoch+1) % save_epoch == 0:
                        progress = int(100*(epoch/max_iter))                     
                        if not os.path.exists(os.path.join(outdir,"weights")): os.mkdir(os.path.join(outdir,"weights"))
                        # weight_tensor = self.encoder.hidden[0].weight.detach().cpu().data
                        weight_tensor = self.encoder.hidden[0].weight.clone().cpu().data
                        peakImportance = torch.std(weight_tensor, 0).numpy()
                        # peakImportance = (var_tensor - min(var_tensor)) / (max(var_tensor) - min(var_tensor))
                        peakImportance.tofile(os.path.join(outdir,"weights", f'{epoch+1}.csv'),sep=",")
        torch.save(self.state_dict(), os.path.join(outdir, 'CountModel.pt'))  # save model
        
class ChromatinAccessibilityAutoEncoder_scale(ChromatinAccessibilityAutoEncoder):
    def __init__(self, dims, bn=False, dropout=0, binary=True, device=torch.device("cuda")):
        super().__init__(dims, bn, dropout, binary, device)
    def fit(self,X,
        lr=0.002,
        weight_decay=5e-4,
        max_iter=200,
        verbose=True,
        outdir="./",
        save_epoch=None,
        clip=0,
        ):
        # todo 定义优化器
        self.train()
        progress = None
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        iters = 50
        with tqdm(range(max_iter), total=max_iter, desc='CountModel',leave=None) as tq:
            if max_iter>iters:
                for epoch in range(iters):
                    tk0 = tqdm(enumerate(X), total=len(X), leave=False, desc='Iterations')
                    for i, x in tk0:
                        x = torch.from_numpy(x).float().to(self.device)
                        optimizer.zero_grad()
                        loss = self.loss_function(x)
                        loss.backward()
                        if clip: torch.nn.utils.clip_grad_norm(self.parameters(), 10)  # clip
                        optimizer.step()
                        if verbose:
                            outstr = "loss={:.4f}".format(loss.item())
                            tq.set_postfix_str(outstr)
                            tq.update(0)
                    if save_epoch is not None:
                        if ((epoch+1) % save_epoch == 0) or (epoch == 0):
                            progress = int(100*(epoch/max_iter))                   
                            if not os.path.exists(os.path.join(outdir,"weights")): os.mkdir(os.path.join(outdir,"weights"))
                            weight_tensor = self.encoder.hidden[0].weight.detach().cpu().data
                            peakImportance = torch.std(weight_tensor, 0).numpy()
                            # peakImportance = (var_tensor - min(var_tensor)) / (max(var_tensor) - min(var_tensor))
                            peakImportance.tofile(os.path.join(outdir,"weights", f'{epoch}.csv'),sep=",") 
            else:
                iters = 0
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)            
            for epoch in range(max_iter-iters):
                tk0 = tqdm(enumerate(X), total=len(X), leave=False, desc='Iterations')
                for i, x in tk0:
                    x = torch.from_numpy(x).float().to(self.device)
                    optimizer.zero_grad()
                    loss = self.loss_function(x)
                    loss.backward()
                    if clip: torch.nn.utils.clip_grad_norm(self.parameters(), 10)  # clip
                    optimizer.step()
                    if verbose:
                        outstr = "loss={:.4f}".format(loss.item())
                        tq.set_postfix_str(outstr)
                        tq.update(0)
                if save_epoch is not None:
                    if ((epoch+1) % save_epoch == 0) or (epoch == 0):
                        progress = int(100*(epoch/max_iter))                   
                        if not os.path.exists(os.path.join(outdir,"weights")): os.mkdir(os.path.join(outdir,"weights"))
                        weight_tensor = self.encoder.hidden[0].weight.detach().cpu().data
                        peakImportance = torch.std(weight_tensor, 0).numpy()
                        # peakImportance = (var_tensor - min(var_tensor)) / (max(var_tensor) - min(var_tensor))
                        peakImportance.tofile(os.path.join(outdir,"weights", f'{epoch}.csv'),sep=",") 
        torch.save(self.state_dict(), os.path.join(outdir, 'CountModel.pt'))  # save model


    def encodeBatch(self, input):
        output = []
        for x in input:
            x = torch.from_numpy(x).float().to(self.device)
            z, _, _ = self.encoder.encode(x)
            output.append(z.detach().cpu())
        output = torch.cat(output).numpy()
        return output

    
class GCNEncoder(nn.Module):
    def __init__(self, dims, bn=False, dropout=0, device=torch.device('cuda'),basic="GCN"):
        super(GCNEncoder, self).__init__()
        [x_dim, h_dim, z_dim] = dims
        self.hidden = build_gcn([x_dim] + h_dim, bn=bn, dropout=dropout, device=device,basic=basic)
        self.sample = GCNGaussianSample([x_dim, *h_dim][-1], z_dim, device=device)

    def forward(self, adj, x):
        input = (adj, x)
        _, x = self.hidden(input)
        return self.sample(adj, x)

    def encode(self, adj, x):
        input = (adj, x)
        _, x = self.hidden(input)
        return self.sample.encode(adj, x)


class GCNEncoder_scale(nn.Module):
    def __init__(self, dims, bn=False, dropout=0, device=torch.device('cuda')):
        super(GCNEncoder, self).__init__()
        [x_dim, h_dim, z_dim] = dims
        self.hidden = build_gcn([x_dim] + h_dim, bn=bn, dropout=dropout, device=device)
        self.sample = GCNGaussianSample([x_dim, *h_dim][-1], z_dim, device=device)

    def forward(self, adj, x):
        input = (adj, x)
        _, x = self.hidden(input)
        return self.sample(adj, x)

    def encode(self, adj, x):
        input = (adj, x)
        _, x = self.hidden(input)
        return self.sample.encode(adj, x)



class GraphEmbeddingModel(BaseModel):
    def __init__(self,  dims,  n_centroids, bn=False, dropout=0, binary=True, device=torch.device("cuda"),wADJ=10,wX=5,wKL=1,wDEC=1,sigma=0,basic="GCN",cluster_opt=True):
        super(GraphEmbeddingModel, self).__init__()
        [x_dim, z_dim, encode_dim, decode_dim] = dims
        self.n_centroids = n_centroids
        self.z_dim = z_dim
        self.device = device
        self.cluster_opt = cluster_opt
        if binary:
            decode_activation = nn.Sigmoid()
            # decode_activation = nn.Softmax()
        else:
            decode_activation = None
        self.encoder = GCNEncoder([x_dim, encode_dim, z_dim], bn=bn, dropout=dropout,basic=basic,device=device)
        self.decoder = Decoder([z_dim, decode_dim, x_dim], bn=bn, dropout=dropout,
                                  output_activation=decode_activation)
        self.cluster_assignment = ClusterAssignment(cluster_number=n_centroids,
                                                    embedding_dimension=dims[1],
                                                    alpha=1.0,
                                                    device=device)

        self.reset_parameters()
        self.DEC_loss_function = nn.KLDivLoss(reduction="sum")
        self.wADJ = wADJ
        self.wX = wX
        self.wKL = wKL
        self.wDEC = wDEC

    def loss_function(self, adj, x, epoch=-1, max_iter=4000, update_interval=100, thres0=0.55, thres1=0.8, w1=0.3,w2=1):
        z, mu, logvar = self.encoder(adj, x)
        recon_x = self.decoder(z)
        recon_adj = torch.sigmoid(torch.matmul(z, z.t()))
        loss_ReconAdj = torch.sum(F.binary_cross_entropy(recon_adj, adj))
        loss_Likelihood = torch.sum(binary_cross_entropy(recon_x, x)) / len(x) * 0.5
        w = w1 + epoch * (w2 - w1) / max_iter if epoch != -1 else 1
        loss_KLDivergence = torch.sum(kl_divergence(mu, logvar)) / len(x) * w
        
        if self.cluster_opt:
            # dec
            q = self.cluster_assignment(z)
            if epoch % update_interval == 0:
                q = q.data
                self.p = target_distribution(q)
            loss_Dec = self.DEC_loss_function(q.log(), self.p) / len(x)
            # pseudo
            clu_assignment = torch.argmax(self.p, -1)
            clu_assignment_onehot = F.one_hot(clu_assignment, self.n_centroids)
            # 阈值动态变化，一开始低一些，后面高一些
            thres = thres0 + (thres1 - thres0) / max_iter * epoch
            thres_matrix = torch.zeros_like(self.p) + thres
            weight_label = torch.ge(F.normalize(self.p, p=2), thres_matrix).type(torch.FloatTensor).to(self.device)
            loss_Pseudo = BCE(self.p, clu_assignment_onehot, weight_label)


            return self.wADJ*loss_ReconAdj + self.wX *loss_Likelihood + self.wKL*loss_KLDivergence + self.wDEC*(loss_Dec + loss_Pseudo), loss_Likelihood
        else:
            return self.wADJ*loss_ReconAdj + self.wX *loss_Likelihood + self.wKL*loss_KLDivergence, loss_Likelihood
        
        
    def loss_function_pretrain(self, adj, x, epoch=-1, max_iter=4000, w1=0.3,w2=1):
        z, mu, logvar = self.encoder(adj, x)
        recon_x = self.decoder(z)
        recon_adj = torch.sigmoid(torch.matmul(z, z.t()))
        loss_ReconAdj = torch.sum(F.binary_cross_entropy(recon_adj, adj))
        loss_Likelihood = torch.sum(binary_cross_entropy(recon_x, x)) / len(x)
        loss_KLDivergence = torch.sum(kl_divergence(mu, logvar)) / len(x) * 1

        return loss_ReconAdj + loss_Likelihood + loss_KLDivergence, loss_Likelihood

    def fit(self, adj_n, X,
            lr=0.002,
            weight_decay=5e-4,
            max_iter=30000,
            pre_iter=2000,
            outdir='./',
            verbose=True,
            update_interval=100,
            patience=50,
            clip=0,
            early_stopping=False
            ):
        self.early_stopping = EarlyStopping(save_path="",patience=patience)
        adj_n = torch.from_numpy(adj_n).float().to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        # TODO 预训练
        pre_optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)  # 预训练
        reconLoss = 0
        if pre_iter > 0:
            self.train()
            with tqdm(range(pre_iter), total=pre_iter, desc='GraphModel-preTrain') as tq:
                x = torch.from_numpy(X).float().to(self.device)
                for epoch in tq:
                    pre_optimizer.zero_grad()
                    """calculate_loss"""
                    loss, loss_Likelihood = self.loss_function_pretrain(adj_n, x, epoch, pre_iter)
                    loss.backward()
                    if clip:
                        torch.nn.utils.clip_grad_norm(self.parameters(), 10)  # clip
                    pre_optimizer.step()
                    reconLoss += loss_Likelihood.item()   
                    if verbose:
                        outstr = "loss={:.4f}".format(loss.item())
                        tq.set_postfix_str(outstr)
                        tq.update(0)
                    if early_stopping:
                        self.early_stopping(reconLoss,self)
                        if self.early_stopping.early_stop:
                            print("early stopping")
                            break     
        torch.save(self.state_dict(), os.path.join(outdir, 'pretrain.pt'))  # save model
        if self.cluster_opt:
            with torch.no_grad():
                z = self.encodeBatch(adj_n, X,impute=False)
            self.init_center(z)
        self.early_stopping = EarlyStopping(save_path="",patience=patience)
        # TODO 正式训练
        self.train()
        reconLoss = 0
        with tqdm(range(max_iter), total=max_iter, desc='GraphModel') as tq:
            x = torch.from_numpy(X).float().to(self.device)
            for epoch in tq:
                optimizer.zero_grad()
                loss, loss_Likelihood = self.loss_function(adj_n, x, epoch, max_iter, update_interval)
                loss.backward()
                if clip:
                    torch.nn.utils.clip_grad_norm(self.parameters(), 10)  # clip
                optimizer.step()
                reconLoss += loss_Likelihood.item()
                if verbose:
                    outstr = "loss={:.4f}".format(loss.item())
                    tq.set_postfix_str(outstr)
                    tq.update(0)
                if early_stopping:
                    self.early_stopping(reconLoss,self)
                    if self.early_stopping.early_stop:
                        print("early stopping")
                        break   
        torch.save(self.state_dict(), os.path.join(outdir, 'GraphModel.pt'))  # save model
        self.p = None


    def init_center(self, feature):
        kmeans = KMeans(n_clusters=self.n_centroids, n_init=1, random_state=0)
        kmeans.fit_predict(np.float64(feature))
        self.cluster_assignment.cluster_centers.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))

    def encodeBatch(self, adj, x, impute=True):
        if not isinstance(adj,torch.Tensor):
            adj = torch.from_numpy(adj).float().to(self.device)
        if not isinstance(x,torch.Tensor):
            x = torch.from_numpy(x).float().to(self.device)
        z, _, _ = self.encoder.encode(adj, x)
        if impute:
            x = self.decoder(z)
            return z.detach().cpu().numpy(), x.detach().cpu().numpy()
        else:
            return z.detach().cpu().numpy()
        
                
class GraphEmbeddingModel_scale(GraphEmbeddingModel):
    def __init__(self, dims, n_centroids, bn=False, dropout=0, binary=True, device=torch.device("cuda"), wADJ=.1, wX=5, wKL=1, wDEC=1, sigma=0, cluster_opt=True):
        super().__init__(dims, n_centroids, bn, dropout, binary, device, wADJ, wX, wKL, wDEC, sigma, basic="GCNskip",cluster_opt=cluster_opt)
    

    def fit(self, adj_n, X, 
            lr=0.002,
            weight_decay=5e-4,
            max_iter=300,
            pre_iter=200,
            outdir='./',
            verbose=True,
            update_interval=1,
            patience=50,
            clip=0,
            early_stopping=False,
            centroids_init="kmeans",
            ):
        self.early_stopping = EarlyStopping(save_path="",patience=patience)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        # TODO 预训练
        pre_optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)  # 预训练
        if pre_iter > 0:
            self.train()
            with tqdm(range(pre_iter), total=pre_iter, desc='Pretrain Epochs') as tq:
                for epoch in tq:
                    reconLoss = 0
                    tk0 = tqdm(enumerate(X), total=len(X), leave=False, desc='Iterations')
                    for i,x in tk0:
                        x = torch.from_numpy(x).float().to(self.device)
                        adj_n_batch = torch.FloatTensor(adj_n[i]).to(self.device)
                        pre_optimizer.zero_grad()
                        """calculate_loss"""
                        loss, loss_Likelihood = self.loss_function_pretrain(adj_n_batch, x)
                        loss.backward()
                        if clip:
                            torch.nn.utils.clip_grad_norm(self.parameters(), 10)  # clip
                        pre_optimizer.step()
                        reconLoss += loss_Likelihood.item()                        
                        if verbose:
                            outstr = "loss={:.4f}".format(loss.item())
                            tk0.set_postfix_str(outstr)
                            tk0.update(0)
                    if early_stopping:
                        self.early_stopping(reconLoss,self)
                        if self.early_stopping.early_stop:
                            print("early stopping")
                            break     
        torch.save(self.state_dict(), os.path.join(outdir, 'pretrain.pt'))  # save model
        if self.cluster_opt:
            with torch.no_grad():
                z = self.encodeBatch(adj_n, X, impute=False)
            self.init_center(z,init=centroids_init)        
        self.early_stopping = EarlyStopping(save_path="",patience=patience)
        # TODO 正式训练
        self.train()
        with tqdm(range(max_iter), total=max_iter, desc='Epochs') as tq:
            for epoch in tq:
                reconLoss = 0
                tk0 = tqdm(enumerate(X), total=len(X), leave=False, desc='Iterations')
                for i, x, in tk0:
                    x = torch.from_numpy(x).float().to(self.device)
                    adj_n_batch = torch.FloatTensor(adj_n[i]).to(self.device)
                    optimizer.zero_grad()
                    loss, loss_Likelihood = self.loss_function( adj_n_batch, x, epoch, max_iter, update_interval)
                    loss.backward()
                    if clip:
                        torch.nn.utils.clip_grad_norm(self.parameters(), 10)  # clip
                    optimizer.step()
                    reconLoss += loss_Likelihood.item()
                    # 输出显示loss value
                    if verbose:
                        outstr = "loss={:.4f}".format(loss.item())
                        tk0.set_postfix_str(outstr)
                        tk0.update(0)
                if early_stopping:
                    self.early_stopping(reconLoss,self)
                    if self.early_stopping.early_stop:
                        print("early stopping")
                        break         
        torch.save(self.state_dict(), os.path.join(outdir, 'GraphModel.pt'))  # save model
        self.p = None

    def encodeBatch(self, adj, dataloader, impute=False):
        output = []
        if impute: output_reconx = []
        for i,x in enumerate(dataloader):
            x = torch.from_numpy(x).float().to(self.device)
            adj_n = torch.from_numpy(adj[i]).float().to(self.device)
            z, _, _ = self.encoder.encode(adj_n, x)
            output.append(z.detach().cpu())
            if impute:
                recon_x = self.decoder(z)
                output_reconx.append(recon_x.detach().cpu().data)
        output = torch.cat(output).numpy()
        if impute: 
            output_reconx = torch.cat(output_reconx).numpy()
            return output, output_reconx
        else:
            return output
        
    def init_center(self, feature, init="kmeans"):
        if init == "kmeans":
            kmeans = KMeans(n_clusters=self.n_centroids, n_init=1, random_state=0)
            kmeans.fit_predict(np.float64(feature))
            self.cluster_assignment.cluster_centers.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))
        elif init == "leiden":
            import anndata
            import scanpy as sc
            tmp = anndata.AnnData(feature)
            sc.pp.neighbors(tmp, n_neighbors = 15, use_rep="X")
            sc.tl.leiden(tmp, resolution = 0.2)
            labels = tmp.obs['leiden']
            n_clusters = len(np.unique(labels))
            centers = np.array([feature[labels == i].mean(0) for i in range(n_clusters)])
            self.cluster_assignment.cluster_centers.data.copy_(torch.tensor(centers, dtype=torch.float32))


if __name__ == '__main__':
    import torch
    a = GraphEmbeddingModel_scale([2088, 10, [128], []], n_centroids=8, device=torch.device("cuda"))
    print(a.encoder)
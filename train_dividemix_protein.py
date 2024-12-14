from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import gc
import argparse
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from torch.utils.data import Dataset, DataLoader
from accelerate.utils import set_seed
from transformers import EsmTokenizer, EsmModel
from src.models.pooling import Attention1dPoolingHead



class ProteinDataset(Dataset):
    def __init__(self, csv_file, mode='train', pred=[], prob=[]):
        self.data = pd.read_csv(csv_file)
        self.mode = mode
        self.pred = pred
        self.prob = prob
                
        # 氨基酸映射字典
        self.aa_to_idx = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
            'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
            'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
            'X': 20  # 未知氨基酸
        }
        self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D')
        
        if mode == 'train':
            if len(pred) > 0:
                self.data = self.data[pred]
                self.probability = prob[pred]
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        max_len = 1024
        sequence = self.data.iloc[idx]['aa_seq']
        label = self.data.iloc[idx]['label']
        
        # 使用tokenizer的padding功能
        inputs = self.tokenizer(
            sequence,
            padding='max_length',
            max_length=max_len,
            truncation=True,
            return_tensors='pt'
        )
        seq_idx = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
            
        if self.mode == 'train':
            # 数据增强:随机mask
            seq_idx2 = seq_idx.clone()
            # 只对非padding token进行mask
            attention_mask = inputs.attention_mask.squeeze(0)
            mask = (torch.rand(max_len) < 0.15) & (attention_mask == 1)
            seq_idx2[mask] = self.tokenizer.mask_token_id
            
            if len(self.pred) > 0:
                return seq_idx, seq_idx2, attention_mask, label, self.probability[idx]
            else:
                return seq_idx, seq_idx2, attention_mask, label
        else:
            return seq_idx, attention_mask, label

class ProteinDataLoader:
    def __init__(self, train_csv, val_csv, test_csv, batch_size, num_workers=4):
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers

    def run(self, mode, pred=[], prob=[]):
        if mode == 'warmup':
            warmup_dataset = ProteinDataset(self.train_csv, mode='warmup')
            warmup_loader = DataLoader(
                warmup_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )
            return warmup_loader
            
        elif mode == 'train':
            labeled_dataset = ProteinDataset(
                self.train_csv,
                mode='train',
                pred=pred,
                prob=prob
            )
            unlabeled_dataset = ProteinDataset(
                self.train_csv,
                mode='train',
                pred=~pred,
                prob=prob
            )
            
            labeled_loader = DataLoader(
                labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True
            )
            
            unlabeled_loader = DataLoader(
                unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True
            )
            return labeled_loader, unlabeled_loader
            
        elif mode == 'val':
            val_dataset = ProteinDataset(self.val_csv, mode='val')
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
            return val_loader
            
        elif mode == 'test':
            test_dataset = ProteinDataset(self.test_csv, mode='test')
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
            return test_loader
            
        elif mode == 'eval_train':
            eval_dataset = ProteinDataset(self.train_csv, mode='eval_train')
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
            return eval_loader

def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
    net.train()
    net2.eval()
    
    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//args.batch_size)+1
    optimizer.zero_grad()
    
    for batch_idx, (inputs_x, inputs_x2, attention_mask_x, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u_data = next(unlabeled_train_iter)
            inputs_u, inputs_u2, attention_mask_u = inputs_u_data[0], inputs_u_data[1], inputs_u_data[2]
            if inputs_u.size(0) != inputs_x.size(0):
                continue
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u_data = next(unlabeled_train_iter)
            inputs_u, inputs_u2, attention_mask_u = inputs_u_data[0], inputs_u_data[1], inputs_u_data[2]
            if inputs_u.size(0) != inputs_x.size(0):
                continue
        
        batch_size = inputs_x.size(0)
        
        labels_x = torch.zeros(batch_size, args.num_labels).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, attention_mask_x, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), attention_mask_x.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2, attention_mask_u = inputs_u.cuda(), inputs_u2.cuda(), attention_mask_u.cuda()

        with torch.no_grad():
            outputs_u11 = net(inputs_u, attention_mask_u, plm_model)
            outputs_u12 = net(inputs_u2, attention_mask_u, plm_model)
            outputs_u21 = net2(inputs_u, attention_mask_u, plm_model)
            outputs_u22 = net2(inputs_u2, attention_mask_u, plm_model)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T)
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()       
            
            outputs_x = net(inputs_x, attention_mask_x, plm_model)
            outputs_x2 = net(inputs_x2, attention_mask_x, plm_model)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/args.T)
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)           
            targets_x = targets_x.detach()
        
        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
        
        mixed_input = l * inputs_x + (1-l) * inputs_u
        mixed_input = mixed_input.long()
        mixed_attention_mask = l * attention_mask_x + (1-l) * attention_mask_u
        
        mixed_target = l * targets_x + (1-l) * targets_u
        
        logits = net(mixed_input, mixed_attention_mask, plm_model)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        
        # regularization
        prior = torch.ones(args.num_labels)/args.num_labels
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + penalty
        
        # 梯度累积
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        sys.stdout.write('\r')
        sys.stdout.write('Protein | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.4f'
                %(epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))
        sys.stdout.flush()

def warmup(net, optimizer, dataloader):
    net.train()
    optimizer.zero_grad()  # 移到循环外部
    
    for batch_idx, (inputs, attention_mask, labels) in enumerate(dataloader):      
        inputs, attention_mask, labels = inputs.cuda(), attention_mask.cuda(), labels.cuda() 
        outputs = net(inputs, attention_mask, plm_model)              
        loss = CEloss(outputs, labels)  
        
        penalty = conf_penalty(outputs)
        L = loss + penalty
        
        # 梯度累积
        L = L / args.gradient_accumulation_steps
        L.backward()
        
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        sys.stdout.write('\r')
        sys.stdout.write('| Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
                %(batch_idx+1, len(dataloader), loss.item(), penalty.item()))
        sys.stdout.flush()

def val(net, val_loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, attention_mask, targets) in enumerate(val_loader):
            inputs, attention_mask, targets = inputs.cuda(), attention_mask.cuda(), targets.cuda()
            outputs = net(inputs, attention_mask, plm_model)
            _, predicted = torch.max(outputs, 1)         
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()              
    acc = 100.*correct/total
    return acc

def test(net1, net2, test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, attention_mask, targets) in enumerate(test_loader):
            inputs, attention_mask, targets = inputs.cuda(), attention_mask.cuda(), targets.cuda()
            outputs1 = net1(inputs, attention_mask, plm_model)       
            outputs2 = net2(inputs, attention_mask, plm_model)           
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                    
    acc = 100.*correct/total
    print("\n| Test Acc: %.2f%%\n" %(acc))  
    return acc

def eval_train(net, eval_loader):
    net.eval()
    losses = torch.zeros(len(eval_loader.dataset))
    with torch.no_grad():
        for batch_idx, (inputs, attention_mask, targets) in enumerate(eval_loader):
            inputs, attention_mask, targets = inputs.cuda(), attention_mask.cuda(), targets.cuda() 
            outputs = net(inputs, attention_mask, plm_model) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[batch_idx*args.batch_size+b] = loss[b]
            
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter %3d\t' %(batch_idx)) 
            sys.stdout.flush()
            
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    losses = losses.reshape(-1,1)
    
    gmm = GaussianMixture(n_components=2,max_iter=10,reg_covar=5e-4,tol=1e-2)
    gmm.fit(losses)
    prob = gmm.predict_proba(losses) 
    prob = prob[:,gmm.means_.argmin()]       
    return prob

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

class PredictorPLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = Attention1dPoolingHead(1280, args.num_labels, 0.5)
    
    @torch.no_grad()
    def plm_embedding(self, plm_model, aa_seq, attention_mask):
        outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask)
        seq_embeds = outputs.last_hidden_state
        gc.collect()
        torch.cuda.empty_cache()
        return seq_embeds
    
    def forward(self, x, attention_mask, plm_model):
        x = self.plm_embedding(plm_model, x, attention_mask)
        x = self.classifier(x, attention_mask)
        return x

class PredictorConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(33, 32)
        self.conv1 = nn.Conv1d(32, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128*256, 256)  # 1024/4 = 256 after 2 pooling layers
        self.fc2 = nn.Linear(256, args.num_labels)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, attention_mask, plm_model):
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = x.transpose(1, 2)  # [batch, embed_dim, seq_len]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class PredictorPLMConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(640, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128*256, 256)
        self.fc2 = nn.Linear(256, args.num_labels)
        self.dropout = nn.Dropout(0.5)
    
    @torch.no_grad()
    def plm_embedding(self, plm_model, aa_seq, attention_mask):
        outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask)
        seq_embeds = outputs.last_hidden_state
        gc.collect()
        torch.cuda.empty_cache()
        return seq_embeds
    
    def forward(self, x, attention_mask, plm_model):
        x = self.plm_embedding(plm_model, x, attention_mask)
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Protein Solubility Training')
    parser.add_argument('--batch_size', default=4, type=int, help='train batchsize') 
    parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate')
    parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
    parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--num_epochs', default=100, type=int, help='total epochs')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='warmup epochs')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--num_labels', default=2, type=int, help='number of classes')
    parser.add_argument('--dataset_dir', default='data/PDBSol', type=str, help='path to dataset')
    parser.add_argument('--test_file', default='ExternalTest.csv', type=str, help='test file')
    parser.add_argument('--id', default='protein')
    parser.add_argument('--output_dir', default='ckpt')
    parser.add_argument('--gradient_accumulation_steps', default=32, type=int, help='gradient accumulation steps')
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    plm_model = EsmModel.from_pretrained('facebook/esm2_t33_650M_UR50D').cuda().eval()
    
    # build model
    print('| Building net')
    net1 = PredictorPLM().cuda()
    net2 = PredictorPLM().cuda()
    
    # Print model parameters
    total_params1 = sum(p.numel() for p in net1.parameters())
    total_params2 = sum(p.numel() for p in net2.parameters())
    print(f'Total parameters of net1: {total_params1/1e6:.2f}M')
    print(f'Total parameters of net2: {total_params2/1e6:.2f}M')
    cudnn.benchmark = True

    optimizer1 = optim.AdamW(net1.parameters(), lr=args.lr)
    optimizer2 = optim.AdamW(net2.parameters(), lr=args.lr)

    CE = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()
    conf_penalty = NegEntropy()
    
    best_acc = [0,0]
    
    # create data loader
    loader = ProteinDataLoader(
        train_csv=os.path.join(args.dataset_dir, 'train.csv'),
        val_csv=os.path.join(args.dataset_dir, 'valid.csv'),
        test_csv=os.path.join(args.dataset_dir, args.test_file),
        batch_size=args.batch_size
    )

    log = open(f'{args.output_dir}/{args.id}.txt', 'w')
    log.flush()

    for epoch in range(args.num_epochs+1):
        if epoch < args.warmup_epochs:     # warmup
            train_loader = loader.run('warmup')
            print('Warmup Net1')
            warmup(net1, optimizer1, train_loader)     
            train_loader = loader.run('warmup')
            print('\nWarmup Net2')
            warmup(net2, optimizer2, train_loader)                 
        else:       
            pred1 = (prob1 > args.p_threshold)  
            pred2 = (prob2 > args.p_threshold)      
            
            print('\nTrain Net1')
            labeled_loader1, unlabeled_loader1 = loader.run('train', pred2, prob2)
            train(epoch, net1, net2, optimizer1, labeled_loader1, unlabeled_loader1)
            
            print('\nTrain Net2')
            labeled_loader2, unlabeled_loader2 = loader.run('train', pred1, prob1)
            train(epoch, net2, net1, optimizer2, labeled_loader2, unlabeled_loader2)
        
        # Validation
        val_loader = loader.run('val')
        acc1 = val(net1, val_loader)
        print("\n| Validation\t Net1  Acc: %.2f%%" %(acc1))
        # save best model
        if acc1 > best_acc[0]:
            print('| >>> Saving Best Net1 ...')
            torch.save(net1.state_dict(), f'{args.output_dir}/{args.id}_net1.pth.tar')
            best_acc[0] = acc1
        
        
        acc2 = val(net2, val_loader)
        print("\n| Validation\t Net2  Acc: %.2f%%" %(acc2))
        # save best model
        if acc2 > best_acc[1]:
            print('| >>> Saving Best Net2 ...')
            torch.save(net2.state_dict(), f'{args.output_dir}/{args.id}_net2.pth.tar')
            best_acc[1] = acc2
        
        log.write('Validation Epoch:%d      Acc1:%.2f  Acc2:%.2f\n'%(epoch,acc1,acc2))
        log.flush()
        
        print('\n==== net 1 evaluate next epoch training data loss ====')
        eval_loader = loader.run('eval_train')
        prob1 = eval_train(net1, eval_loader)
        
        print('\n==== net 2 evaluate next epoch training data loss ====')
        eval_loader = loader.run('eval_train')
        prob2 = eval_train(net2, eval_loader)

        

    # Final test
    test_loader = loader.run('test')
    checkpoint1 = torch.load(f'{args.output_dir}/{args.id}_net1.pth.tar', weights_only=True)
    checkpoint2 = torch.load(f'{args.output_dir}/{args.id}_net2.pth.tar', weights_only=True)

    net1.load_state_dict(checkpoint1)
    net2.load_state_dict(checkpoint2)
    acc = test(net1, net2, test_loader)
    
    log.write('Test %s Acc:%.2f\n'%(args.test_file, acc))
    log.close()
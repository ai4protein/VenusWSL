from __future__ import print_function
import sys
import torch
import os
import gc
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from torch.utils.data import Dataset, DataLoader
from accelerate.utils import set_seed
from transformers import EsmTokenizer, EsmModel
from src.models.pooling import Attention1dPoolingHead
from datasets import load_dataset, Dataset, DatasetDict



class ProteinDataset(Dataset):
    def __init__(self, tokenizer, max_seq_len, dataset, mode='train', pred=[], prob=[]):
        self.dataset = dataset
        self.mode = mode
        self.pred = pred
        self.prob = prob
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        if mode == 'train' and len(pred) > 0:
            # Filter dataset based on pred
            indices = [i for i, p in enumerate(pred) if p]
            self.dataset = self.dataset[indices]
            self.probability = prob[pred]
            
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        sequence = self.dataset[idx]['aa_seq']
        label = self.dataset[idx]['label']
        
        inputs = self.tokenizer(
            sequence, padding='max_length', max_length=self.max_seq_len,
            truncation=True, return_tensors='pt'
        )
        seq_idx = inputs.input_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)
            
        if self.mode == 'train':
            seq_idx2 = seq_idx.clone()
            attention_mask = inputs.attention_mask.squeeze(0)
            mask = (torch.rand(self.max_seq_len) < 0.05) & (attention_mask == 1)
            seq_idx2[mask] = self.tokenizer.mask_token_id
            
            if len(self.pred) > 0:
                return seq_idx, seq_idx2, attention_mask, label, self.probability[idx]
            else:
                return seq_idx, seq_idx2, attention_mask, label
        else:
            return seq_idx, attention_mask, label

class ProteinDataLoader:
    def __init__(self, tokenizer, max_seq_len, dataset, batch_size, num_workers=6):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        
    def run(self, mode, pred=[], prob=[]):
        if mode == 'warmup':
            warmup_dataset = ProteinDataset(self.tokenizer, self.max_seq_len, self.dataset['train'], mode='warmup')
            warmup_loader = DataLoader(
                warmup_dataset, batch_size=self.batch_size,
                shuffle=True, num_workers=self.num_workers,
                drop_last=True
            )
            return warmup_loader
            
        elif mode == 'train':
            if len(pred) == 0:
                return self._create_train_loader()
                
            labeled_dataset = ProteinDataset(
                self.tokenizer, self.max_seq_len, self.dataset['train'], 
                mode='train', pred=pred, prob=prob
            )
            unlabeled_dataset = ProteinDataset(
                self.tokenizer, self.max_seq_len, self.dataset['train'], 
                mode='train', pred=~pred, prob=prob
            )
            
            labeled_loader = DataLoader(
                labeled_dataset, batch_size=self.batch_size, 
                shuffle=True, num_workers=self.num_workers,
                drop_last=True
            )
            
            unlabeled_loader = DataLoader(
                unlabeled_dataset, batch_size=self.batch_size,
                shuffle=True, num_workers=self.num_workers,
                drop_last=True
            )
            return labeled_loader, unlabeled_loader
            
        elif mode == 'val':
            return DataLoader(
                self.val_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=self.num_workers
            )
            
        elif mode == 'test':
            return DataLoader(
                self.test_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=self.num_workers
            )
            
        elif mode == 'eval_train':
            return DataLoader(
                ProteinDataset(self.tokenizer, self.max_seq_len, self.dataset['train'], mode='eval_train'),
                batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers
            )

def train(epoch, net, net2, net_type, plm_model, embedding_dict, optimizer, labeled_trainloader, unlabeled_trainloader):
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
            if 'plm' in net_type:
                embed_u = get_embedding_from_dict(embedding_dict, inputs_u)
                embed_u2 = plm_embedding(plm_model, inputs_u2, attention_mask_u)
                outputs_u11 = net(embed_u, attention_mask_u)
                outputs_u12 = net(embed_u2, attention_mask_u)
                outputs_u21 = net2(embed_u, attention_mask_u)
                outputs_u22 = net2(embed_u2, attention_mask_u)
            else:
                outputs_u11 = net(inputs_u)
                outputs_u12 = net(inputs_u2)
                outputs_u21 = net2(inputs_u)
                outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/args.T)
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()       
            
            if 'plm' in net_type:
                embed_x = get_embedding_from_dict(embedding_dict, inputs_x)
                embed_x2 = plm_embedding(plm_model, inputs_x2, attention_mask_x)
                outputs_x = net(embed_x, attention_mask_x)
                outputs_x2 = net(embed_x2, attention_mask_x)            
            else:
                outputs_x = net(inputs_x)
                outputs_x2 = net(inputs_x2)            
            
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
        
        if 'plm' in args.net_type:
            embed = plm_embedding(plm_model, mixed_input, mixed_attention_mask)
            logits = net(embed, mixed_attention_mask)
        else:
            logits = net(mixed_input)
        
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        
        # regularization
        prior = torch.ones(args.num_labels)/args.num_labels
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + penalty
        
        # gradient accumulation
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        sys.stdout.write('\r')
        sys.stdout.write('Protein | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.4f'
                %(epoch, args.num_epochs, batch_idx+1, num_iter, Lx.item()))
        sys.stdout.flush()

def warmup(net, net_type, embedding_dict, optimizer, dataloader):
    net.train()
    optimizer.zero_grad()
    
    for batch_idx, (inputs, attention_mask, labels) in enumerate(dataloader):      
        inputs, attention_mask, labels = inputs.cuda(), attention_mask.cuda(), labels.cuda()
        if 'plm' in net_type:
            embed = get_embedding_from_dict(embedding_dict, inputs)
            outputs = net(embed, attention_mask)
        else:
            outputs = net(inputs)
        loss = CEloss(outputs, labels)  
        
        penalty = conf_penalty(outputs)
        L = loss + penalty
        
        # gradient accumulation
        L = L / args.gradient_accumulation_steps
        L.backward()
        
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        sys.stdout.write('\r')
        sys.stdout.write('| Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
                %(batch_idx+1, len(dataloader), loss.item(), penalty.item()))
        sys.stdout.flush()

def val(net, net_type, embedding_dict, val_loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, attention_mask, targets) in enumerate(val_loader):
            inputs, attention_mask, targets = inputs.cuda(), attention_mask.cuda(), targets.cuda()
            if 'plm' in net_type:
                embed = get_embedding_from_dict(embedding_dict, inputs)
                outputs = net(embed, attention_mask)
            else:
                outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)         
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()              
    acc = 100.*correct/total
    return acc

def test(net1, net2, net_type, embedding_dict, test_loader):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, attention_mask, targets) in enumerate(test_loader):
            inputs, attention_mask, targets = inputs.cuda(), attention_mask.cuda(), targets.cuda()
            if 'plm' in net_type:
                embed = get_embedding_from_dict(embedding_dict, inputs)
                outputs1 = net1(embed, attention_mask)       
                outputs2 = net2(embed, attention_mask)           
            else:
                outputs1 = net1(inputs)       
                outputs2 = net2(inputs)           
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                    
    acc = 100.*correct/total
    print("\n| Test Acc: %.2f%%\n" %(acc))  
    return acc

@torch.no_grad()
def eval_train(net, net_type, embedding_dict, eval_loader):
    net.eval()
    losses = torch.zeros(len(eval_loader.dataset))
    with torch.no_grad():
        for batch_idx, (inputs, attention_mask, targets) in enumerate(eval_loader):
            inputs, attention_mask, targets = inputs.cuda(), attention_mask.cuda(), targets.cuda() 
            if 'plm' in net_type:
                embed = get_embedding_from_dict(embedding_dict, inputs)
                outputs = net(embed, attention_mask) 
            else:
                outputs = net(inputs) 
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


@torch.no_grad()
def plm_embedding(plm_model, aa_seq, attention_mask):
    outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask)
    seq_embeds = outputs.last_hidden_state
    gc.collect()
    torch.cuda.empty_cache()
    return seq_embeds

@torch.no_grad()
def pre_calculate_plm_embedding(plm_model, tokenizer, max_seq_len, dataset):
    print('Pre-calculating PLM embeddings...')
    aa_seq = list(dataset['train']['aa_seq']) + list(dataset['validation']['aa_seq']) + list(dataset['test']['aa_seq'])
    inputs = tokenizer(aa_seq, padding='max_length', max_length=max_seq_len, truncation=True, return_tensors='pt')
    # get embedding dict, key is each aa_seq input ids, value is each aa_seq embedding
    batch_size = 96
    embedding_dict = {}
    for i in tqdm(range(0, len(inputs['input_ids']), batch_size)):
        batch_inputs = inputs['input_ids'][i:i+batch_size].cuda()
        batch_attention_mask = inputs['attention_mask'][i:i+batch_size].cuda()
        batch_embeds = plm_embedding(plm_model, batch_inputs, batch_attention_mask)
        for input_ids, embed in zip(batch_inputs, batch_embeds):
            embedding_dict[str(input_ids.cpu())] = embed.cpu()
    return embedding_dict

def get_embedding_from_dict(embedding_dict, input_ids):
    embeds = []
    for input_id in input_ids:
        embeds.append(embedding_dict[str(input_id.cpu())])
    embeds = torch.stack(embeds).cuda()
    return embeds

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

class PredictorPLM(nn.Module):
    def __init__(self, plm_embed_dim, num_labels):
        super().__init__()
        self.classifier = Attention1dPoolingHead(plm_embed_dim, num_labels, 0.5)
    
    # x is plm embedding
    def forward(self, x, attention_mask):
        x = self.classifier(x, attention_mask)
        return x

class PredictorConv(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(33, 32)
        self.conv1 = nn.Conv1d(32, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128*256, 256)  # 1024/4 = 256 after 2 pooling layers
        self.fc2 = nn.Linear(256, num_labels)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
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
    def __init__(self, plm_embed_dim, num_labels):
        super().__init__()
        self.conv1 = nn.Conv1d(plm_embed_dim, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(128*256, 256)
        self.fc2 = nn.Linear(256, num_labels)
        self.dropout = nn.Dropout(0.5)
    
    # x is plm embedding
    def forward(self, x, attention_mask):
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
    parser = argparse.ArgumentParser(description='Protein Dividemix Training')
    parser.add_argument('--plm_model', default='facebook/esm2_t33_650M_UR50D', type=str, help='PLM model')
    parser.add_argument('--net_type', default='plm_attn1d', choices=['plm_attn1d', 'plm_conv', 'conv'], type=str, help='net type')
    parser.add_argument('--batch_size', default=4, type=int, help='train batchsize') 
    parser.add_argument('--max_seq_len', default=1024, type=int, help='max sequence length')
    parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
    parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--num_epochs', default=100, type=int, help='total epochs')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='warmup epochs')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    
    # dataset
    parser.add_argument('--num_labels', default=2, type=int, help='number of classes')
    parser.add_argument('--dataset_name', default='your_dataset_name', type=str, help='huggingface dataset name')
    parser.add_argument('--dataset_dir', default=None, type=str, help='path to dataset')
    parser.add_argument('--test_file', default='ExternalTest.csv', type=str, help='test file')
    parser.add_argument('--id', default='protein')
    parser.add_argument('--output_dir', default='ckpt')
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    log = open(f'{args.output_dir}/{args.id}.txt', 'w')
    log.flush()
    
    tokenizer = EsmTokenizer.from_pretrained(args.plm_model)
    if 'plm' in args.net_type:
        plm_model = EsmModel.from_pretrained(args.plm_model).cuda().eval()
    else:
        plm_model = None
    
    # build model
    print('| Building net')
    if args.net_type == 'plm_attn1d':
        net1 = PredictorPLM(plm_model.config.hidden_size, args.num_labels)
        net2 = PredictorPLM(plm_model.config.hidden_size, args.num_labels)
    elif args.net_type == 'plm_conv':
        net1 = PredictorPLMConv(plm_model.config.hidden_size, args.num_labels)
        net2 = PredictorPLMConv(plm_model.config.hidden_size, args.num_labels)
    elif args.net_type == 'conv':
        net1 = PredictorConv(args.num_labels)
        net2 = PredictorConv(args.num_labels)
    net1 = net1.cuda()
    net2 = net2.cuda()
    
    # Print model parameters
    total_params1 = sum(p.numel() for p in net1.parameters())
    total_params2 = sum(p.numel() for p in net2.parameters())
    print(f'Total parameters of net1: {total_params1/1e6:.2f}M')
    print(f'Total parameters of net2: {total_params2/1e6:.2f}M')
    log.write(f'Total parameters of net1: {total_params1/1e6:.2f}M\n')
    log.write(f'Total parameters of net2: {total_params2/1e6:.2f}M\n')
    log.flush()
    cudnn.benchmark = True

    optimizer1 = optim.AdamW(net1.parameters(), lr=args.lr)
    optimizer2 = optim.AdamW(net2.parameters(), lr=args.lr)

    CE = nn.CrossEntropyLoss(reduction='none')
    CEloss = nn.CrossEntropyLoss()
    conf_penalty = NegEntropy()
    
    best_acc = [0,0]
    
    # create data loader
    # train_file = os.path.join(args.dataset_dir, 'train.csv')
    # val_file = os.path.join(args.dataset_dir, 'valid.csv')
    # test_file = os.path.join(args.dataset_dir, args.test_file)
    if args.dataset_dir is not None:
        dataset = load_dataset(args.dataset_dir)
    else:
        dataset = load_dataset(args.dataset_name)
    loader = ProteinDataLoader(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        dataset=dataset,
        batch_size=args.batch_size
    )
    embedding_dict = pre_calculate_plm_embedding(plm_model, tokenizer, args.max_seq_len, dataset)
    
    for epoch in range(args.num_epochs+1):
        # warmup
        if epoch < args.warmup_epochs:
            train_loader = loader.run('warmup')
            print('Warmup Net1')
            warmup(net1, args.net_type, embedding_dict, optimizer1, train_loader)     
            train_loader = loader.run('warmup')
            print('\nWarmup Net2')
            warmup(net2, args.net_type, embedding_dict, optimizer2, train_loader)                 
        else:       
            pred1 = (prob1 > args.p_threshold)
            pred2 = (prob2 > args.p_threshold)
            print(pred1)
            print('\nTrain Net1')
            labeled_loader1, unlabeled_loader1 = loader.run('train', pred2, prob2)
            train(epoch, net1, net2, args.net_type, plm_model, embedding_dict, optimizer1, labeled_loader1, unlabeled_loader1)
            
            print('\nTrain Net2')
            labeled_loader2, unlabeled_loader2 = loader.run('train', pred1, prob1)
            train(epoch, net2, net1, args.net_type, plm_model, embedding_dict, optimizer2, labeled_loader2, unlabeled_loader2)
        
        # Validation
        val_loader = loader.run('val')
        acc1 = val(net1, args.net_type, embedding_dict, val_loader)
        print("\n| Validation\t Net1  Acc: %.2f%%" %(acc1))
        # save best model
        if acc1 > best_acc[0]:
            print('| >>> Saving Best Net1 ...')
            torch.save(net1.state_dict(), f'{args.output_dir}/{args.id}_net1.pth.tar')
            best_acc[0] = acc1
        
        
        acc2 = val(net2, args.net_type, embedding_dict, val_loader)
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
        prob1 = eval_train(net1, args.net_type, embedding_dict, eval_loader)
        
        print('\n==== net 2 evaluate next epoch training data loss ====')
        eval_loader = loader.run('eval_train')
        prob2 = eval_train(net2, args.net_type, embedding_dict, eval_loader)

        
    # Final test
    test_loader = loader.run('test')
    checkpoint1 = torch.load(f'{args.output_dir}/{args.id}_net1.pth.tar', weights_only=True)
    checkpoint2 = torch.load(f'{args.output_dir}/{args.id}_net2.pth.tar', weights_only=True)

    net1.load_state_dict(checkpoint1)
    net2.load_state_dict(checkpoint2)
    acc = test(net1, net2, args.net_type, embedding_dict, test_loader)
    
    log.write('Test %s Acc:%.2f\n'%(args.test_file, acc))
    log.close()
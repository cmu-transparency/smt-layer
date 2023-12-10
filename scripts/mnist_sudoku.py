import numpy as np
import numpy.random as npr

import os
import requests
import itertools
import random
import z3
import click
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from smtlayer import SMTLayer

class MNISTExtractor(nn.Module):
    def __init__(self, n_feats):
        super(MNISTExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 2)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.fc1 = nn.Linear(6272, 256)
        self.fc2 = nn.Linear(256, n_feats)
        
        torch.nn.init.orthogonal_(self.conv1.weight)
        torch.nn.init.orthogonal_(self.conv2.weight)
        torch.nn.init.orthogonal_(self.conv3.weight)
        torch.nn.init.orthogonal_(self.conv4.weight)
        torch.nn.init.orthogonal_(self.fc1.weight)
        torch.nn.init.orthogonal_(self.fc2.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.zeros_(self.conv2.bias)
        torch.nn.init.zeros_(self.conv3.bias)
        torch.nn.init.zeros_(self.conv4.bias)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out

class MNISTDecoder(nn.Module):
    
    def __init__(self, n_feats):
        super().__init__()
        
        self.fc1 = nn.Linear(n_feats, 64)
        self.fc2 = nn.Linear(64, 3*3*64)
        self.conv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=0)
        self.conv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
        
        torch.nn.init.orthogonal_(self.conv1.weight)
        torch.nn.init.orthogonal_(self.conv2.weight)
        torch.nn.init.orthogonal_(self.conv3.weight)
        torch.nn.init.orthogonal_(self.fc1.weight)
        torch.nn.init.orthogonal_(self.fc2.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.zeros_(self.conv2.bias)
        torch.nn.init.zeros_(self.conv3.bias)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)
                
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = x.view((x.shape[0],64,3,3))
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        
        return x

class MNISTSudokuSolver(nn.Module):
    def __init__(self, size=9):
        super(MNISTSudokuSolver, self).__init__()
        
        self.size = size
        
        self.extractor = MNISTExtractor(4)
        self.decoder = MNISTDecoder(4)
        self.fc1 = nn.Linear(4*size*size, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 4*size*size)
        
        inputs = [z3.Bool('v{}'.format(i)) for i in range(4*size*size)]
        cells = [[z3.BitVec('c{}_{}'.format(i,j), 4) for j in range(size)] for i in range(size)]
        zerobv = z3.BitVecVal(0, 1)
        onebv = z3.BitVecVal(1, 1)

        input_vectors = [[z3.Concat([z3.If(inputs[4*i*size+4*j+k], onebv, zerobv) for k in range(4)])
                          for j in range(size)] for i in range(size)]
        
        cell_cons = [z3.And([cells[i][j] == input_vectors[i][j] 
                         for (i,j) in itertools.product(range(self.size), range(self.size))])]
        val_cons = [z3.And(z3.ULE(z3.BitVecVal(1,4), cells[i][j]), 
                           z3.ULE(cells[i][j], z3.BitVecVal(self.size,4)))
                    for (i,j) in itertools.product(range(self.size),range(self.size))]
        row_cons = [z3.Distinct([cells[j][i] for i in range(self.size)]) for j in range(self.size)]
        col_cons = [z3.Distinct([cells[i][j] for i in range(self.size)]) for j in range(self.size)]

        if self.size == 3:
            group_cons = []
        elif self.size == 6:
            group_cons = [z3.Distinct([cells[r+y][c+x] for y, x in itertools.product(range(0, 2), range(0, 3))])
                          for r,c in itertools.product(range(0,self.size,2),range(0,self.size,3))]
        else:
            group_cons = [z3.Distinct([cells[r+y][c+x] for y, x in itertools.product(range(0, 3), range(0, 3))])
                          for r,c in itertools.product(range(0,self.size,3),range(0,self.size,3))]
                
        clauses = cell_cons+val_cons+row_cons+col_cons+group_cons
        variables = inputs
                
        self.sat = SMTLayer(
            input_size=4*size*size+size*size,
            output_size=4*size*size,
            theory=clauses,
            variables=variables,
            solverop='smt',
            default_mask=None)

        self.decode = MNISTDecoder(4)
    
    def forward(self, x, mask, return_sat=True, return_feats=False, 
                grad_scaling=None, do_maxsat=False):
        
        acts = [self.extractor(xi) for xi in x]
                
        combined = torch.cat(acts, dim=1)
        if return_feats:
            return combined
        else:
            if return_sat:
                is_input = torch.relu(mask.repeat_interleave(4,-1).view(mask.size(0),-1))
                out = self.sat(combined, 
                               mask=is_input,
                               grad_scaling=grad_scaling,
                               do_maxsat_forward=do_maxsat)
                
                full_out = []
                for i in range(out.size(0)):
                    imask = out[i] != 2
                    outi = out[i][imask]
                    curout = []
                    cptr, optr = 0, 0
                    for j in range(combined[i].size(0)):
                        if is_input[i][j] == 1:
                            curout.append(combined[i][j].unsqueeze(0))
                        else:
                            curout.append(outi[optr].unsqueeze(0))
                            optr += 1
                    full_out.append(torch.cat(curout).unsqueeze(0))

                return torch.cat(full_out, dim=0).to(x[0].device)

            else:
                out = F.relu(self.fc1(combined))
                out = F.relu(self.fc2(out))
                out = self.fc3(out)

            return out

def cached_tensors_available(prefix=''):
	path = os.path.join(prefix, 'data/sudoku_processed.pt')
	return os.path.exists(path)

def process_inputs(X, Ximg, Y, boardSz):

    if cached_tensors_available():
        Ximg, Y, is_input = torch.load('data/sudoku_processed.pt')
    else:
        bin_repr = np.vectorize(
            lambda x: np.array(list(np.binary_repr(x, 4)), dtype=np.float32), 
            signature="()->({})".format(4))

        is_input = (X.argmax(dim=-1).int().sign()-0.5).sign()
        y_int = Y.argmax(dim=-1)+1
        
        targ = []
        ybar = tqdm(range(len(Y)), total=len(Y))
        ybar.set_description('processing dataset')
        for i in ybar:
            t = []
            for j in range(boardSz):
                row = []
                for k in range(boardSz):
                    row.append(bin_repr(y_int[i,j,k].item()))
                t.append(row)
            targ.append(t)
        Y = torch.tensor(targ)

        Ximg = Ximg.flatten(start_dim=1, end_dim=2).unsqueeze(2).float() / 255.

        torch.save([Ximg, Y, is_input], 'data/sudoku_processed.pt')

    return Ximg, Y, is_input

@torch.no_grad()
def computeErr(pred_flat, n=3):

    nsq = n ** 2
    pred = pred_flat.view(-1, 81, 4)

    batchSz = pred.size(0)
    s = (nsq-1)*nsq//2 # 0 + 1 + ... + n^2-1
    I = (pred[:,:,3] + 2*pred[:,:,2] + 4*pred[:,:,1] + 8*pred[:,:,0] - 1).view(batchSz,nsq,nsq)

    def invalidGroups(x):
        valid = (x.min(1)[0] == 0)
        valid *= (x.max(1)[0] == nsq-1)
        valid *= (x.sum(1) == s)
        return valid.bitwise_not()

    boardCorrect = torch.ones(batchSz).bool()
    for j in range(nsq):
        # Check the jth row and column.
        boardCorrect[invalidGroups(I[:,j,:])] = False
        boardCorrect[invalidGroups(I[:,:,j])] = False

        # Check the jth block.
        row, col = n*(j // n), n*(j % n)
        M = invalidGroups(I[:,row:row+n,col:col+n].contiguous().view(batchSz,-1))
        boardCorrect[M] = False

    return boardCorrect

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print(f"ERROR: {filename} did not download correctly; downloaded {progress_bar.n} bytes, expected {total_size_in_bytes} bytes. If there are further errors, then delete this file and try again.")

def load(file):
    base_url = "https://github.com/cmu-transparency/smt-layer/releases/download/v0.0.1/"
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists(f'data/{file}'):
        print(f'Downloading {file}...')
        download_file(base_url + file, f'data/{file}')
    with open(f'data/{file}', 'rb') as f:
        return torch.load(f)

def get_dataloader(train_pct, batch_size=1):

    X_in = load('sudoku_features.pt')
    Ximg_in = load('sudoku_features_img.pt')
    Y_in = load('sudoku_labels.pt')
    perm = load('sudoku_perm.pt')

    Ximg, Y, is_input = process_inputs(X_in, Ximg_in, Y_in, 9)

    nTrain = int(Ximg.size(0)*train_pct)

    sud_train = TensorDataset(Ximg[:nTrain], is_input[:nTrain], Y[:nTrain])
    sud_test = TensorDataset(Ximg[max(9000,nTrain):], is_input[max(9000,nTrain):], Y[max(9000,nTrain):])
    train_load = torch.utils.data.DataLoader(sud_train, batch_size=batch_size, shuffle=True)
    test_load = torch.utils.data.DataLoader(sud_test, batch_size=batch_size)

    return train_load, test_load

def train_epoch(epoch_idx, model, optimizer, train_load, use_satlayer=True,
                clip_norm=None, sched=None, grad_scaling=None, do_maxsat_forward=False):

    tloader = tqdm(enumerate(train_load), total=len(train_load))
    acc_total = 0.
    loss_total = 0.
    total_samp = 0.
    batch_total = 0.
    
    model.train()
    
    for batch_idx, (data, mask, target) in tloader:

        data, target, mask = data.cuda(), target.cuda(), mask.cuda()

        if random.choice(range(10)) < 5:
            alldata = [data[i].unsqueeze(0) for i in range(len(data))]
            alltarget = [target[i].unsqueeze(0) for i in range(len(target))]
            allmask = [mask[i].unsqueeze(0) for i in range(len(mask))]
        else:
            alldata, alltarget, allmask = [data], [target], [mask]

        for i in range(len(alldata)):

            data, target, mask = alldata[i], alltarget[i], allmask[i]

            data = [data[:,i] for i in range(data.size(1))]
            target = target.view(target.size(0),-1)
            
            optimizer.zero_grad()
            output = model(data, torch.relu(mask), return_sat=use_satlayer, 
                            grad_scaling=grad_scaling, do_maxsat=do_maxsat_forward)

            errs = computeErr(torch.sign(torch.relu(output)))

            mask_inter = mask.repeat_interleave(4,-1).view(mask.size(0),-1)
            loss = (F.binary_cross_entropy_with_logits(output, target, reduction='none') * torch.relu(-mask_inter)).mean()
            loss.backward()
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

            acc = torch.sum(errs.float())
            
            acc_total += acc.item()
            loss_total += loss.item()
            total_samp += float(len(data[0]))
            batch_total += 1.
            
            tloader.set_description('train {} loss={:.4} acc={:.4} lr={:.4}'.format(epoch_idx, 
                loss_total/(batch_total), acc_total/total_samp, optimizer.param_groups[0]['lr']))
        
        if sched is not None:
            sched.step()

    train_acc = acc_total / total_samp
    train_loss = loss_total / (1+len(train_load))

    return train_acc, train_loss, tloader.format_dict['elapsed']

def test_epoch(epoch_idx, model, test_load, use_satlayer=True, do_maxsat=False):

    tloader = tqdm(enumerate(test_load), total=len(test_load))
    acc_total = 0.
    loss_total = 0.
    total_samp = 0.
    
    model.eval()
    
    for batch_idx, (data, mask, target) in tloader:
        with torch.no_grad():
            
            data, target, mask = data.cuda(), target.cuda(), mask.cuda()
            data = [data[:,i] for i in range(data.size(1))]
            target = target.view(target.size(0),-1)

            output = model(data, torch.relu(mask), return_sat=use_satlayer, do_maxsat=do_maxsat)

            errs = computeErr(torch.sign(torch.relu(output)))
            acc = torch.sum(errs.float())

            mask_inter = mask.repeat_interleave(4,-1).view(mask.size(0),-1)
            loss = (F.binary_cross_entropy_with_logits(output, target, reduction='none') * torch.relu(-mask_inter)).mean()
            
            acc_total += acc
            loss_total += loss.item()
            total_samp += float(len(data[0]))
            
            tloader.set_description('test {} loss={:.4} acc={:.4}'.format(epoch_idx, 
                loss_total/(batch_idx+1), acc_total/total_samp))
    
    test_acc = acc_total.cpu().detach().numpy()/total_samp
    test_loss = loss_total / (1+len(test_load))

    return test_acc, test_loss

def pretrain(model, optimizer, train_load, epochs, sched=None, clip_norm=None):

    for epoch in range(1, epochs+1):

        tloader = tqdm(enumerate(train_load), total=len(train_load))
        loss_total = 0.
        total_samp = 0.
        
        model.train()
        
        for batch_idx, (data, _, _) in tloader:

            data = data.cuda()
            data = [data[:,i] for i in range(data.size(1))]
            
            optimizer.zero_grad()
            output = [model.decoder(model.extractor(data[i])) for i in range(len(data))]
            loss = sum([F.mse_loss(output[i], data[i]) for i in range(len(data))])
            loss.backward()
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

            if sched is not None:
                try:
                    sched.step()
                except:
                    pass
            
            loss_total += loss.item()
            total_samp += float(len(data[0]))
            
            tloader.set_description('pretrain {} loss={:.4}'.format(epoch, 
                loss_total/(batch_idx+1)))

def train(model, optimizer, train_load, test_load, epochs, pct,
          use_satlayer=True, clip_norm=None, sched=None, do_sched_batch=False,
          grad_scaling=None, do_maxsat_forward=None):
    
    times = []
    prev_acc = None
    
    for epoch in range(1, epochs+1):

        if do_sched_batch:
            train_acc, train_loss, elapsed = train_epoch(epoch, model, optimizer, train_load, 
                                                use_satlayer=use_satlayer, clip_norm=clip_norm, sched=sched,
                                                grad_scaling=grad_scaling, do_maxsat_forward=do_maxsat_forward)
        else:
            train_acc, train_loss, elapsed = train_epoch(epoch, model, optimizer, train_load, 
                                                use_satlayer=use_satlayer, clip_norm=clip_norm, sched=None,
                                                grad_scaling=grad_scaling, do_maxsat_forward=do_maxsat_forward)
            if sched is not None:
                sched.step()

        test_acc, test_loss = test_epoch(epoch, model, test_load, use_satlayer=use_satlayer)
        
        prev_acc = train_acc
        times.append(elapsed)

        if train_acc > 0.999:
            break

    return train_acc, test_acc, sum(times)/float(epochs)

@click.command()
@click.option('--lr', default=1., show_default=True, help='Learning rate.')
@click.option('--pretrain_epochs', default=20, show_default=True, help='Number of pretraining epochs.')
@click.option('--pct', default=0.5, show_default=True, help='% pairs to use in training data.')
@click.option('--epochs', default=5, show_default=True, help='Number of epochs.')
@click.option('--batch_size', default=16, show_default=True, help='Batch size.')
@click.option('--trials', default=1, show_default=True, help='Number of trials.')
@click.option('--clip_norm', default=0.1, show_default=True, help='Gradient clipping norm.')
@click.option('--maxsat_forward', is_flag=True, show_default=True, help='Maxsat forward pass.')
@click.option('--maxsat_backward', is_flag=True, show_default=True, help='Maxsat backward pass.')
def main(
    lr=1.,
    pretrain_epochs=20,
    pct=0.5,
    epochs=20,
    batch_size=4,
    trials=1,
    clip_norm=0.1,
    maxsat_forward=False,
    maxsat_backward=False
):

    pretrain_load, _ = get_dataloader(0.9, batch_size=256)
    train_load, test_load = get_dataloader(pct, batch_size=batch_size)
    grad_scaling = None

    for i in range(trials):
        model = MNISTSudokuSolver(size=9).cuda()
        if pretrain_epochs > 0:
            # pre_optimizer = optim.SGD([{'params': model.parameters(), 'lr': lr, 'momentum': 0.9, 'nesterov': True}])
            pre_optimizer = optim.Adam([{'params': model.extractor.parameters(), 'lr': 1.e-3},
                                        {'params': model.decoder.parameters(), 'lr': 1.e-3}])
            pre_sched = torch.optim.lr_scheduler.OneCycleLR(pre_optimizer, 
                                                            1.e-3, 
                                                            epochs=pretrain_epochs, 
                                                            steps_per_epoch=len(pretrain_load), 
                                                            pct_start=0.5)
            pre_sched = None

            pretrain(model, pre_optimizer, pretrain_load, pretrain_epochs, sched=pre_sched, clip_norm=None)
        
        optimizer = optim.SGD([{'params': model.parameters(), 'lr': lr, 'momentum': 0.9, 'nesterov': True}])
        # optimizer = optim.Adam([{'params': model.parameters(), 'lr': lr}])
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    lr, 
                                                    epochs=epochs, 
                                                    steps_per_epoch=len(train_load), 
                                                    pct_start=1./float(epochs))
        
        train_acc, test_acc, elapsed = train(model, optimizer, train_load, test_load, epochs, pct,
                                                clip_norm=clip_norm, sched=sched, do_sched_batch=True, grad_scaling=grad_scaling, 
                                                do_maxsat_forward=maxsat_forward)

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        times.append(elapsed)

        print('\n[{} of 10]: train={:.4}, test={:.4}, time={:.4}\n'.format(i, train_acc, test_acc, elapsed))
        print('-'*20)

if __name__ == '__main__':
    main()
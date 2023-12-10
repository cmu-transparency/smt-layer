import numpy as np
import numpy.random as npr

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

class MNISTAddition(torch.utils.data.Dataset):
    def __init__(self, x_by_classes, label_pairs):
        super(MNISTAddition).__init__()
        
        self.by_classes = x_by_classes
        self.label_pairs = label_pairs
        
        self.len = sum([len(x_by_classes[cl]) for cl in x_by_classes])
        
        self.bin_repr = np.vectorize(
            lambda x: np.array(list(np.binary_repr(x, 5)), dtype=np.float32), 
            signature="()->({})".format(5))
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        pair = random.choice(self.label_pairs)
        inst1 = random.choice(self.by_classes[pair[0]]).unsqueeze(0)
        inst2 = random.choice(self.by_classes[pair[1]]).unsqueeze(0)
        
        return (inst1, inst2), torch.tensor(self.bin_repr(sum(pair))).float()

class MNISTExtractor(nn.Module):
    def __init__(self, n_feats):
        super(MNISTExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 2)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.fc1 = nn.Linear(6272, 256)
        self.fc2 = nn.Linear(256, n_feats)

        nn.init.orthogonal_(self.conv1.weight)
        nn.init.orthogonal_(self.conv2.weight)
        nn.init.orthogonal_(self.conv3.weight)
        nn.init.orthogonal_(self.conv4.weight)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)
        nn.init.zeros_(self.conv4.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out

class MNISTAdder(nn.Module):
    def __init__(self, use_maxsmt=False):
        super(MNISTAdder, self).__init__()
        self.extractor = MNISTExtractor(4)
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 5)
        
        inputs = z3.Bools('v0 v1 v2 v3 v4 v5 v6 v7')
        outputs = z3.Bools('v8 v9 v10 v11 v12')
        x0, x1, x2, x3, x4, x5, x6, x7 = inputs
        y0, y1, y2, y3, y4 = outputs
        z1, z2, y = z3.BitVecs('z1 z2 y', 5)
    
        cl9 = z1 == z3.Concat(z3.BitVecVal(0, 1),
                              z3.If(x0, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)),
                              z3.If(x1, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)),
                              z3.If(x2, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)),
                              z3.If(x3, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)))
        cl10 = z2 == z3.Concat(z3.BitVecVal(0, 1),
                               z3.If(x4, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)),
                               z3.If(x5, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)),
                               z3.If(x6, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)),
                               z3.If(x7, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)))
        cl11 = y == z1 + z2
        cl12 = y4 == (z3.Extract(0, 0, y) == z3.BitVecVal(1, 1))
        cl13 = y3 == (z3.Extract(1, 1, y) == z3.BitVecVal(1, 1))
        cl14 = y2 == (z3.Extract(2, 2, y) == z3.BitVecVal(1, 1))
        cl15 = y1 == (z3.Extract(3, 3, y) == z3.BitVecVal(1, 1))
        cl16 = y0 == (z3.Extract(4, 4, y) == z3.BitVecVal(1, 1))
        
        clauses = [cl9,cl10,cl11,cl12,cl13,cl14,cl15,cl16]
        mask = torch.tensor([1.,1.,1.,1.,1.,1.,1.,1.,0.,0.,0.,0.,0.])
        
        self.sat = SMTLayer(
            input_size=8,
            output_size=5,
            variables=inputs+outputs,
            theory=clauses,
            default_mask=mask,
            solverop='smt' if not use_maxsmt else 'maxsmt')
    
    def forward(self, x, return_sat=True, return_feats=False, do_maxsat=False):
        out1 = self.extractor(x[0])
        out2 = self.extractor(x[1])
        combined = torch.cat([out1, out2], dim=1)
        if return_feats:
            return combined
        else:
            if return_sat:
                pads = torch.zeros((x[0].shape[0], 5), dtype=out1.dtype, device=out1.device)
                combined = torch.cat([combined, pads], dim=1)
                out = self.sat(combined, do_maxsat_forward=do_maxsat)
            else:
                out = F.relu(self.fc1(combined))
                out = F.relu(self.fc2(out))
                out = self.fc3(out)

            return out

def get_dataloader(
    train_label_pairs, 
    test_label_pairs, 
    batch_size=128,
    data_dir='/data/data'
):

    transform = torchvision.transforms.Compose([
       torchvision.transforms.ToTensor()
    ])

    mnist_train = torchvision.datasets.MNIST(
        data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    mnist_test = torchvision.datasets.MNIST(
        data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )

    x_train, y_train = mnist_train.data/255., mnist_train.targets
    x_test, y_test = mnist_test.data/255., mnist_test.targets

    x_tr_by_classes = {cl: x_train[y_train==cl] for cl in range(10)}
    x_te_by_classes = {cl: x_test[y_test==cl] for cl in range(10)}

    train_data = MNISTAddition(x_tr_by_classes, train_label_pairs)
    test_data = MNISTAddition(x_te_by_classes, test_label_pairs)

    train_load = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_load = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_load, test_load

def train_epoch(epoch_idx, model, optimizer, train_load, use_satlayer=True,
                clip_norm=None, sched=None, do_maxsat_forward=False):

    tloader = tqdm(enumerate(train_load), total=len(train_load))
    acc_total = 0.
    loss_total = 0.
    total_samp = 0.
    
    model.train()
    
    for batch_idx, (data, target) in tloader:

        (data1, data2), target = data, target.cuda()
        data1, data2 = data1.cuda(), data2.cuda()
        data = (data1, data2)
        
        optimizer.zero_grad()
        output = model(data, return_sat=use_satlayer, do_maxsat=do_maxsat_forward)
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        if sched is not None:
            sched.step()

        with torch.no_grad():
            acc = torch.sum((torch.all(torch.sign(output) == 2*(target-0.5), dim=1)).type(torch.FloatTensor))
        
        acc_total += acc.item()
        loss_total += loss.item()
        total_samp += float(len(data1))
        
        tloader.set_description('train {} loss={:.4} acc={:.4} lr={:.4}'.format(epoch_idx, 
            loss_total/(batch_idx+1), acc_total/total_samp, optimizer.param_groups[0]['lr']))

    train_acc = acc_total / total_samp
    train_loss = loss_total / (1+len(train_load))

    return train_acc, train_loss, tloader.format_dict['elapsed']

def test_epoch(epoch_idx, model, test_load, use_satlayer=True):

    tloader = tqdm(enumerate(test_load), total=len(test_load))
    acc_total = 0.
    loss_total = 0.
    total_samp = 0.
    
    model.eval()
    
    for batch_idx, (data, target) in tloader:
        with torch.no_grad():
            
            (data1, data2), target = data, target.cuda()
            data1, data2 = data1.cuda(), data2.cuda()
            data = (data1, data2)
                        
            output = model(data, return_sat=use_satlayer, do_maxsat=True)

            loss = F.binary_cross_entropy_with_logits(output, target)
            acc = torch.sum((torch.all(torch.sign(output) == 2*(target-0.5), dim=1)).type(torch.FloatTensor))
            label = torch.argmax(output, dim=1)            
            
            acc_total += acc
            loss_total += loss.item()
            total_samp += float(len(data1))
            
            tloader.set_description('test {} loss={:.4} acc={:.4}'.format(epoch_idx, 
                loss_total/(batch_idx+1), acc_total/total_samp))
    
    test_acc = acc_total.cpu().detach().numpy()/total_samp
    test_loss = loss_total / (1+len(test_load))

    return test_acc, test_loss

def pretrain(model, optimizer, train_load, epochs, clip_norm=None):

    for epoch in range(1, epochs+1):

        tloader = tqdm(enumerate(train_load), total=len(train_load))
        acc_total = 0.
        loss_total = 0.
        total_samp = 0.
        
        model.train()
        
        for batch_idx, (data, target) in tloader:

            (data1, data2), target = data, target.cuda()
            data1, data2 = data1.cuda(), data2.cuda()
            data = (data1, data2)
            
            optimizer.zero_grad()
            output = model(data, return_sat=False)
            loss = F.binary_cross_entropy_with_logits(output, target)
            loss.backward()
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

            with torch.no_grad():
                acc = torch.sum((torch.all(torch.sign(output) == 2*(target-0.5), dim=1)).type(torch.FloatTensor))
            
            acc_total += acc.item()
            loss_total += loss.item()
            total_samp += float(len(data1))
            
            tloader.set_description('pretrain {} loss={:.4} acc={:.4}'.format(epoch, 
                loss_total/(batch_idx+1), acc_total/total_samp))

def train(model, optimizer, train_load, test_load, epochs,
          use_satlayer=True, clip_norm=None, sched=None, do_sched_batch=False,
          do_maxsat_forward=None):
    
    times = []
    
    for epoch in range(1, epochs+1):

        if do_sched_batch:
            train_acc, train_loss, elapsed = train_epoch(epoch, model, optimizer, train_load, 
                                                use_satlayer=use_satlayer, clip_norm=clip_norm, sched=sched,
                                                do_maxsat_forward=do_maxsat_forward)
        else:
            train_acc, train_loss, elapsed = train_epoch(epoch, model, optimizer, train_load, 
                                                use_satlayer=use_satlayer, clip_norm=clip_norm, sched=None,
                                                do_maxsat_forward=do_maxsat_forward)
            if sched is not None:
                sched.step()

        test_acc, test_loss = test_epoch(epoch, model, test_load, use_satlayer=use_satlayer)

        if train_acc > 0.999:
            break

    return train_acc, test_acc, sum(times)/float(epochs)

@click.command()
@click.option('--lr', default=1., show_default=True, help='Learning rate.')
@click.option('--pretrain_epochs', default=0, show_default=True, help='Number of pretraining epochs.')
@click.option('--pct', default=10, show_default=True, help='% pairs to use in training data.')
@click.option('--epochs', default=5, show_default=True, help='Number of epochs.')
@click.option('--batch_size', default=128, show_default=True, help='Batch size.')
@click.option('--trials', default=1, show_default=True, help='Number of trials.')
@click.option('--clip_norm', default=0.1, show_default=True, help='Gradient clipping norm.')
@click.option('--maxsat_forward', is_flag=True, show_default=True, help='Maxsat forward pass.')
@click.option('--maxsat_backward', is_flag=True, show_default=True, help='Maxsat backward pass.')
def main(
    lr=1.,
    pretrain_epochs=0,
    pct=10,
    epochs=5,
    batch_size=128,
    trials=1,
    clip_norm=0.1,
    maxsat_forward=False,
    maxsat_backward=False
):

    test_label_pairs = list(itertools.product(list(range(0,10)), repeat=2))
    if pct <= 10:
        train_label_pairs = [(0,0), (1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9)]
    else:
        train_label_pairs = test_label_pairs[:int(pct)]
    train_load, test_load = get_dataloader(train_label_pairs, test_label_pairs, batch_size=batch_size)
    pretrain_load, _ = get_dataloader(train_label_pairs, test_label_pairs, batch_size=512)

    print('')
    print('-'*30)
    print('new training sample')
    print(train_label_pairs)
    print('-'*30)

    train_accs = []
    test_accs = []
    times = []

    for i in range(trials):
        model = MNISTAdder(use_maxsmt=maxsat_backward).cuda()
        optimizer = optim.SGD([{'params': model.parameters(), 'lr': lr, 'momentum': 0.9, 'nesterov': True}])
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    lr, 
                                                    epochs=epochs, 
                                                    steps_per_epoch=len(train_load), 
                                                    pct_start=1./float(epochs))

        pretrain(model, optimizer, train_load, pretrain_epochs, clip_norm=clip_norm)
        train_acc, test_acc, elapsed = train(
            model, optimizer, train_load, test_load, epochs,
            clip_norm=clip_norm, sched=sched, do_sched_batch=True,
            do_maxsat_forward=maxsat_forward
        )

        train_accs.append(train_acc)
        test_accs.append(test_acc)
        times.append(elapsed)

        print('\n[{} of {trials}]: train={:.4}, test={:.4}, time={:.4}\n'.format(i, train_acc, test_acc, elapsed))
        print('-'*20)

    train_accs = np.array(train_accs)
    test_accs = np.array(test_accs)
    times = np.array(times)

    print('\nsmt pct={} stats: train={:.4} ({:.8}), test={:.4} ({:.8}), time={:.4} ({:.8})'.format(
            pct, train_accs.mean(), train_accs.std(), test_accs.mean(), test_accs.std(), times.mean(), times.std()))

if __name__ == '__main__':
    main()

<<<<<<< HEAD
import numpy as np
import argparse
import os
import imp
import re
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic=True

from utils.mortality import utils
from utils.mortality.readers import InHospitalMortalityReader
from utils.mortality.preprocessing import Discretizer, Normalizer
from utils.mortality import metrics
from utils.mortality import common_utils
from model import model_mortality

def parse_arguments(parser):
    parser.add_argument('--data_path', type=str, metavar='<data_path>', help='The path to the MIMIC-III data directory')
    parser.add_argument('--save_path', type=str, metavar='<data_path>', help='The path to save the model')
    parser.add_argument('--small_part', type=int, default=0, help='Use part of training data')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learing rate')

    parser.add_argument('--cell', type=str, default='gru', help='RNN cell')
    parser.add_argument('--input_dim', type=int, default=76, help='Dimension of visit record data')
    parser.add_argument('--demo_dim', type=int, default=12, help='Dimension of demographic data')
    parser.add_argument('--rnn_dim', type=int, default=32, help='Dimension of hidden units in RNN')
    parser.add_argument('--agent_dim', type=int, default=32, help='Dimension of MLP in two agents')
    parser.add_argument('--mlp_dim', type=int, default=20, help='Dimension of MLP for output')
    parser.add_argument('--output_dim', type=int, default=1, help='Dimension of prediction target')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--lamda', type=float, default=1.0, help='Value of hyper-parameter lamda')
    parser.add_argument('--K', type=int, default=10, help='Value of hyper-parameter K')
    parser.add_argument('--gamma', type=float, default=0.9, help='Value of hyper-parameter gamma')
    parser.add_argument('--entropy_term', type=float, default=0.01, help='Value of reward entropy term')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    ''' Prepare training data'''
    print('Preparing training data ... ')
    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data_path, 'train'),
                                            listfile=os.path.join(args.data_path, 'train_listfile.csv'),
                                            period_length=48.0)

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data_path, 'train'),
                                        listfile=os.path.join(args.data_path, 'val_listfile.csv'),
                                        period_length=48.0)

    discretizer = Discretizer(timestep=1.0,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')    
    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    normalizer_state = 'ihm_normalizer'
    normalizer_state = os.path.join(os.path.dirname(args.data_path), normalizer_state)
    normalizer.load_params(normalizer_state)

    train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part, return_names=True)
    val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part, return_names=True)
    demographic_data = []
    diagnosis_data = []
    idx_list = []

    demo_path = args.data_path + 'demographic/'
    for cur_name in os.listdir(demo_path):
        cur_id, cur_episode = cur_name.split('_', 1)
        cur_episode = cur_episode[:-4]
        cur_file = demo_path + cur_name

        with open(cur_file, "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            if header[0] != "Icustay":
                continue
            cur_data = tsfile.readline().strip().split(',')
            
        if len(cur_data) == 1:
            cur_demo = np.zeros(12)
            cur_diag = np.zeros(128)
        else:
            if cur_data[3] == '':
                cur_data[3] = 60.0
            if cur_data[4] == '':
                cur_data[4] = 160
            if cur_data[5] == '':
                cur_data[5] = 60

            cur_demo = np.zeros(12)
            cur_demo[int(cur_data[1])] = 1
            cur_demo[5 + int(cur_data[2])] = 1
            cur_demo[9:] = cur_data[3:6]
            cur_diag = np.array(cur_data[8:], dtype=np.int)

        demographic_data.append(cur_demo)
        diagnosis_data.append(cur_diag)
        idx_list.append(cur_id+'_'+cur_episode)

    for each_idx in range(9,12):
        cur_val = []
        for i in range(len(demographic_data)):
            cur_val.append(demographic_data[i][each_idx])
        cur_val = np.array(cur_val)
        _mean = np.mean(cur_val)
        _std = np.std(cur_val)
        _std = _std if _std > 1e-7 else 1e-7
        for i in range(len(demographic_data)):
            demographic_data[i][each_idx] = (demographic_data[i][each_idx] - _mean) / _std

    class Dataset(data.Dataset):
        def __init__(self, x, y, name):
            self.x = x
            self.y = y
            self.name = name

        def __getitem__(self, index):
            return self.x[index], self.y[index], self.name[index]

        def __len__(self):
            return len(self.x)
        
    train_dataset = Dataset(train_raw['data'][0], train_raw['data'][1], train_raw['names'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = Dataset(val_raw['data'][0], val_raw['data'][1], val_raw['names'])
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)


    '''Model structure'''
    print('Constructing model ... ')
    device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
    print("available device: {}".format(device))

    model = model_mortality.Agent(cell=args.cell,
                        use_baseline=True,
                        n_actions=args.K,
                        n_units=args.agent_dim,
                        n_input=args.input_dim,
                        demo_dim=args.demo_dim,
                        fusion_dim=args.mlp_dim,
                        n_hidden=args.rnn_dim,
                        n_output=args.output_dim,
                        dropout=args.dropout_rate,
                        lamda=args.lamda, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    '''Train phase'''
    print('Start training ... ')

    train_loss = []
    valid_loss = []
    max_auroc = 0

    file_name = args.save_path+'/agent_mortality'
    for each_chunk in range(args.epochs):
        batch_loss = []
        batch_rlloss = []
        batch_taskloss = []
        model.train()
        for step, (batch_x, batch_y, batch_name) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            batch_demo = []
            for i in range(len(batch_name)):
                cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                cur_idx = cur_id + '_' + cur_ep
                cur_demo = torch.tensor(demographic_data[idx_list.index(cur_idx)], dtype=torch.float32)
                batch_demo.append(cur_demo)   
            batch_demo = torch.stack(batch_demo).to(device)

            optimizer.zero_grad()
            cur_output = model(batch_x, batch_demo)

            loss, loss_rl, loss_task = model_mortality.get_loss(model, cur_output, batch_y.unsqueeze(-1), gamma=args.gamma, entropy_term=args.entropy_term, use_baseline=True, device=device)            
            batch_loss.append(loss.cpu().detach().numpy())
            batch_rlloss.append(loss_rl.cpu().detach().numpy())
            batch_taskloss.append(loss_task.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
            
            if each_chunk % 10 == 0:
                print('Epoch %d Batch %d: Train Loss = %.4f Task Loss = %.4f RL Loss = %.4f'%(each_chunk, step, np.mean(np.array(batch_loss)), np.mean(np.array(batch_taskloss)), np.mean(np.array(batch_rlloss))))

        train_loss.append(np.mean(np.array(batch_taskloss)))
        
        batch_loss = []
        batch_rlloss = []
        batch_taskloss = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            model.eval()
            for step, (batch_x, batch_y, batch_name) in enumerate(valid_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                
                batch_demo = []
                for i in range(len(batch_name)):
                    cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                    cur_idx = cur_id + '_' + cur_ep
                    cur_demo = torch.tensor(demographic_data[idx_list.index(cur_idx)], dtype=torch.float32)
                    batch_demo.append(cur_demo)

                batch_demo = torch.stack(batch_demo).to(device)
                output = model(batch_x, batch_demo)

                loss, loss_rl, loss_task = model_mortality.get_loss(model, output, batch_y.unsqueeze(-1), gamma=args.gamma, entropy_term=args.entropy_term, use_baseline=True, device=device)
                batch_loss.append(loss.cpu().detach().numpy())
                batch_rlloss.append(loss_rl.cpu().detach().numpy())
                batch_taskloss.append(loss_task.cpu().detach().numpy())
                y_pred += list(output.cpu().detach().numpy().flatten())
                y_true += list(batch_y.cpu().numpy().flatten())
                
        valid_loss.append(np.mean(np.array(batch_taskloss)))
        print("\n==>Predicting on validation")
        print('Valid Loss = %.4f Task Loss = %.4f RL Loss = %.4f'%(valid_loss[-1], np.mean(np.array(batch_taskloss)), np.mean(np.array(batch_rlloss))))
        y_pred = np.array(y_pred)
        y_pred = np.stack([1 - y_pred, y_pred], axis=1)
        ret = metrics.print_metrics_binary(y_true, y_pred)
        print()

        cur_auroc = ret['auroc']
        if cur_auroc > max_auroc:
            max_auroc = cur_auroc
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': each_chunk
            }
            torch.save(state, file_name)
            print('\n------------ Save best model ------------\n')


    '''Evaluate phase'''
    print('Testing model ... ')

    checkpoint = torch.load(file_name)
    save_chunk = checkpoint['epoch']
    print("last saved model is in epoch {}".format(save_chunk))
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data_path, 'test'),
                                            listfile=os.path.join(args.data_path, 'test_listfile.csv'),
                                            period_length=48.0)
    test_raw = utils.load_data(test_reader, discretizer, normalizer, args.small_part, return_names=True)
    test_dataset = Dataset(test_raw['data'][0], test_raw['data'][1], test_raw['names'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    batch_loss = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        model.eval()
        for step, (batch_x, batch_y, batch_name) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_demo = []
            for i in range(len(batch_name)):
                cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                cur_idx = cur_id + '_' + cur_ep
                cur_demo = torch.tensor(demographic_data[idx_list.index(cur_idx)], dtype=torch.float32)
                batch_demo.append(cur_demo)

            batch_demo = torch.stack(batch_demo).to(device)
            output = model(batch_x, batch_demo)

            loss, loss_rl, loss_task = model_mortality.get_loss(model, output, batch_y.unsqueeze(-1), gamma=args.gamma, entropy_term=args.entropy_term, use_baseline=True, device=device)
            batch_loss.append(loss.cpu().detach().numpy())
            y_pred += list(output.cpu().detach().numpy().flatten())
            y_true += list(batch_y.cpu().numpy().flatten())

    print("\n==>Predicting on test")
    print('Test Loss = %.4f'%(np.mean(np.array(batch_loss))))
    y_pred = np.array(y_pred)
    y_pred = np.stack([1 - y_pred, y_pred], axis=1)
    test_res = metrics.print_metrics_binary(y_true, y_pred)
=======
import numpy as np
import argparse
import os
import imp
import re
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic=True

from utils.mortality import utils
from utils.mortality.readers import InHospitalMortalityReader
from utils.mortality.preprocessing import Discretizer, Normalizer
from utils.mortality import metrics
from utils.mortality import common_utils
from model import model_mortality

def parse_arguments(parser):
    parser.add_argument('--data_path', type=str, metavar='<data_path>', help='The path to the MIMIC-III data directory')
    parser.add_argument('--save_path', type=str, metavar='<data_path>', help='The path to save the model')
    parser.add_argument('--small_part', type=int, default=0, help='Use part of training data')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learing rate')

    parser.add_argument('--cell', type=str, default='gru', help='RNN cell')
    parser.add_argument('--input_dim', type=int, default=76, help='Dimension of visit record data')
    parser.add_argument('--demo_dim', type=int, default=12, help='Dimension of demographic data')
    parser.add_argument('--rnn_dim', type=int, default=32, help='Dimension of hidden units in RNN')
    parser.add_argument('--agent_dim', type=int, default=32, help='Dimension of MLP in two agents')
    parser.add_argument('--mlp_dim', type=int, default=20, help='Dimension of MLP for output')
    parser.add_argument('--output_dim', type=int, default=1, help='Dimension of prediction target')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--lamda', type=float, default=1.0, help='Value of hyper-parameter lamda')
    parser.add_argument('--K', type=int, default=10, help='Value of hyper-parameter K')
    parser.add_argument('--gamma', type=float, default=0.9, help='Value of hyper-parameter gamma')
    parser.add_argument('--entropy_term', type=float, default=0.01, help='Value of reward entropy term')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    ''' Prepare training data'''
    print('Preparing training data ... ')
    train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data_path, 'train'),
                                            listfile=os.path.join(args.data_path, 'train_listfile.csv'),
                                            period_length=48.0)

    val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data_path, 'train'),
                                        listfile=os.path.join(args.data_path, 'val_listfile.csv'),
                                        period_length=48.0)

    discretizer = Discretizer(timestep=1.0,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')    
    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
    cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

    normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
    normalizer_state = 'ihm_normalizer'
    normalizer_state = os.path.join(os.path.dirname(args.data_path), normalizer_state)
    normalizer.load_params(normalizer_state)

    train_raw = utils.load_data(train_reader, discretizer, normalizer, args.small_part, return_names=True)
    val_raw = utils.load_data(val_reader, discretizer, normalizer, args.small_part, return_names=True)
    demographic_data = []
    diagnosis_data = []
    idx_list = []

    demo_path = args.data_path + 'demographic/'
    for cur_name in os.listdir(demo_path):
        cur_id, cur_episode = cur_name.split('_', 1)
        cur_episode = cur_episode[:-4]
        cur_file = demo_path + cur_name

        with open(cur_file, "r") as tsfile:
            header = tsfile.readline().strip().split(',')
            if header[0] != "Icustay":
                continue
            cur_data = tsfile.readline().strip().split(',')
            
        if len(cur_data) == 1:
            cur_demo = np.zeros(12)
            cur_diag = np.zeros(128)
        else:
            if cur_data[3] == '':
                cur_data[3] = 60.0
            if cur_data[4] == '':
                cur_data[4] = 160
            if cur_data[5] == '':
                cur_data[5] = 60

            cur_demo = np.zeros(12)
            cur_demo[int(cur_data[1])] = 1
            cur_demo[5 + int(cur_data[2])] = 1
            cur_demo[9:] = cur_data[3:6]
            cur_diag = np.array(cur_data[8:], dtype=np.int)

        demographic_data.append(cur_demo)
        diagnosis_data.append(cur_diag)
        idx_list.append(cur_id+'_'+cur_episode)

    for each_idx in range(9,12):
        cur_val = []
        for i in range(len(demographic_data)):
            cur_val.append(demographic_data[i][each_idx])
        cur_val = np.array(cur_val)
        _mean = np.mean(cur_val)
        _std = np.std(cur_val)
        _std = _std if _std > 1e-7 else 1e-7
        for i in range(len(demographic_data)):
            demographic_data[i][each_idx] = (demographic_data[i][each_idx] - _mean) / _std

    class Dataset(data.Dataset):
        def __init__(self, x, y, name):
            self.x = x
            self.y = y
            self.name = name

        def __getitem__(self, index):
            return self.x[index], self.y[index], self.name[index]

        def __len__(self):
            return len(self.x)
        
    train_dataset = Dataset(train_raw['data'][0], train_raw['data'][1], train_raw['names'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = Dataset(val_raw['data'][0], val_raw['data'][1], val_raw['names'])
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)


    '''Model structure'''
    print('Constructing model ... ')
    device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
    print("available device: {}".format(device))

    model = model_mortality.Agent(cell=args.cell,
                        use_baseline=True,
                        n_actions=args.K,
                        n_units=args.agent_dim,
                        n_input=args.input_dim,
                        demo_dim=args.demo_dim,
                        fusion_dim=args.mlp_dim,
                        n_hidden=args.rnn_dim,
                        n_output=args.output_dim,
                        dropout=args.dropout_rate,
                        lamda=args.lamda, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    '''Train phase'''
    print('Start training ... ')

    train_loss = []
    valid_loss = []
    max_auroc = 0

    file_name = args.save_path+'/agent_mortality'
    for each_chunk in range(args.epochs):
        batch_loss = []
        batch_rlloss = []
        batch_taskloss = []
        model.train()
        for step, (batch_x, batch_y, batch_name) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            batch_demo = []
            for i in range(len(batch_name)):
                cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                cur_idx = cur_id + '_' + cur_ep
                cur_demo = torch.tensor(demographic_data[idx_list.index(cur_idx)], dtype=torch.float32)
                batch_demo.append(cur_demo)   
            batch_demo = torch.stack(batch_demo).to(device)

            optimizer.zero_grad()
            cur_output = model(batch_x, batch_demo)

            loss, loss_rl, loss_task = model_mortality.get_loss(model, cur_output, batch_y.unsqueeze(-1), gamma=args.gamma, entropy_term=args.entropy_term, use_baseline=True, device=device)            
            batch_loss.append(loss.cpu().detach().numpy())
            batch_rlloss.append(loss_rl.cpu().detach().numpy())
            batch_taskloss.append(loss_task.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
            
            if each_chunk % 10 == 0:
                print('Epoch %d Batch %d: Train Loss = %.4f Task Loss = %.4f RL Loss = %.4f'%(each_chunk, step, np.mean(np.array(batch_loss)), np.mean(np.array(batch_taskloss)), np.mean(np.array(batch_rlloss))))

        train_loss.append(np.mean(np.array(batch_taskloss)))
        
        batch_loss = []
        batch_rlloss = []
        batch_taskloss = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            model.eval()
            for step, (batch_x, batch_y, batch_name) in enumerate(valid_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                
                batch_demo = []
                for i in range(len(batch_name)):
                    cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                    cur_idx = cur_id + '_' + cur_ep
                    cur_demo = torch.tensor(demographic_data[idx_list.index(cur_idx)], dtype=torch.float32)
                    batch_demo.append(cur_demo)

                batch_demo = torch.stack(batch_demo).to(device)
                output = model(batch_x, batch_demo)

                loss, loss_rl, loss_task = model_mortality.get_loss(model, output, batch_y.unsqueeze(-1), gamma=args.gamma, entropy_term=args.entropy_term, use_baseline=True, device=device)
                batch_loss.append(loss.cpu().detach().numpy())
                batch_rlloss.append(loss_rl.cpu().detach().numpy())
                batch_taskloss.append(loss_task.cpu().detach().numpy())
                y_pred += list(output.cpu().detach().numpy().flatten())
                y_true += list(batch_y.cpu().numpy().flatten())
                
        valid_loss.append(np.mean(np.array(batch_taskloss)))
        print("\n==>Predicting on validation")
        print('Valid Loss = %.4f Task Loss = %.4f RL Loss = %.4f'%(valid_loss[-1], np.mean(np.array(batch_taskloss)), np.mean(np.array(batch_rlloss))))
        y_pred = np.array(y_pred)
        y_pred = np.stack([1 - y_pred, y_pred], axis=1)
        ret = metrics.print_metrics_binary(y_true, y_pred)
        print()

        cur_auroc = ret['auroc']
        if cur_auroc > max_auroc:
            max_auroc = cur_auroc
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': each_chunk
            }
            torch.save(state, file_name)
            print('\n------------ Save best model ------------\n')


    '''Evaluate phase'''
    print('Testing model ... ')

    checkpoint = torch.load(file_name)
    save_chunk = checkpoint['epoch']
    print("last saved model is in epoch {}".format(save_chunk))
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()

    test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data_path, 'test'),
                                            listfile=os.path.join(args.data_path, 'test_listfile.csv'),
                                            period_length=48.0)
    test_raw = utils.load_data(test_reader, discretizer, normalizer, args.small_part, return_names=True)
    test_dataset = Dataset(test_raw['data'][0], test_raw['data'][1], test_raw['names'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    batch_loss = []
    y_true = []
    y_pred = []
    with torch.no_grad():
        model.eval()
        for step, (batch_x, batch_y, batch_name) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_demo = []
            for i in range(len(batch_name)):
                cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                cur_idx = cur_id + '_' + cur_ep
                cur_demo = torch.tensor(demographic_data[idx_list.index(cur_idx)], dtype=torch.float32)
                batch_demo.append(cur_demo)

            batch_demo = torch.stack(batch_demo).to(device)
            output = model(batch_x, batch_demo)

            loss, loss_rl, loss_task = model_mortality.get_loss(model, output, batch_y.unsqueeze(-1), gamma=args.gamma, entropy_term=args.entropy_term, use_baseline=True, device=device)
            batch_loss.append(loss.cpu().detach().numpy())
            y_pred += list(output.cpu().detach().numpy().flatten())
            y_true += list(batch_y.cpu().numpy().flatten())

    print("\n==>Predicting on test")
    print('Test Loss = %.4f'%(np.mean(np.array(batch_loss))))
    y_pred = np.array(y_pred)
    y_pred = np.stack([1 - y_pred, y_pred], axis=1)
    test_res = metrics.print_metrics_binary(y_true, y_pred)
>>>>>>> 742f7a833bbc02da15cd7752dc1d2cdd8606aeb1

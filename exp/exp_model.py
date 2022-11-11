from data.data_loader import Dataset_ETT_hour, Dataset_ETT_min, Dataset_Custom
from exp.exp_basic import Exp_Basic
from FDNet.model import FDNet

from utils.tools import EarlyStopping, adjust_learning_rate, loss_process
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import pandas as pd
import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_Model(Exp_Basic):
    def __init__(self, args):
        super(Exp_Model, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'FDNet': FDNet,
        }
        if self.args.model == 'FDNet':
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.c_out,
                self.args.label_len,
                self.args.pred_len,
                self.args.kernel,
                self.args.attn_nums,
                self.args.timebed,
                self.args.d_model,
                self.args.pyramid,
                self.args.ICOM,
                self.args.dropout,
            ).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args
        data_set = None

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_min,
            'ETTm2': Dataset_ETT_min,
            'weather': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Exchange': Dataset_Custom,
            'Traffic': Dataset_Custom
        }
        Data = data_dict[self.args.data]

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.label_len, args.pred_len],
            features=args.features,
            timebed=args.timebed,
            target=args.target,
            criterion=args.criterion
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data=None, vali_loader=None, criterion=None):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, embed) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    batch_x, embed)
                loss = loss_process(pred, true, criterion, flag=1)
                total_loss.append(loss)

            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = self._select_optimizer()

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        criterion = self._select_criterion()
        for epoch in range(self.args.train_epochs):
            iter_count = 0

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, embed) in enumerate(train_loader):
                model_optim.zero_grad()
                iter_count += 1
                pred, true = self._process_one_batch(
                    batch_x, embed)
                TE = (pred - true) ** 2 + torch.abs(pred - true)
                loss = torch.mean(TE)

                loss.backward(torch.ones_like(loss))
                model_optim.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1,
                                                                            torch.mean(loss).item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Pred_len: {0} | Epoch: {1}, Steps: {2} | Vali Loss: {3:.7f} Test Loss: {4:.7f}".
                  format(self.args.pred_len, epoch + 1, train_steps, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, load=False, write_loss=True, save_loss=True):
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        test_data, test_loader = self._get_data(flag='test')
        criterion = self._select_criterion()
        if save_loss:
            preds = []
            trues = []

            with torch.no_grad():
                for i, (batch_x, embed) in enumerate(test_loader):
                    pred, true = self._process_one_batch(
                        batch_x, embed)
                    if self.args.test_inverse:
                        pred = loss_process(pred, true, criterion, flag=2, dataset=test_data)
                        pred = pred.reshape(self.args.batch_size, self.args.pred_len, self.args.c_out)
                        true = true.reshape(-1, pred.shape[-1])
                        true = test_data.inverse_transform(true.detach().cpu().numpy())
                        true = test_data.standard_transformer(true)
                        true = true.reshape(self.args.batch_size, self.args.pred_len, self.args.c_out)
                    else:
                        pred = loss_process(pred, true, criterion, flag=2)
                        pred = pred.detach().cpu().numpy()
                        true = true.detach().cpu().numpy()
                    preds.append(pred)
                    trues.append(true)
            preds = np.array(preds)
            trues = np.array(trues)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

            mae, mse = metric(preds, trues)
            print('|{}_{}|pred_len{}|mse:{}, mae:{}'.
                  format(self.args.data, self.args.features, self.args.pred_len, mse, mae) + '\n')
            # result save
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            np.save(folder_path + f'metrics.npy', np.array([mae, mse]))
            np.save(folder_path + f'pred.npy', preds)
            np.save(folder_path + f'true.npy', trues)
        else:
            mse_list = []
            mae_list = []
            with torch.no_grad():
                for i, (batch_x, embed) in enumerate(test_loader):
                    pred, true = self._process_one_batch(
                        batch_x, embed)
                    if self.args.test_inverse:
                        pred = loss_process(pred, true, criterion, flag=2, dataset=test_data)
                        pred = pred.reshape(self.args.batch_size, self.args.pred_len, self.args.c_out)
                        true = true.reshape(-1, pred.shape[-1])
                        true = test_data.inverse_transform(true.detach().cpu().numpy())
                        true = test_data.standard_transformer(true)
                        true = true.reshape(self.args.batch_size, self.args.pred_len, self.args.c_out)
                    else:
                        pred = loss_process(pred, true, criterion, flag=2)
                        pred = pred.detach().cpu().numpy()
                        true = true.detach().cpu().numpy()
                    t_mae, t_mse = metric(pred, true)
                    mse_list.append(t_mse)
                    mae_list.append(t_mae)
                mse = np.average(mse_list)
                mae = np.average(mae_list)
            print('|{}_{}|pred_len{}|mse:{}, mae:{}'.
                  format(self.args.data, self.args.features, self.args.pred_len, mse, mae) + '\n')

        if write_loss:
            path = './result.log'
            with open(path, "a") as f:
                f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
                f.write('|{}_{}|pred_len{}|mse:{}, mae:{}'.
                        format(self.args.data, self.args.features, self.args.pred_len, mse,
                               mae) + '\n')
                f.flush()
                f.close()
        else:
            pass

        if not save_loss:
            dir_path = os.path.join(self.args.checkpoints, setting)
            check_path = dir_path + '/' + 'checkpoint.pth'
            if os.path.exists(check_path):
                os.remove(check_path)
                os.removedirs(dir_path)

        return mse, mae

    def _process_one_batch(self, batch_x, embed):
        batch_x = batch_x.float().to(self.device)
        embed = int(embed[0])
        input_seq = batch_x[:, :self.args.label_len, :]
        outputs = self.model(input_seq)
        if embed:
            batch_y = batch_x[:, -self.args.pred_len:, :-embed].to(self.device)
        else:
            batch_y = batch_x[:, -self.args.pred_len:, :].to(self.device)

        return outputs, batch_y

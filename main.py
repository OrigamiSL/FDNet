import argparse
import torch
import numpy as np

from exp.exp_model import Exp_Model

parser = argparse.ArgumentParser(description='[FDNet] Focal Decomposed Network')

parser.add_argument('--model', type=str, required=True, default='FDNet',
                    help='model of experiment, options: [FDNet]')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S]; M:multivariate predict multivariate, '
                         'S: univariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or M task')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--label_len', type=int, default=24, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

parser.add_argument('--enc_in', type=int, default=7, help='input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--timebed', type=str, default='None', choices=['None', 'hour', 'day', 'year', 'year_min']
                    , help='time embedding type')
parser.add_argument('--d_model', type=int, default=28, help='dimension of model')
parser.add_argument('--pyramid', type=int, default=1, help='The number of input sub-sequences with different '
                                                           'sequence lengths divided by focal input sequence '
                                                           'decomposition method')
parser.add_argument('--ICOM',  action='store_true',
                    help='whether make focal ICOM'
                    , default=False)
parser.add_argument('--kernel', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--criterion', type=str, default='Standard', choices=['Standard', 'Maxabs'],
                    help='options:[Standard, Maxabs]')
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--attn_nums', type=int, default=3, help='The number of decomposed feature extraction layers')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--test_inverse', action='store_true', help='only inverse test data', default=False)
parser.add_argument('--save_loss', action='store_false', help='whether saving results and checkpoints', default=True)
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--train', action='store_true',
                    help='whether to train'
                    , default=False)
parser.add_argument('--reproducible', action='store_true',
                    help='whether to make results reproducible'
                    , default=False)

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() else False

if args.reproducible:
    np.random.seed(4321)  # reproducible
    torch.manual_seed(4321)
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.deterministic = False

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'M': [7, 7], 'S': [1, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'M': [7, 7], 'S': [1, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'M': [7, 7], 'S': [1, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'M': [7, 7], 'S': [1, 1]},
    'ECL': {'data': 'ECL.csv', 'M': [321, 321], 'S': [1, 1]},
    'Exchange': {'data': 'Exchange.csv', 'M': [8, 8], 'S': [1, 1]},
    'Traffic': {'data': 'Traffic.csv', 'M': [862, 862], 'S': [1, 1]},
    'weather': {'data': 'weather.csv', 'M': [21, 21], 'S': [1, 1]}
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.enc_in, args.c_out = data_info[args.features]

assert args.timebed in ['None', 'hour', 'year', 'year_min', 'day']
type_bed = {'None': 0, 'hour': 1, 'day': 1, 'year': 6, 'year_min': 7}
set_bed = type_bed[args.timebed]
args.enc_in = args.enc_in + int(set_bed)

args.target = args.target.replace('/r', '').replace('/t', '').replace('/n', '')

lr = args.learning_rate
print('Args in experiment:')
print(args)

mse_list = []
mae_list = []

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}_ll{}_pl{}_{}'.format(args.model,
                                               args.data, args.features,
                                               args.label_len, args.pred_len, ii)
    Exp = Exp_Model
    exp = Exp(args)  # set experiments
    if args.train:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        try:
            exp.train(setting)
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mse, mae = exp.test(setting, load=True, write_loss=True, save_loss=args.save_loss)
    mse_list.append(mse)
    mae_list.append(mae)

    torch.cuda.empty_cache()
    args.learning_rate = lr

mse = np.asarray(mse_list)
mae = np.asarray(mae_list)
avg_mse = np.mean(mse)
std_mse = np.std(mse)
avg_mae = np.mean(mae)
std_mae = np.std(mae)
print('|Mean|mse:{}, mae:{}|Std|mse:{}, mae:{}'.format(avg_mse, avg_mae, std_mse, std_mae))
path = './result.log'
with open(path, "a") as f:
    f.write('|{}_{}|pred_len{}: '.
            format(args.data, args.features, args.pred_len) + '\n')
    f.write('|Mean|mse:{}, mae:{}|Std|mse:{}, mae:{}'.
            format(avg_mse, avg_mae, std_mse, std_mae) + '\n')
    f.flush()
    f.close()
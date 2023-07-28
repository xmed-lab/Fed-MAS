'''
Main file to set up the FL system and train
Code design inspired by https://github.com/FedML-AI/FedML and https://github.com/mmendiet/FedAlign
'''
import argparse
import math
import os
import time
import sys
import logging

from methods.utils import create_criterion_dict


def add_args(parser):
    # Training settings
    parser.add_argument('--method', type=str, default='fedavg', metavar='N',
                        help='Options are: fedavg, fedprox, moon, fedmas')

    parser.add_argument('--validate_client_every', default=1, type=int, metavar='N',
                        help='number of iterations for generation')

    parser.add_argument('--cosine_restart_n', default=20, type=int, metavar='N',
                        help='number of iterations for generation')

    parser.add_argument('--aggregate_method', type=str, default='avg', metavar='N',
                        help='Options are: avg, free_higher_more, free_lower_more')


    parser.add_argument('--loss_fn_name', type=str, default='CE', metavar='N',
                        help='Options are: CE, BSM, focal, LDAM')

    parser.add_argument('--model', type=str, default='resnet18', metavar='N',
                        help='Options are: resnet18, resnet56')

    parser.add_argument('--data_dir', type=str, default='data/cifar100',
                        help='data directory: data/cifar100, data/cifar10, or another dataset')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local clients')

    parser.add_argument('--pre_trained_models', type=str, default='moco', metavar='N',
                        help='what models to load for each client')

    parser.add_argument('--partition_alpha', type=float, default=0.1, metavar='PA',
                        help='alpha value for Dirichlet distribution partitioning of data(default: 0.5)')

    parser.add_argument('--free_u', default=0, type=float, help='coeff of free model')
    parser.add_argument('--distill_u', default=0, type=float, help='coeff for global loss'
                        )

    parser.add_argument('--blur_sigma', default=0, type=float, help='coeff of blurring')
    parser.add_argument('--blur', action='store_true', help='use adam opt')

    parser.add_argument('--balanced_free', action='store_true', help='use balanced l2 loss ')

    parser.add_argument('--adam', action='store_true', help='use adam opt')
    parser.add_argument('--adamw', action='store_true', help='use adam w')


    parser.add_argument('--ramp_fed', action='store_true', help='auto scaler to ramp down kl loss')

    parser.add_argument('--save_free_losses', action='store_true', help='auto scaler to ramp down kl loss')


    parser.add_argument('--pre_trained_tv', action='store_true', default=False, help='load pretrained model')


    parser.add_argument('save_free_losses', action='store_true')


    parser.add_argument('--update_projector_head', type=str, default='both', metavar='N',
                        help='update projector head, both, free, global')

    parser.add_argument('--update_encoder', action='store_true',
                        help='update encoder')

    parser.add_argument('--adjust_learning_rate', action='store_true',
                        help='adjust the learning rate')

    parser.add_argument('--fusion', action='store_true',
                        help='fuse layers')

    parser.add_argument('--LRD', action='store_true', help='use Linear ramp down')

    parser.add_argument('--num-warmup-rounds',
                        '--num-warm-up-rounds',
                        dest="num_warmup_rounds",
                        default=100,
                        type=int,
                        help='number of warm-up epochs for unsupervised loss ramp-up during training'
                             'set to 0 to disable ramp-up')

    parser.add_argument('--start_free_rd', default=30,
                        type=int,
                        help='time to start ramp down the kl loss')

    parser.add_argument('--seed', default=0,
                        type=int,
                        help='seed')

    parser.add_argument('--fold', default=1,
                        type=int,
                        help='seed')

    parser.add_argument('--update_clients', default=1,
                        type=int,
                        help='update clients with global model')

    parser.add_argument('--add_free_projector', default=1,
                        type=int,
                        help='add free projector on top')

    parser.add_argument('--client_number', type=int, default=5, metavar='NN',
                        help='number of clients in the FL system')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
                        help='learning rate (default: 3e-4)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.0001)

    parser.add_argument('--multi_step', action='store_true', default=False,
                        help='test pretrained model')

    parser.add_argument('--adam_w', action='store_true', default=False,
                        help='test pretrained model')

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally per round')

    parser.add_argument('--aug', default=True, action='store_false',
                        help='use baseline augmentation')

    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='test pretrained model')


    parser.add_argument('--instance_sampling', action='store_true', default=False,
                        help='whether to use instance sampling or not')

    parser.add_argument('--mu', type=float, default=0.45, metavar='MU',
                        help='mu value for various methods')


    parser.add_argument('--save_client', action='store_true', default=False,
                        help='Save client checkpoints each round')

    parser.add_argument('--validate_client', action='store_true', default=False,
                        help='validate client checkpoints each round')

    parser.add_argument('--thread_number', type=int, default=5, metavar='NN',
                        help='number of parallel training threads')

    parser.add_argument('--client_sample', type=float, default=1.0, metavar='MT',
                        help='Fraction of clients to sample')

    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

    # options for CReff paper
    parser.add_argument('--comm_round', type=int, default=100,
                        help='how many rounds of communications are conducted')


    args = parser.parse_args()

    return args


# parse arguments
parser = argparse.ArgumentParser()
args = add_args(parser)

#set visible devices before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu



if args.adamw:
    optimizer_name = f'adamw_lr{args.lr}_wd{args.wd}'
elif args.adam:
    optimizer_name = f'adam_lr{args.lr}_wd{args.wd}'
elif args.adjust_learning_rate:
    optimizer_name = f'SGDWP_lr{args.lr}_wd{args.wd}'
else:
    optimizer_name = f'SGD_lr{args.lr}_wd{args.wd}'

unq_method_out_path = f'{args.data_dir}/{args.method}' \
                      f'mu_{args.mu}_' \
                      f'opt_{optimizer_name}' \
                      f'_{args.partition_method}' \
                      f'_{args.model}_' \
                      f'c{args.client_number}_' \
                      f'f{args.fold}_' \
                      f'r{args.comm_round}_' \
                      f'e{args.epochs}_' \
                      f'p_{args.pre_trained_models}_' \
                      f'is_pre_{args.pre_trained_tv}_' \
                      f'Lmdfree={args.free_u}_' \
                      f'LmdDistill={args.distill_u}_' \
                      f'{args.loss_fn_name}_' \
                      f'b{args.batch_size}_' \
                      f'p{args.update_projector_head}_a_' \
                      f'{args.aggregate_method}' \
                      f'is_{args.instance_sampling}_'

test_out_path = f'logs/test_csv/{unq_method_out_path}'
args.tensorboard_path = f'logs_tb/{unq_method_out_path}'

os.makedirs(test_out_path, exist_ok=True)
os.makedirs(args.tensorboard_path, exist_ok=True)

if os.path.exists(os.path.join(test_out_path, 'server.csv')):
    print('experiment exist')
    sys.exit()


logging.basicConfig(filename=args.tensorboard_path + '/log.txt', level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)

import pickle
import torch
import numpy as np
import random
import data_preprocessing.data_loader as dl
from models.resnet import resnet56, resnet18, resnet50
from models.resnet8 import resnet8
from models.resnet_free import resnet18 as resnet18_free
from models.resnet_free import resnet56 as resnet56_free
from models.resnet_free import resnet50 as resnet50_free
from models.efficientnet import efficientnet_b0_normal as effnetb0
from models.efficientnet import efficientnet_b0_free as effnetb0_free
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import set_start_method, Queue
from collections import defaultdict

from methods import free
import methods.fedavg as fedavg
import methods.fedprox as fedprox
import methods.moon as moon
import data_preprocessing.custom_multiprocess as cm

torch.multiprocessing.set_sharing_strategy('file_system')


# Setup Functions
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ## NOTE: If you want every run to be exactly the same each time
    ##       uncomment the following lines
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Helper Functions
def init_process(q, Client, seed):
    set_random_seed(seed)
    global client
    ci = q.get()
    client = Client(ci[0], ci[1])

#parallel run clients
def run_clients(received_info):
    try:
        return client.run(received_info)
    except KeyboardInterrupt:
        logger.info('exiting')
        return None


def allocate_clients_to_threads(args):
    mapping_dict = defaultdict(list)
    for round in range(args.comm_round):
        if args.client_sample < 1.0:
            num_clients = int(args.client_number * args.client_sample)
            client_list = random.sample(range(args.client_number), num_clients)
        else:
            num_clients = args.client_number
            client_list = list(range(num_clients))
        if num_clients % args.thread_number == 0 and num_clients > 0:
            clients_per_thread = int(num_clients / args.thread_number)
            for c, t in enumerate(range(0, num_clients, clients_per_thread)):
                idxs = [client_list[x] for x in range(t, t + clients_per_thread)]
                mapping_dict[c].append(idxs)
        else:
            raise ValueError("Sampled client number not divisible by number of threads")
    return mapping_dict


if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    set_random_seed(args.seed)
    parition_path = os.path.join(os.getcwd(), 'data/partition_strategy_LT',
                                 f'{args.data_dir.split("/")[-1]}_LT_{args.fold}' + '_' + args.partition_method + '_' + str(
                                     args.partition_alpha) + f'_{args.client_number}_client_{str(args.seed)}.pkl')
    if os.path.exists(parition_path):
        with open(parition_path, 'rb') as f:  # Python 3: open(..., 'rb')
            class_num, net_dataidx_map, traindata_cls_counts = pickle.load(f)
    # simulated settings
    if 'isic-FL' in args.data_dir:
        logger.info('Loading ISIC-FL setting')
        return_index = args.method == 'IF_dual'
        train_data_num, test_data_num, train_data_global, val_data_global, test_data_global, data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict, \
        class_num = dl.load_partition_isic_fl(args.data_dir, args.client_number, args.batch_size,
                                              instance_sampling=args.instance_sampling, return_index=return_index,
                                              fold=args.fold, aug=args.aug)
        class_num = 8

    elif 'Flamby_isic' in args.data_dir:
        logger.info('Loading ISIC-FL setting')
        return_index = args.method == 'IF_dual'
        train_data_num, test_data_num, train_data_global, val_data_global, test_data_global, data_local_num_dict, train_data_local_dict, val_data_local_dict, test_data_local_dict, \
        class_num = dl.load_partition_flampy_isic(args.data_dir, args.client_number, args.batch_size,
                                                  instance_sampling=args.instance_sampling, return_index=return_index,
                                                  fold=args.fold, aug=args.aug)
        class_num = 8

    else:
        raise Exception('Dataset is not supported jere'

                        )
    mapping_dict = allocate_clients_to_threads(args)
    # init method and model type

    if args.method == 'fedavg':
        Server = fedavg.Server
        Client = fedavg.Client
        if 'resnet56' in args.model:
            Model = resnet56
        elif 'resnet50' in args.model:
            Model = resnet50
        elif 'resnet8' in args.model:
            Model = resnet8
        elif 'resnet18' in args.model:
            Model = resnet18
        elif 'effnetb0' in args.model:
            Model = effnetb0
        else:
            raise Exception("Not implemented")

        criterion_dict = create_criterion_dict(train_data_local_dict, args.comm_round, args.loss_fn_name,
                                               num_classes=class_num)

        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                       'val_data': val_data_global,
                       'num_classes': class_num,
                       'test_out_path': test_out_path}
        client_dict = [
            {'train_data': train_data_local_dict,
             'val_data': val_data_local_dict, 'test_data': test_data_local_dict,
             'device': i % torch.cuda.device_count(),
             'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num, 'criterion': criterion_dict,
             'test_out_path': test_out_path}
            for i in
            range(args.thread_number)]

    elif args.method == 'fedprox':
        Server = fedprox.Server
        Client = fedprox.Client
        if 'resnet56' in args.model:
            Model = resnet56
        else:
            Model = resnet18 if 'resnet18' in args.model else resnet8
        server_dict = {'train_data': train_data_global,
                       'val_data': val_data_global,
                       'test_data': test_data_global, 'model_type': Model,
                       'num_classes': class_num, 'test_out_path': test_out_path}

        criterion_dict = create_criterion_dict(train_data_local_dict, args.comm_round, args.loss_fn_name,
                                               num_classes=class_num)

        client_dict = [
            {'train_data': train_data_local_dict, 'val_data': val_data_local_dict, 'test_data': test_data_local_dict,
             'device': i % torch.cuda.device_count(),
             'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num,
             'criterion': criterion_dict,
             'test_out_path': test_out_path
             } for i in
            range(args.thread_number)]

    elif args.method == 'fedmas':
        Server = free.Server
        Client = free.Client
        if 'resnet56' in args.model:
            Model = resnet56_free
        elif 'resnet50' in args.model:
            Model = resnet50_free
        elif 'resnet8' in args.model:
            Model = resnet8
        elif 'resnet18' in args.model:
            Model = resnet18_free
        elif 'effnetb0' in args.model:
            Model = effnetb0_free
        else:
            raise Exception("Not implemented")

        criterion_dict, criterion_bal_l2 = create_criterion_dict(train_data_local_dict, args.comm_round,
                                                               args.loss_fn_name,
                                                               num_classes=class_num, byol=True)

        free_criterion_dict = create_criterion_dict(train_data_local_dict, args.comm_round,
                                                    args.loss_fn_name,
                                                    num_classes=class_num)

        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'model_type': Model,
                       'val_data': val_data_global,
                       'num_classes': class_num, 'test_out_path': test_out_path}

        #support different local pre-trained models
        if args.pre_trained_models == 'mix':
            pre_trained_unique_models = ['IMAGENET1K_V1', 'moco']
        elif args.pre_trained_models == 'moco':
            pre_trained_unique_models = ['moco']
        elif args.pre_trained_models == 'clip':
            pre_trained_unique_models = ['clip']
        elif args.pre_trained_models == 'dinov2':
            pre_trained_unique_models = ['dinov2']
        elif args.pre_trained_models == 'IMAGENET1K_V1':
            pre_trained_unique_models = ['IMAGENET1K_V1']
        elif args.pre_trained_models == '18_img':
            pre_trained_unique_models = ['18_img']
        elif args.pre_trained_models == 'eb0':
            pre_trained_unique_models = ['eb0']
        else:
            logger.info('pre_trained _models not defined')
            raise Exception

        pre_trained_models = np.random.permutation(
            np.repeat(pre_trained_unique_models, math.ceil(args.client_number / len(pre_trained_unique_models))))[
                             :args.client_number]

        logger.info('Clients pre_trained_models is ')
        logger.info(np.unique(pre_trained_models, return_counts=True))
        client_dict = [
            {'train_data': train_data_local_dict, 'val_data': val_data_local_dict, 'test_data': test_data_local_dict,
             'device': i % torch.cuda.device_count(),
             'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num,
             'pre_trained': pre_trained_models[i],
             'criterion': criterion_dict,
             'criterion_bal_l2': criterion_bal_l2,
             'free_criterion': free_criterion_dict, 'test_out_path': test_out_path} for i
            in
            range(args.thread_number)]
    elif args.method == 'moon':
        Server = moon.Server
        Client = moon.Client
        if 'resnet56' in args.model:
            Model = resnet56
        else:
            Model = resnet18 if 'resnet18' in args.model else resnet8
        server_dict = {'train_data': train_data_global, 'test_data': test_data_global, 'val_data': val_data_global,
                       'model_type': Model,
                       'num_classes': class_num, 'test_out_path': test_out_path}
        criterion_dict = create_criterion_dict(train_data_local_dict, args.comm_round, args.loss_fn_name,
                                               num_classes=class_num)

        client_dict = [
            {'train_data': train_data_local_dict, 'val_data': val_data_local_dict, 'test_data': test_data_local_dict,
             'device': i % torch.cuda.device_count(),
             'criterion': criterion_dict,
             'client_map': mapping_dict[i], 'model_type': Model, 'num_classes': class_num,
             'test_out_path': test_out_path} for i in
            range(args.thread_number)]
    else:
        raise ValueError('Invalid --method chosen! Please choose from availible methods.')

    # init nodes
    client_info = Queue()
    for i in range(args.thread_number):
        client_info.put((client_dict[i], args))

    # Start server and get initial outputs
    pool = cm.MyPool(processes=args.thread_number, initializer=init_process, initargs=(client_info, Client, args.seed))

    # init server
    server_dict['save_path'] = args.tensorboard_path
    writer = SummaryWriter(os.path.join(server_dict['save_path'], 'log'))

    server = Server(server_dict, args)

    server_outputs = server.start()
    # Start Federated Training
    time.sleep(150 * (args.client_number / 16))  # Allow time for threads to start up

    best_server_acc = 0

    best_client_mean = 0


    class_losses = torch.zeros((args.comm_round + 1, args.client_number, class_num), dtype=torch.float)
    class_weights = torch.zeros((args.comm_round + 1, args.client_number, class_num), dtype=torch.float)

    for r in range(args.comm_round):
        logger.info('************** Round: {} ***************'.format(r))
        round_start = time.time()
        client_outputs = pool.map(run_clients, server_outputs)
        client_outputs = [c for sublist in client_outputs for c in sublist]
        server_outputs = server.run(client_outputs)
        round_end = time.time()

        logger.info('Round {} Time: {}s'.format(r, round_end - round_start))

        server_acc = server_outputs[0]['server_acc']
        server_fusion_acc = server_outputs[0].get('server_fusion_acc', 0)

        writer.add_scalar('Server %d Acc.', server_acc, global_step=r)

        mean_acc = 0

        for client_o in client_outputs:
            idx = client_o['client_index']
            writer.add_scalar('Local Client %d Acc.' % idx, client_o['acc'], global_step=r)
            mean_acc += client_o['acc']
            if args.method == 'fedmas':
                class_losses[r, idx] = client_o['class_losses']
                class_weights[r, idx] = client_o['class_weights']

                writer.add_scalar('loss on client %d' % idx, client_o['loss'], global_step=r)
                writer.add_scalar('loss sup on client %d' % idx, client_o['loss_sup'], global_step=r)
                writer.add_scalar('loss glob on client %d' % idx, client_o['loss_glob'], global_step=r)
                writer.add_scalar('loss free client %d' % idx, client_o['loss_free'], global_step=r)

        mean_acc /= len(client_outputs)

        if server_acc > best_server_acc:
            best_server_acc = server_acc

        if mean_acc > best_client_mean:
            best_client_mean = mean_acc


        logger.info(f'Round {r},Server Acc:{round(server_acc, 4)}, '
                    )

        logger.info(f'Round {r}, Best Server Acc:{round(best_server_acc, 4)}, '
                    )

        logger.info(f'Round {r}, Client Acc: {round(mean_acc, 4)}, '
                    )

        logger.info(f'Round {r}, Best Client Acc: {round(best_client_mean, 4)}, '
                    )

    if args.save_free_losses:
        torch.save(class_losses,
                   f'{args.data_dir}_class_loss_{args.free_u}_{args.partition_alpha}_{args.aggregate_method}_{args.fold}.pt')
        torch.save(class_weights,
                   f'{args.data_dir}_class_weights_{args.free_u}_{args.partition_alpha}_{args.aggregate_method}_{args.fold}.pt')

    pool.close()
    pool.join()

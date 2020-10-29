from __future__ import print_function
import argparse
import os,sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from utils.common import load_loss_scheme
from dataloader import KITTILoader as DA

from networks.ESNet import ESNet
from networks.ESNet_M import ESNet_M
from networks.FADNet import FADNet
from networks.stackhourglass import PSMNet
from networks.DispNetC import DispNetC
from losses.multiscaleloss import multiscaleloss

from tensorboardX import SummaryWriter
parser = argparse.ArgumentParser(description='ESNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='esnet',
                    help='select model')
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/training/',
                    help='datapath')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--devices', type=str, help='indicates CUDA devices, e.g. 0,1,2', default='0')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--loss', type=str, help='indicates the loss scheme', default='simplenet_flying')
#parser.add_argument('--trainloss', type=str, help='indicates the train loss scheme, only used for pwcnet', default = None, choices=['L1_pwc','smoothL1_pwc','robust_EPE'])

args = parser.parse_args()

if not os.path.exists(args.savemodel):
    os.makedirs(args.savemodel)
writer = SummaryWriter(args.savemodel)

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.datatype == '2015':
    from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
    from dataloader import KITTIloader2012 as ls

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img,all_right_img,all_left_disp, True), 
         batch_size= 8, shuffle= True, num_workers= 8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
         batch_size= 8, shuffle= False, num_workers= 4, drop_last=False)

devices = [int(item) for item in args.devices.split(',')]
ngpus = len(devices)

if args.model == 'esnet':
    model = ESNet(batchNorm=False, lastRelu=True, maxdisp=-1)
elif args.model == 'esnet_m':
    model = ESNet_M(batchNorm=False, lastRelu=True, maxdisp=-1)
elif args.model == 'fadnet':
    model = FADNet(False, True)
elif args.model == 'psmnet':
    model = PSMNet(maxdisp=args.maxdisp)
elif args.model == 'dispnetc':
    model = DispNetC(batchNorm=False, lastRelu=True, maxdisp=-1)
else:
    print('no model')
    sys.exit(-1)

if args.cuda:
    model = nn.DataParallel(model, device_ids=devices)
    model.cuda()

# if args.loadmodel is not None:
#     state_dict = torch.load(args.loadmodel)
#     model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

init_lr = 1e-5 #Zhengyu: Checked with Qiang Wang to use 1e-5.
optimizer = optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999))

loss_json = load_loss_scheme(args.loss)
train_round = loss_json["round"]
loss_scale = loss_json["loss_scale"]
loss_weights = loss_json["loss_weights"]
epoches = loss_json["epoches"]

def train(imgL,imgR,disp_L, criterion):
    model.train()
    imgL   = Variable(torch.FloatTensor(imgL))
    imgR   = Variable(torch.FloatTensor(imgR))   
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    #---------
    mask = (disp_true > 0)
    mask.detach_()
    #----

    optimizer.zero_grad()
    
    if args.model == 'psmnet':
        output1, output2, output3 = model(torch.cat((imgL, imgR), 1))
        output1 = torch.squeeze(output1,1)
        output2 = torch.squeeze(output2,1)
        output3 = torch.squeeze(output3,1)
        loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
    elif args.model == 'fadnet':
        output_net1, output_net2 = model(torch.cat((imgL, imgR), 1))

        # multi-scale loss
        disp_true = disp_true.unsqueeze(1)
        loss_net1 = criterion(output_net1, disp_true)
        loss_net2 = criterion(output_net2, disp_true)
        loss = loss_net1 + loss_net2 
        
    elif args.model in ['esnet', 'esnet_m', 'dispnetc', 'dispnets']:        
        output_net = model(torch.cat((imgL, imgR), 1))
        disp_true = disp_true.unsqueeze(1)
        loss = criterion(output_net, disp_true)

    else:
        raise NotImplementedError
    
    loss.backward()
    optimizer.step()

    return loss.data.item()

def test(imgL,imgR,disp_true):
    model.eval()
    imgL   = Variable(torch.FloatTensor(imgL))
    imgR   = Variable(torch.FloatTensor(imgR))   
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    #print(imgL.size())
    #imgL = F.pad(imgL, (0, 48, 0, 16), "constant", 0)
    #imgR = F.pad(imgR, (0, 48, 0, 16), "constant", 0)
    #print(imgL.size())

    with torch.no_grad():
        if args.model == "psmnet":
            output_net = model(torch.cat((imgL, imgR), 1))
            pred_disp = output_net.squeeze(1)
        elif args.model == "fadnet":
            output_net1, output_net2 = model(torch.cat((imgL, imgR), 1))
            pred_disp = output_net2.squeeze(1)
        elif args.model in ['esnet', 'esnet_m', 'dispnetc', 'dispnets']:
            output_net = model(torch.cat((imgL, imgR), 1))
            pred_disp = output_net[0].squeeze(1)

    pred_disp = pred_disp.data.cpu()
    #pred_disp = pred_disp[:, :368, :1232]

    #computing 3-px error#
    true_disp = disp_true.clone()
    index = np.argwhere(true_disp>0)
    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
    torch.cuda.empty_cache()

    return 1-(float(torch.sum(correct))/float(len(index[0])))

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 600:
       lr = init_lr
    else:
       lr = init_lr / 10.0
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    min_acc=1000
    min_epo=0
    min_round=0
    counter_total_iter=0
    counter_total_epoch=0
    start_full_time = time.time()

    # test on the loaded model
    total_test_loss = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        test_loss = test(imgL,imgR, disp_L)
        print('Iter %d 3-px error in val = %.3f' %(batch_idx, test_loss*100))
        total_test_loss += test_loss
    min_acc=total_test_loss/len(TestImgLoader)*100
    writer.add_scalar('Error_rate_val', min_acc, -1)
    print('MIN epoch %d of round %d total test error = %.3f' %(min_epo, min_round, min_acc))
    
    start_round = 0
    for r in range(start_round, train_round):
        criterion = multiscaleloss(loss_scale, 1, loss_weights[r], train_loss = 'smoothL1', test_loss='L1', mask=True)
        print(loss_weights[r])

        for epoch in range(1, epoches[r]+1):
            total_train_loss = 0
            total_test_loss = 0
            adjust_learning_rate(optimizer,epoch)

            ## training ##
            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
                start_time = time.time() 

                loss = train(imgL_crop,imgR_crop, disp_crop_L, criterion)
                writer.add_scalar('Loss', loss, counter_total_iter)
                print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
                total_train_loss += loss
                counter_total_iter += 1

            print('epoch %d of round %d total training loss = %.3f' %(epoch, r, total_train_loss/len(TrainImgLoader)))
	       
            ## Test ##
            for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
                test_loss = test(imgL,imgR, disp_L)
                print('Iter %d 3-px error in val = %.3f' %(batch_idx, test_loss*100))
                total_test_loss += test_loss

            writer.add_scalar('Error_rate_val', total_test_loss/len(TestImgLoader)*100, counter_total_epoch)
            print('epoch %d of round %d total 3-px error in val = %.3f' %(epoch, r, total_test_loss/len(TestImgLoader)*100))
            if total_test_loss/len(TestImgLoader)*100 < min_acc:
                min_acc = total_test_loss/len(TestImgLoader)*100
                min_epo = epoch
                min_round = r
                savefilename = args.savemodel+'best.tar'
                torch.save({
                    'epoch': epoch,
                        'round': r,
                    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss/len(TrainImgLoader),
                    'test_loss': total_test_loss/len(TestImgLoader)*100,
                }, savefilename)
                print('MIN epoch %d of round %d total test error = %.3f' %(min_epo, min_round, min_acc))

	        #SAVE
            if (epoch - 1) % 100 == 0:
                savefilename = args.savemodel+'finetune_%s_%s' % (str(r), str(epoch))+'.tar'
                torch.save({
	                 'epoch': epoch,
                         'round': r, 
	                 'state_dict': model.state_dict(),
	                 'train_loss': total_train_loss/len(TrainImgLoader),
	                 'test_loss': total_test_loss/len(TestImgLoader)*100,
	             }, savefilename)

            counter_total_epoch += 1
	
    print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))
    print(min_epo)
    print(min_round)
    print(min_acc)


if __name__ == '__main__':
    #For resolving vscode debugger bug:AssertionError: can only join a child process
    # import multiprocessing
    # multiprocessing.set_start_method('spawn', True)
    
    main()

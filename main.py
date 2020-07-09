from __future__ import print_function
import os
import argparse
from glob import glob

from PIL import Image
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from model import DecomNet
from utils import *
import torch.nn.functional as F
import pdb
torch.cuda.set_device(3)
parser = argparse.ArgumentParser(description='')

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default="0", help='GPU idx')
parser.add_argument('--gpu_mem', dest='gpu_mem', type=float, default=0.8, help="0 to 1, gpu memory usage")
parser.add_argument('--phase', dest='phase', default='train', help='train or test')

parser.add_argument('--epoch', dest='epoch', type=int, default=10000, help='number of total epoches')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=96, help='patch size')
parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', default=100, help='evaluating and saving checkpoints every #  epoch')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='directory for checkpoints')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='directory for evaluating outputs')

parser.add_argument('--save_dir', dest='save_dir', default='./test_results', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./data/test', help='directory for testing inputs')
parser.add_argument('--decom', dest='decom', default=1, help='decom flag, 0 for enhanced results only and 1 for decomposition results')

args = parser.parse_args()


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()  
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def get_data(low_im, high_im, train_low_data_eq):
    #tran = transforms.ToTensor()
    batch_size = args.batch_size
    patch_size = args.patch_size
    
    # im = np.clip(train_low_data_eq * 255.0, 0, 255.0).astype('uint8')
    # cv2.imwrite('train_low_data_eq.png', cv2.cvtColor(im, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
    # print('low_im',train_low_data_eq * 255.0)
    # im_p = data_augmentation(low_im[12 : 12+220, 22 : 22+220, :], 2).copy()
    # im_p = np.clip(im_p * 255.0, 0, 255.0).astype('uint8')
    # cv2.imwrite('low_im_patch.png', cv2.cvtColor(im_p, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
    # print('im_p',im_p)

    batch_input_low = torch.empty(batch_size, 3, patch_size, patch_size)
    batch_input_high = torch.empty(batch_size, 3, patch_size, patch_size)
    batch_input_low_eq = torch.empty(batch_size, 1, patch_size, patch_size)
    
    for patch_id in range(batch_size):
        h, w, _ = low_im.shape
        x = np.random.randint(0, h - patch_size)
        y = np.random.randint(0, w - patch_size)

        rand_mode = np.random.randint(0, 7)

        batch_input_low[patch_id, :, :, :] = transform(data_augmentation(low_im[x : x+patch_size, y : y+patch_size, :], rand_mode).copy())
        batch_input_high[patch_id, :, :, :] = transform(data_augmentation(high_im[x : x+patch_size, y : y+patch_size, :], rand_mode).copy())
        batch_input_low_eq[patch_id, :, :, :] = transform(data_augmentation(train_low_data_eq[x : x+patch_size, y : y+patch_size, :], rand_mode).copy())
        
    return batch_input_low, batch_input_high, batch_input_low_eq
    

def rgbTOgray(tensor):
    R = tensor[:,0,:,:]
    G = tensor[:,1,:,:]
    B = tensor[:,2,:,:]
    tensor_new =0.299*R+0.587*G+0.114*B
    tensor_new = tensor_new.unsqueeze(1)
    return tensor_new

def gradient(x):
    # tf.image.image_gradients(image)
    h_x = x.size()[-2]
    w_x = x.size()[-1]
    # gradient step=1
    l = x
    r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    t = x
    b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]
    dx, dy = torch.abs(r - l), torch.abs(b - t)
    # dx will always have zeros in the last column, r-l
    # dy will always have zeros in the last row,    b-t
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy

def Func_smooth(input_I, input_R):
    input_R = rgbTOgray(input_R)
    # another_=tf.reduce_mean(self.gradient(input_I, "x")/tf.maximum(self.gradient(input_I, "x"),0.01)+self.gradient(input_I, "y")/tf.maximum(self.gradient(input_I, "y"),0.01))
    return torch.mean(gradient(input_I)[0] * torch.exp(-10 * gradient(input_R)[0]) + gradient(input_I)[1] * torch.exp(-10 * gradient(input_R)[1]))#+another_

def Func_low_loss_smooth(R_L):
    RL = rgbTOgray(R_L)
    return torch.mean(torch.abs(gradient(RL)[0]) + torch.abs(gradient(RL)[1]))

def save_img(filepath, result_1, result_2 = None):
    #print(result_1.shape)
    result_1 = result_1.squeeze(0).clamp(0, 1).numpy().transpose(1,2,0)
    #result_2 = result_2.squeeze().clamp(0, 1).numpy().transpose(1,2,0)

    #if not result_2.any():
    cat_image = result_1
   # else:
    #    cat_image = np.concatenate([result_1, result_2], axis = 1)
    cat_image = np.clip(cat_image * 255.0, 0, 255.0).astype('uint8')
    #im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    #im.save(filepath, 'png')
    cv2.imwrite(filepath, cv2.cvtColor(cat_image, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

def train(model, epoch, optimizer, low_im, high_im, train_low_data_eq):
    epoch_loss = 0
    model.train()
    optimizer.zero_grad()
    #im = np.clip(low_im * 255.0, 0, 255.0).astype('uint8')
    #cv2.imwrite('low_im_train.png', cv2.cvtColor(im, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    input_low, input_high, input_low_eq  = get_data(low_im, high_im, train_low_data_eq)
    #input_low, input_high, input_low_eq  = torch.randn(16,3,24,24), torch.randn(16,3,24,24), torch.randn(16,1,24,24)
    input_low, input_high, input_low_eq = Variable(input_low).cuda(), Variable(input_high).cuda(), Variable(input_low_eq).cuda()
    R_low, I_low = model(input_low)
    I_low_3 = torch.cat((I_low, I_low, I_low), 1)

    recon_loss_low = criterion(R_low * I_low_3, input_high)
    
    
    R_low_max, _ = torch.max(R_low, 1, keepdims=True)
    recon_loss_low_eq = criterion(R_low_max, input_low_eq)
    
    #R_low_loss_smooth = Smoothloss(R_low)
    R_low_loss_smooth = Func_low_loss_smooth(R_low)
    
    Ismooth_loss_low = Func_smooth(I_low, R_low)# smooth
    
    loss_Decom= recon_loss_low + 0.1 * Ismooth_loss_low + 0.1 * recon_loss_low_eq + 0.01 * R_low_loss_smooth
    #loss_Decom= recon_loss_low + 0.1 * recon_loss_low_eq 
    
    epoch_loss += loss_Decom.data
    loss_Decom.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print("Epoch: [%3d], loss: %.6f"%(epoch, epoch_loss))
    if epoch % 200 == 0:
        eval_R, eval_L = model(eval_in)
        #I_low = torch.cat((I_low,I_low,I_low), 1)
        filepath_I = os.path.join('Results/', str(epoch) + "_I_low.png")
        filepath_R = os.path.join('Results/', str(epoch) + "_R_low.png")
        eval_save = eval_R.cpu().data
        #print(eval_save)
        save_img(filepath_R, eval_save)
        #save_img(filepath_I, I_low.cpu().data)
        #save_images(filepath_R, R_low.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0))
        #save_images(filepath_I, I_low.cpu().data.squeeze().clamp(0, 1).numpy().transpose(1,2,0))

def main(model, optimizer):
    #pdb.set_trace()
    low_im = load_images('/home/zhangdy/Retinex/eval15/low/22.png')
    
    high_im = load_images('/home/zhangdy/Retinex/eval15/low/22.png')
    
    train_low_data_max_chan = np.max(high_im,axis=2,keepdims=True)
    weight_eq_clahe = 0#sigmoid(5*(meanFilter(train_low_data_max_chan,(20,20))-0.5))
    train_low_data_eq = (1-weight_eq_clahe) * histeq(train_low_data_max_chan) + weight_eq_clahe * adapthisteq(train_low_data_max_chan)

    for epoch in range(0, args.epoch):
        train(model, epoch, optimizer, low_im, high_im, train_low_data_eq)
        if (epoch+1) == 100:
            for param_group in optimizer.param_groups:
                param_group['lr'] /=  10.0
    print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

low_im = load_images('/home/zhangdy/Retinex/eval15/low/22.png')
high_im = load_images('/home/zhangdy/Retinex/eval15/high/22.png')
transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)

eval_in = transform(low_im).unsqueeze(0)
eval_high = transform(high_im).unsqueeze(0)
print(eval_high.cpu().data)
eval_in = eval_in.cuda()
DeNet = DecomNet()
DeNet = DeNet.cuda()
criterion = nn.L1Loss()
criterion = criterion.cuda()
Smoothloss = TVLoss()
Smoothloss = Smoothloss.cuda()
optimizer = optim.Adam([paras for paras in DeNet.parameters() if paras.requires_grad == True], lr=args.start_lr)
main(DeNet, optimizer)

        
        

    

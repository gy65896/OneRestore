import os, time, torch, argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
import numpy as np
from torchvision import transforms
from makedataset import Dataset
from utils.utils import print_args, load_restore_ckpt_with_optim, load_embedder_ckpt, adjust_learning_rate, data_process, tensor_metric, load_excel, save_checkpoint
from model.loss import Total_loss
from model.Embedder import Embedder
from model.OneRestore import OneRestore
from torch.utils.data.distributed import DistributedSampler
from PIL import Image

torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


transform_resize = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
        ]) 

def main(args):
    

    print('> Model Initialization...')
    embedder = load_embedder_ckpt(device, freeze_model=True, ckpt_name=args.embedder_model_path)
    restorer, optimizer, cur_epoch = load_restore_ckpt_with_optim(device, local_rank=local_rank, freeze_model=False, ckpt_name=args.restore_model_path, lr=args.lr)
    loss = Total_loss(args)
    
    print('> Loading dataset...')
    data = Dataset(args.train_input)
    dataset = DataLoader(dataset=data, batch_size=args.bs,
                                    shuffle=False, 
                                    num_workers=args.num_works,
                                    pin_memory=True,drop_last=False,                                     
                                    sampler=DistributedSampler(data,shuffle=True))
    
    print('> Start training...')
    start_all = time.time()
    train(restorer, embedder, optimizer, loss, cur_epoch, args, dataset, device)
    end_all = time.time()
    print('Whloe Training Time:' +str(end_all-start_all)+'s.')

def train(restorer, embedder, optimizer, loss, cur_epoch, args, dataset, device):

    metric = []
    for epoch in range(cur_epoch, args.epoch):
        optimizer = adjust_learning_rate(optimizer, epoch, args.adjust_lr)
        learnrate = optimizer.param_groups[-1]['lr']
        restorer.train()

        for i, data in enumerate(dataset,0):
            pos, inp, neg = data_process(data, args, device)

            text_embedding,_,_ = embedder(inp[1],'text_encoder')
            out = restorer(inp[0], text_embedding)

            restorer.zero_grad()
            total_loss = loss(inp, pos, neg, out)
            total_loss.backward()
            optimizer.step()

            mse = tensor_metric(pos,out, 'MSE', data_range=1)
            psnr = tensor_metric(pos,out, 'PSNR', data_range=1)
            ssim = tensor_metric(pos,out, 'SSIM', data_range=1)

            print("[epoch %d][%d/%d] lr :%f Floss: %.4f MSE: %.4f PSNR: %.4f SSIM: %.4f"%(epoch+1, i+1, \
                len(dataset), learnrate, total_loss.item(), mse, psnr, ssim))
            

        psnr_t1, ssim_t1, psnr_t2, ssim_t2 = test(args, restorer, embedder, device, epoch)
        metric.append([psnr_t1, ssim_t1, psnr_t2, ssim_t2])
        print("[epoch %d] Test images PSNR1: %.4f SSIM1: %.4f"%(epoch+1, psnr_t1,ssim_t1))

        load_excel(metric)
        save_checkpoint({'epoch': epoch + 1,'state_dict': restorer.state_dict(),'optimizer' : optimizer.state_dict()},\
                        args.save_model_path, epoch+1, psnr_t1,ssim_t1,psnr_t2,ssim_t2)

def test(args, restorer, embedder, device, epoch=-1):
    combine_type = args.degr_type
    psnr_1, psnr_2, ssim_1, ssim_2 = 0, 0, 0, 0
    os.makedirs(args.output,exist_ok=True)

    for i in range(len(combine_type)-1):
        file_list =  os.listdir(f'{args.test_input}/{combine_type[i+1]}/')
        for j in range(len(file_list)):
            hq = Image.open(f'{args.test_input}/{combine_type[0]}/{file_list[j]}')
            lq = Image.open(f'{args.test_input}/{combine_type[i+1]}/{file_list[j]}')
            restorer.eval()
            with torch.no_grad():
                lq_re = torch.Tensor((np.array(lq)/255).transpose(2, 0, 1)).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
                lq_em = transform_resize(lq).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
                hq = torch.Tensor((np.array(hq)/255).transpose(2, 0, 1)).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

                starttime = time.time()

                text_embedding_1,_,text_1 = embedder([combine_type[i+1]],'text_encoder')
                text_embedding_2,_, text_2 = embedder(lq_em,'image_encoder')
                out_1 = restorer(lq_re, text_embedding_1)
                if text_1 != text_2:
                    print(text_1, text_2)
                    out_2 = restorer(lq_re, text_embedding_2)
                else:
                    out_2 = out_1
                
                endtime1 = time.time()

                imwrite(torch.cat((lq_re, out_1, out_2, hq), dim=3), args.output \
                    + file_list[j][:-4] + '_' + str(epoch) + '_' + combine_type[i+1] + '.png', range=(0, 1))
                # due to the vision problem, you can replace above line by
                # imwrite(torch.cat((lq_re, out_1, out_2, hq), dim=3), args.output \
                #     + file_list[j][:-4] + '_' + str(epoch) + '_' + combine_type[i+1] + '.png')
            psnr_1 += tensor_metric(hq, out_1, 'PSNR', data_range=1)
            ssim_1 += tensor_metric(hq, out_1, 'SSIM', data_range=1)
            psnr_2 += tensor_metric(hq, out_2, 'PSNR', data_range=1)
            ssim_2 += tensor_metric(hq, out_2, 'SSIM', data_range=1)
            print('The ' + file_list[j][:-4] + ' Time:' + str(endtime1 - starttime) + 's.')

    return psnr_1 / (len(file_list)*len(combine_type)), ssim_1 / (len(file_list)*len(combine_type)),\
        psnr_2 / (len(file_list)*len(combine_type)), ssim_2 / (len(file_list)*len(combine_type))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "OneRestore Training")

    # load model
    parser.add_argument("--embedder-model-path", type=str, default = "./ckpts/embedder_model.tar", help = 'embedder model path')
    parser.add_argument("--restore-model-path", type=str, default = None, help = 'restore model path')
    parser.add_argument("--save-model-path", type=str, default = "./ckpts/", help = 'restore model path')

    parser.add_argument("--epoch", type=int, default = 300, help = 'epoch number')
    parser.add_argument("--bs", type=int, default = 4, help = 'batchsize')
    parser.add_argument("--lr", type=float, default = 1e-4, help = 'learning rate')
    parser.add_argument("--adjust-lr", type=int, default = 30, help = 'adjust learning rate')
    parser.add_argument("--num-works", type=int, default = 4, help = 'number works')
    parser.add_argument("--loss-weight", type=tuple, default = (0.6,0.3,0.1), help = 'loss weights')
    parser.add_argument("--degr-type", type=list, default = ['clear', 'low', 'haze', 'rain', 'snow',\
        'low_haze', 'low_rain', 'low_snow', 'haze_rain', 'haze_snow', 'low_haze_rain', 'low_haze_snow'], help = 'degradation type')
    
    parser.add_argument("--train-input", type=str, default = "./dataset.h5", help = 'train data')
    parser.add_argument("--test-input", type=str, default = "./data/CDD-11_test", help = 'test path')
    parser.add_argument("--output", type=str, default = "./result/", help = 'output path')

    argspar = parser.parse_args()

    print_args(argspar)
    main(argspar)

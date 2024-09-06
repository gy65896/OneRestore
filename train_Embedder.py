import argparse, os, torch, time
import torch.optim

from utils.utils import load_embedder_ckpt_with_optim, adjust_learning_rate, freeze_text_embedder, AverageMeter
from utils.utils_data import init_embedding_data



def train_embedding(cur_epoch, model, optimizer, trainloader, testloader, device, cfg_em):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    
    acc_train_meter = AverageMeter()
    acc_test_meter = AverageMeter()
    loss_train_meter = AverageMeter()
    loss_test_meter = AverageMeter()
    time_train_meter = AverageMeter()
    time_test_meter = AverageMeter()

    freeze_text_embedder(model)
    for k,v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))
    for epoch in range(cur_epoch, cfg_em.epoch+1):

        optimizer = adjust_learning_rate(optimizer, epoch-1, cfg_em.lr_decay)
        lr = optimizer.param_groups[-1]['lr']

        model.train()
        for idx, batch in enumerate(trainloader):
            for i in range(len(batch)):
                batch[i] = batch[i].to("cuda" if torch.cuda.is_available() else "cpu")
            time_start = time.time()
            out = model(batch, 'train')
            loss = out['loss_total']
            acc  = out['acc_type']
            time_train_meter.update(time.time() - time_start)

            acc_train_meter.update(acc)
            loss_train_meter.update(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch:{epoch}|Iter:{idx+1}/{len(trainloader)}|lr:{lr},'
                f'Loss: {loss_train_meter.avg:.3f},' 
                f'Acc: {acc_train_meter.avg:.3f},'
                f'Time: {time_train_meter.avg:.3f},', flush=True)
        
        model.eval()
        for idx, batch in enumerate(testloader):
            for i in range(len(batch)):
                batch[i] = batch[i].to("cuda" if torch.cuda.is_available() else "cpu")

            time_start = time.time()
            out = model(batch, 'train')
            loss = out['loss_total']
            acc  = out['acc_type']
            time_test_meter.update(time.time() - time_start)

            acc_test_meter.update(acc)
            loss_test_meter.update(loss)
            print(f'Epoch:{epoch}|Iter:{idx+1}/{len(testloader)}|lr:{lr},'
                f'Loss: {loss_test_meter.avg:.3f},'
                f'Acc: {acc_test_meter.avg:.3f},'
                f'Time: {time_test_meter.avg:.3f},', flush=True)
        
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()},
                         f'{cfg_em.check_dir}/embedder_model_epoch{epoch}_{acc_train_meter.avg:.3f}_{loss_train_meter.avg:.3f}_{acc_test_meter.avg:.3f}_{loss_test_meter.avg:.3f}.tar')
        acc_train_meter.reset()
        acc_test_meter.reset()
        loss_train_meter.reset()
        loss_test_meter.reset()
        time_train_meter.reset()
        time_test_meter.reset()
    print('Done!')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
     # load model
    parser.add_argument("--seed", type=int, default = 124)
    parser.add_argument("--pre_weight", type=str, default = '')
    parser.add_argument("--lr", type=float, default = 0.0001)
    parser.add_argument("--type_name", type=list, default = ['clear', 'low', 'haze', 'rain',\
                                        'snow', 'low_haze', 'low_rain', 'low_snow', 'haze_rain',\
                                        'haze_snow', 'low_haze_rain', 'low_haze_snow'])
    parser.add_argument("--train-dir", type=str, default = './data/CDD-11_train/')
    parser.add_argument("--test-dir", type=str, default = './data/CDD-11_test/')
    parser.add_argument("--batch", type=int, default = 128)
    parser.add_argument("--num-workers", type=int, default = 0)
    parser.add_argument("--epoch", type=int, default = 200)
    parser.add_argument("--lr-decay", type=int, default = 50)
    parser.add_argument("--check-dir", type=str, default = "./ckpts")
    
    args = parser.parse_args()

    os.makedirs(args.check_dir,exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    embedder, optimizer, cur_epoch, device = load_embedder_ckpt_with_optim(device, args)
    trainloader, testloader = init_embedding_data(args, 'train')
    train_embedding(cur_epoch, embedder, optimizer, trainloader, testloader, device, args)

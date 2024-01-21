
import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import wandb
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from utils.metric_util import MeanIoU
from utils.load_save_util import revise_ckpt, revise_ckpt_2
from dataloader.dataset import get_nuScenes_label_name
from builder import loss_builder

import mmcv
from mmcv import Config
from mmcv.runner import build_optimizer
from mmseg.utils import get_root_logger
from timm.scheduler import CosineLRScheduler
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.9, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Flatten inputs and targets
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        BCE_loss = F.binary_cross_entropy_with_logits(inputs_flat, targets_flat, reduction='none')
        pt = torch.exp(-BCE_loss)

        # Adjusting alpha for binary classification
        alpha = torch.tensor([1-self.alpha, self.alpha], device=inputs.device)
        alpha = alpha.gather(0, targets_flat.long())

        F_loss = alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss



import warnings
warnings.filterwarnings("ignore")

save_val_scene_token = ['e7ef871f77f44331aefdebc24ec034b7', '55b3a17359014f398b6bbd90e94a8e1b', '85889db3628342a482c5c0255e5347e9', '5301151d8b6a42b0b252e95634bd3995', '952cb0bcd89b4ca4b904cdcbbf595523', 'fb73d1a6c16147ee9416faf6b310fadb']

def masked_loss(outputs, targets, mask, step, loss_type='l2', rescale=True, eps=1e-8, ):
    # Apply mask
    masked_outputs = outputs * mask
    masked_targets = targets * mask

    # Calculate loss
    if loss_type == 'l1':
        loss = F.l1_loss(masked_outputs, masked_targets, reduction='none')
    elif loss_type == 'l2':
        loss = F.mse_loss(masked_outputs, masked_targets, reduction='none')
    else:
        raise ValueError("Invalid loss_type. Supported types: 'l1', 'l2'")

    # Sum over all elements and divide by the number of masked elements plus epsilon
    sum_loss = loss.sum()
    num_masked_elements = 64 * mask.sum().float()
    mean_loss = sum_loss / (num_masked_elements + eps) if rescale else sum_loss / (num_masked_elements + eps)

    return mean_loss

def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    dataset_config = cfg.dataset_params
    ignore_label = dataset_config['ignore_label']
    version = dataset_config['version']
    train_dataloader_config = cfg.train_data_loader
    val_dataloader_config = cfg.val_data_loader

    max_num_epochs = cfg.max_epochs
    grid_size = cfg.grid_size

    # init DDP
    distributed = True
    ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT", "20506")
    hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
    rank = int(os.environ.get("RANK", 0))  # node id
    gpus = torch.cuda.device_count()  # gpus per node
    print(f"tcp://{ip}:{port}")
    dist.init_process_group(
        backend="nccl", init_method=f"tcp://{ip}:{port}", 
        world_size=hosts * gpus, rank=rank * gpus + local_rank
    )
    world_size = dist.get_world_size()
    cfg.gpu_ids = range(world_size)
    torch.cuda.set_device(local_rank)

    if dist.get_rank() != 0:
        import builtins
        builtins.print = pass_print

    # configure logger
    if dist.get_rank() == 0:

        # Generate a timestamped run name
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        run_name = f"{current_time}-TPVformer_w_NeRF_exp0_L2_loss"
        print(f"Starting training run {run_name}")
        
        
        work_dir_base = args.work_dir
        args.work_dir = os.path.join(args.work_dir, run_name)
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

        # Initialize wandb
        # wandb.init(project='Neural-Driving_Field', entity='neural-scenes', name=run_name)

        # Path for TensorBoard logs
        log_dir = os.path.join("./tensorboar_log", run_name)
        
        val_save_dir = osp.join(args.work_dir, 'val_output', run_name)
        # mkdir allow exist
        os.makedirs(val_save_dir, exist_ok=True)

        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir)
        

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    if cfg.get('occupancy', False):
        from builder import tpv_occupancy_builder as model_builder
    else:
        from builder import tpv_lidarseg_builder as model_builder
    
    my_model = model_builder.build(cfg.model)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        my_model = my_model.cuda()
    print('done ddp model')

    # generate datasets
    SemKITTI_label_name = get_nuScenes_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(cfg.unique_label)
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label]

    from builder import data_builder
    train_dataset_loader, val_dataset_loader = \
        data_builder.build_w_nerf(
            dataset_config,
            train_dataloader_config,
            val_dataloader_config,
            grid_size=grid_size,
            version=version,
            dist=distributed,
            scale_rate=cfg.get('scale_rate', 1)
        )


    # get optimizer, loss, scheduler
    optimizer = build_optimizer(my_model, cfg.optimizer)
    loss_func, lovasz_softmax = \
        loss_builder.build(ignore_label=ignore_label)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=len(train_dataset_loader)*max_num_epochs,
        lr_min=1e-6,
        warmup_t=500,
        warmup_lr_init=1e-5,
        t_in_epochs=False
    )
    
    CalMeanIou_vox = MeanIoU(unique_label, ignore_label, unique_label_str, 'vox')
    CalMeanIou_pts = MeanIoU(unique_label, ignore_label, unique_label_str, 'pts')
    
    # resume and load
    epoch = 0
    best_val_miou_pts, best_val_miou_vox = 0, 0
    global_iter_train = 0

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    print('resume from: ', cfg.resume_from)
    print('work dir: ', args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(my_model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=False))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        if 'best_val_miou_pts' in ckpt:
            best_val_miou_pts = ckpt['best_val_miou_pts']
        if 'best_val_miou_vox' in ckpt:
            best_val_miou_vox = ckpt['best_val_miou_vox']
        global_iter_train = ckpt['global_iter_train']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        state_dict = revise_ckpt(state_dict)
        try:
            print(my_model.load_state_dict(state_dict, strict=False))
        except:
            state_dict = revise_ckpt_2(state_dict)
            print(my_model.load_state_dict(state_dict, strict=False))
        

    # training
    print_freq = cfg.print_freq

    while epoch < max_num_epochs:
        my_model.train()
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        
        time.sleep(10)
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, (imgs, img_metas, train_nerf_vox_feat, train_occ_label) in enumerate(train_dataset_loader):
            #ipdb.set_trace()
            imgs = imgs.cuda()
            if cfg.lovasz_input == 'voxel' or cfg.ce_input == 'voxel':
                train_nerf_vox_feat = train_nerf_vox_feat.cuda()
            # forward + backward + optimize
            data_time_e = time.time()
            train_nerf_vox_feat = train_nerf_vox_feat.cuda()
            train_occ_label = train_occ_label.cuda()
            outputs_vox_feat, outputs_vox_occ_label = my_model(img=imgs, img_metas=img_metas)
            
            # train_nerf_vox_feat and outputs_vox_feat has shape [bsz, embed_dim, H, W, Z]
            # train_occ_label and outputs_vox_occ_label has shape [bsz, 1, H, W, Z]
            
            # Step 1: calculate loss for voxel occ label
            bce_loss_func = nn.BCELoss()
            focal_loss_func = FocalLoss()
            occ_loss_train = bce_loss_func(outputs_vox_occ_label, train_occ_label)
            
            # Step 2: calculate loss for voxel feat with occ mask

            # Assuming outputs_vox_feat and train_nerf_vox_feat are your output and target tensors
            mask = train_occ_label == 1  # Creating a mask where train_occ_label is 1

            feat_loss_train = masked_loss(outputs = outputs_vox_feat, targets = train_nerf_vox_feat, mask = mask, step = global_iter_train)
            
            total_loss_train = feat_loss_train + occ_loss_train
                
            optimizer.zero_grad()
            total_loss_train.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
            optimizer.step()
            scheduler.step_update(global_iter_train)
            time_e = time.time()

            global_iter_train += 1
            if i_iter % print_freq == 0 and dist.get_rank() == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info('[TRAIN] Epoch %d Iter %5d/%d: Total_Loss: %.4f, Occ_Loss: %.4f, Feat_Loss: %.4f, LR: %.6f, Grad_Norm: %.4f, Step_Time: %.4f, Data_Time: %.4f'%(epoch, i_iter, len(train_dataset_loader), total_loss_train.item(), occ_loss_train.item(), feat_loss_train.item(), lr, grad_norm, time_e - time_s, data_time_e - data_time_s))
            if dist.get_rank() == 0:
                #wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/grad_norm": grad_norm, "train/step_time": time_e - time_s, "train/data_time": data_time_e - data_time_s})
                writer.add_scalar("train/total_loss", total_loss_train.item(), global_iter_train)
                writer.add_scalar("train/occ_loss", occ_loss_train.item(), global_iter_train)
                writer.add_scalar("train/feat_loss", feat_loss_train.item(), global_iter_train)
                writer.add_scalar("train/lr", lr, global_iter_train)
                writer.add_scalar("train/grad_norm", grad_norm, global_iter_train)
                writer.add_scalar("train/step_time", time_e - time_s, global_iter_train)
                writer.add_scalar("train/data_time", data_time_e - data_time_s, global_iter_train)
                
            data_time_s = time.time()
            time_s = time.time()
        
        # save checkpoint
        if dist.get_rank() == 0:
            dict_to_save = {
                'state_dict': my_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter_train': global_iter_train,
                'best_val_miou_pts': best_val_miou_pts,
                'best_val_miou_vox': best_val_miou_vox
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch+1}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(args.work_dir, 'latest.pth')
            mmcv.symlink(save_file_name, dst_file)

        epoch += 1
        
        # eval
        my_model.eval()
        # CalMeanIou_pts.reset()
        # CalMeanIou_vox.reset()

        total_loss_val_list = []
        feat_loss_val_list = []
        occ_loss_val_list = []
        with torch.no_grad():
            for i_iter_val, (imgs, img_metas, val_nerf_vox_feat, val_occ_label) in enumerate(val_dataset_loader):
                imgs = imgs.cuda()
                val_nerf_vox_feat = val_nerf_vox_feat.cuda()
                val_occ_label = val_occ_label.cuda()

                outputs_vox_feat_val, outputs_vox_occ_label_val = my_model(img=imgs, img_metas=img_metas)
                # Step 1: calculate loss for voxel occ label
                bce_loss_func = nn.BCELoss()
                focal_loss_func = FocalLoss()
                occ_loss_val = bce_loss_func(outputs_vox_occ_label_val, val_occ_label)
                
                # Step 2: calculate loss for voxel feat with occ mask
                mask = val_occ_label == 1 # Creating a mask where train_occ_label is 1
                feat_loss_val = masked_loss(outputs = outputs_vox_feat_val, targets = val_nerf_vox_feat, mask = mask)
                
                total_loss_val = feat_loss_val + occ_loss_val
                
                total_loss_val_list.append(total_loss_val.item())
                feat_loss_val_list.append(feat_loss_val.item())
                occ_loss_val_list.append(occ_loss_val.item())
                
                if i_iter_val % print_freq == 0 and dist.get_rank() == 0:
                    logger.info('[EVAL] Epoch %d Iter %5d: Total Loss: %.4f, Occ Loss: %.4f, Feat Loss: %.4f'%(epoch, i_iter_val, total_loss_val.item(), occ_loss_val.item(), feat_loss_val.item()))
                if dist.get_rank() == 0:
                    # wandb.log({"val/loss": loss.item()})
                    writer.add_scalar("val/toal_loss_mean", np.mean(total_loss_val_list), global_iter_train)
                    writer.add_scalar("val/occ_loss_mean", np.mean(occ_loss_val_list), global_iter_train)
                    writer.add_scalar("val/feat_loss_mean", np.mean(feat_loss_val_list), global_iter_train)
                if img_metas[0]['scene_token'] in save_val_scene_token:
                    save_base_dir = osp.join(val_save_dir, "epoch_{}".format(epoch), img_metas[0]['scene_token'])
                    os.makedirs(save_base_dir, exist_ok=True)
                    outputs_vox_feat_val = outputs_vox_feat_val.detach().cpu().numpy()
                    save_path_feat = osp.join(save_base_dir, "{}-{}_pred_vox_feat.npy".format(i_iter_val, img_metas[0]['sample_idx']))
                    np.save(save_path_feat, outputs_vox_feat_val)
                    #logger.info('save pred vox feat to {}'.format(save_path_feat))
                    outputs_vox_occ_label_val = outputs_vox_occ_label_val.detach().cpu().numpy()
                    save_path_occ = osp.join(save_base_dir, "{}-{}_pred_occ.npy".format(i_iter_val, img_metas[0]['sample_idx']))
                    np.save(save_path_occ, outputs_vox_occ_label_val)
                    #logger.info('save pred vox feat to {}'.format(save_path_occ))
                    
                    
        
        # val_miou_pts = CalMeanIou_pts._after_epoch()
        # val_miou_vox = CalMeanIou_vox._after_epoch()

        # if best_val_miou_pts < val_miou_pts:
        #     best_val_miou_pts = val_miou_pts
        # if best_val_miou_vox < val_miou_vox:
        #     best_val_miou_vox = val_miou_vox

        # logger.info('Current val miou pts is %.4f while the best val miou pts is %.4f' %
        #         (val_miou_pts, best_val_miou_pts))
        # logger.info('Current val miou vox is %.4f while the best val miou vox is %.4f' %
        #         (val_miou_vox, best_val_miou_vox))
        # logger.info('Current val loss is %.4f' %
        #         (np.mean(val_loss_list)))
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')

    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(f"Avaliable GPU: {ngpus}")
    print(args)
    main(0,args)
    # torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)

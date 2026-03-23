import copy
import logging
import pickle
import random
from collections import defaultdict
from typing import Tuple, List

import numpy as np
import torch
from deepspeed.profiling.flops_profiler import get_model_profile
from termcolor import colored


class ObjDict(dict):
    """
    Makes a dictionary behave like an object, with attribute-style access.
    """

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __deepcopy__(self, name):
        copy_dict = dict()
        for key, value in self.items():
            if hasattr(value, '__deepcopy__'):
                copy_dict[key] = copy.deepcopy(value)
            else:
                copy_dict[key] = value
        return ObjDict(copy_dict)

    def __getstate__(self):
        return pickle.dumps(self.__dict__)

    def __setstate__(self, state):
        self.__dict__ = pickle.loads(state)

    def __exists__(self, name):
        return name in self.__dict__


def set_random_seed(seed=0, benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = True 




def resume_state(model, ckpt, **args):
    state = torch.load(ckpt)
    model.load_state_dict(state['model'], strict=True)
    logging.info(f'loaded model state from {ckpt}')

    for i in args.keys():
        if hasattr(args[i], "load_state_dict"):
            args[i].load_state_dict(state[i])
            logging.info(f'loaded {i} state from {ckpt}')
    return state


def load_state(model, ckpt, **args):
    state = torch.load(ckpt)
    model_state_dict = state['model']
    

    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(model_state_dict.keys())
    

    if not model_keys.intersection(checkpoint_keys) and any(key.startswith('module.') for key in checkpoint_keys):
        logging.info("Removing 'module.' prefix from checkpoint keys")
        new_state_dict = {}
        for key, value in model_state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        model_state_dict = new_state_dict

    elif not model_keys.intersection(checkpoint_keys) and any(key.startswith('module.') for key in model_keys):
        logging.info("Adding 'module.' prefix to checkpoint keys")
        new_state_dict = {}
        for key, value in model_state_dict.items():
            new_key = f'module.{key}'
            new_state_dict[new_key] = value
        model_state_dict = new_state_dict
    
    if hasattr(model, 'module'):
        incompatible = model.module.load_state_dict(model_state_dict, strict=False)
    else:
        incompatible = model.load_state_dict(model_state_dict, strict=False)
    logging.info(f'loaded model state from {ckpt}')

    if incompatible.missing_keys:
        logging.warning('missing_keys')
        logging.warning(
            get_missing_parameters_message(incompatible.missing_keys),
        )
    if incompatible.unexpected_keys:
        logging.warning('unexpected_keys')
        logging.warning(
            get_unexpected_parameters_message(incompatible.unexpected_keys)
        )
    for i in args.keys():
        if i not in state.keys():
            logging.warning(f'missing {i} state in state_dict, just skipped')
            continue
        if hasattr(args[i], "load_state_dict"):
            args[i].load_state_dict(state[i])
            logging.info(f'loaded {i} state from {ckpt}')
    return state


def save_state(ckpt, **args):
    state = {}
    for i in args.keys():
        item = args[i].state_dict() if hasattr(args[i], "state_dict") else args[i]
        state[i] = item
    torch.save(state, ckpt)


def cal_model_params(model):
    total = sum([param.nelement() for param in model.parameters()])
    trainable = sum([param.nelement() for param in model.parameters() if param.requires_grad])

    return total, trainable


def cal_model_flops(model, inputs: [List | Tuple] = None, profile=True, warmup=10):
    flops, macs, params = get_model_profile(
        model=model,
        args=inputs,
        print_profile=profile,  # prints the model graph with the measured profile attached to each module
        detailed=True,  # print the detailed profile
        warm_up=warmup,  # the number of warm-ups before measuring the time of each module
        as_string=False,  # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
        output_file=None,  # path to the output file. If None, the profiler prints to stdout.
        ignore_modules=None)  # the list of modules to ignore in the profiling
    return flops, macs, params


def get_missing_parameters_message(keys):
    groups = _group_checkpoint_keys(keys)
    msg = "Some model parameters or buffers are not found in the checkpoint:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "blue") for k, v in groups.items()
    )
    return msg


def get_unexpected_parameters_message(keys):
    groups = _group_checkpoint_keys(keys)
    msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "magenta") for k, v in groups.items()
    )
    return msg


def _group_checkpoint_keys(keys):
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1:]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups


def _group_to_str(group):
    if len(group) == 0:
        return ""

    if len(group) == 1:
        return "." + group[0]

    return ".{" + ", ".join(group) + "}"


def convert_ddp_state_dict(state_dict, remove_module_prefix=True):

    new_state_dict = {}
    for key, value in state_dict.items():
        if remove_module_prefix and key.startswith('module.'):
            new_key = key[7:]  
        elif not remove_module_prefix and not key.startswith('module.'):
            new_key = f'module.{key}'  
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def save_state_ddp(ckpt_path, local_rank, save_strategy='rank0_only', **args):

    import torch.distributed as dist
    
    if save_strategy == 'rank0_only':
        
        if local_rank == 0:
            save_state(ckpt_path, **args)
            logging.info(f'Model saved by rank 0 to {ckpt_path}')
    
    elif save_strategy == 'all_ranks':
        
        rank_ckpt_path = ckpt_path.replace('.ckpt', f'_rank{local_rank}.ckpt')
        save_state(rank_ckpt_path, **args)
        logging.info(f'Model saved by rank {local_rank} to {rank_ckpt_path}')
    
    elif save_strategy == 'separate_files':
        import os
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_name = os.path.basename(ckpt_path)
        rank_dir = os.path.join(ckpt_dir, f'rank_{local_rank}')
        os.makedirs(rank_dir, exist_ok=True)
        rank_ckpt_path = os.path.join(rank_dir, ckpt_name)
        save_state(rank_ckpt_path, **args)
        logging.info(f'Model saved by rank {local_rank} to {rank_ckpt_path}')
    

    if dist.is_initialized():
        dist.barrier()


def save_state_ddp_partial(ckpt_path, local_rank, model_parts=None, **args):

    import torch.distributed as dist
    
    if model_parts is None:

        save_state_ddp(ckpt_path, local_rank, 'rank0_only', **args)
        return
    
    if local_rank in model_parts:
        model = args.get('model')
        if model is not None:

            partial_state_dict = {}
            full_state_dict = model.state_dict()
            
            for part_name in model_parts[local_rank]:
                for key in full_state_dict:
                    if key.startswith(part_name):
                        partial_state_dict[key] = full_state_dict[key]

            partial_ckpt_path = ckpt_path.replace('.ckpt', f'_rank{local_rank}_partial.ckpt')
            
            state = {'model_partial': partial_state_dict}
            for key, value in args.items():
                if key != 'model':
                    item = value.state_dict() if hasattr(value, "state_dict") else value
                    state[key] = item
            
            torch.save(state, partial_ckpt_path)
            logging.info(f'Partial model saved by rank {local_rank} to {partial_ckpt_path}')
            logging.info(f'Saved parts: {model_parts[local_rank]}')
    
    if dist.is_initialized():
        dist.barrier()


def load_state_ddp_partial(model, ckpt_dir, ranks, **args):

    import os
    import glob
    
    full_state_dict = {}

    for rank in ranks:
        partial_files = glob.glob(os.path.join(ckpt_dir, f'*_rank{rank}_partial.ckpt'))
        if partial_files:
            partial_ckpt = partial_files[0] 
            logging.info(f'Loading partial checkpoint from rank {rank}: {partial_ckpt}')
            
            state = torch.load(partial_ckpt)
            partial_state_dict = state['model_partial']
            
  
            full_state_dict.update(partial_state_dict)
    

    if hasattr(model, 'module'):
        incompatible = model.module.load_state_dict(full_state_dict, strict=False)
    else:
        incompatible = model.load_state_dict(full_state_dict, strict=False)
    
    logging.info(f'Loaded merged model state from {len(ranks)} ranks')
    
    if incompatible.missing_keys:
        logging.warning('missing_keys')
        logging.warning(get_missing_parameters_message(incompatible.missing_keys))
    if incompatible.unexpected_keys:
        logging.warning('unexpected_keys')
        logging.warning(get_unexpected_parameters_message(incompatible.unexpected_keys))
    
    return full_state_dict


def all_gather_tensors(tensor, local_rank=None):

    import torch.distributed as dist
    
    if not dist.is_initialized():
        return [tensor]
    
    world_size = dist.get_world_size()
    current_rank = dist.get_rank()
    

    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    

    dist.all_gather(gathered_tensors, tensor)
    

    if current_rank == 0:
        return gathered_tensors
    else:
        return None


def all_gather_varying_tensors(tensor, local_rank=None):

    import torch.distributed as dist
    
    if not dist.is_initialized():
        return tensor
    
    world_size = dist.get_world_size()
    current_rank = dist.get_rank()

    if tensor.device.type == 'cpu':

        if local_rank is not None:
            tensor = tensor.cuda(local_rank)
        else:
            tensor = tensor.cuda()
    

    local_size = torch.tensor([tensor.shape[0]], dtype=torch.long, device=tensor.device)
    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    

    max_size = max(size_list).item()

    if tensor.shape[0] < max_size:
        pad_size = max_size - tensor.shape[0]
        padding_shape = (pad_size,) + tensor.shape[1:]
        padding = torch.zeros(padding_shape, dtype=tensor.dtype, device=tensor.device)
        padded_tensor = torch.cat([tensor, padding], dim=0)
    else:
        padded_tensor = tensor
    

    gathered_tensors = [torch.zeros_like(padded_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, padded_tensor)
    

    if current_rank == 0:

        valid_tensors = []
        for i, gathered_tensor in enumerate(gathered_tensors):
            valid_size = size_list[i].item()
            valid_tensors.append(gathered_tensor[:valid_size])
        
        return torch.cat(valid_tensors, dim=0)
    else:
        return None


def gather_predictions_and_targets(predictions, targets, local_rank=None):

    import torch.distributed as dist
    
    if not dist.is_initialized():
        return predictions, targets
    
    current_rank = dist.get_rank()
    

    all_predictions = all_gather_varying_tensors(predictions, local_rank)
    all_targets = all_gather_varying_tensors(targets, local_rank)
    
    if current_rank == 0:
        return all_predictions.cpu(), all_targets.cpu()
    else:
        return None, None


def distributed_validate(cfg, model, val_loader, epoch, local_rank, metric_calculator=None):

    import torch.nn.functional as F
    from torch.cuda.amp import autocast
    import torch.distributed as dist
    from tqdm import tqdm
    
    model.eval()
    

    all_predictions = []
    all_targets = []
    total_loss = 0.0
    num_samples = 0
    

    if dist.get_rank() == 0:
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc='Distributed Val')
    else:
        pbar = enumerate(val_loader)
    
    with torch.no_grad():
        for idx, cam_points in pbar:
            cam_points.to_cuda(device=f'cuda:{local_rank}', non_blocking=True)
            target = cam_points.y
            
            with autocast():
                pred = model(cam_points)
                loss = F.cross_entropy(pred, target, label_smoothing=cfg.ls, ignore_index=cfg.ignore_index)
            

            pred_probs = F.softmax(pred, dim=1)
            pred_labels = torch.argmax(pred_probs, dim=1)
            

            all_predictions.append(pred_labels)
            all_targets.append(target)
            
            total_loss += loss.item()
            num_samples += 1
            
            if dist.get_rank() == 0:
                pbar.set_description(f"Distributed Val Epoch [{epoch}] Loss {loss:.4f}")
    

    local_predictions = torch.cat(all_predictions, dim=0)
    local_targets = torch.cat(all_targets, dim=0)
    

    global_predictions, global_targets = gather_predictions_and_targets(
        local_predictions, local_targets, local_rank
    )
    

    local_loss_tensor = torch.tensor([total_loss, num_samples], device=f'cuda:{local_rank}')
    loss_info = [torch.zeros_like(local_loss_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(loss_info, local_loss_tensor)
    

    if dist.get_rank() == 0:
 
        total_global_loss = sum([info[0].item() for info in loss_info])
        total_global_samples = sum([info[1].item() for info in loss_info])
        avg_loss = total_global_loss / max(total_global_samples, 1)
        

        if metric_calculator is not None:
            metrics = metric_calculator(global_predictions, global_targets, cfg.num_classes)
        else:
            metrics = calculate_classification_metrics(global_predictions, global_targets, cfg.num_classes, cfg.ignore_index)
        
        logging.info(f'@E{epoch} Global validation results:')
        logging.info(f'  Total samples: {len(global_predictions)}')
        logging.info(f'  Average loss: {avg_loss:.4f}')
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logging.info(f'  {key}: {value:.4f}')
        
        return avg_loss, metrics
    else:

        return None, None


def calculate_classification_metrics(predictions, targets, num_classes, ignore_index=-100):
    
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    valid_mask = targets != ignore_index
    valid_preds = predictions[valid_mask].numpy()
    valid_targets = targets[valid_mask].numpy()
    
    if len(valid_preds) == 0:
        return {'accuracy': 0.0, 'macc': 0.0, 'miou': 0.0}
    
    overall_acc = accuracy_score(valid_targets, valid_preds)
    
    cm = confusion_matrix(valid_targets, valid_preds, labels=list(range(num_classes)))
    class_accs = []
    for i in range(num_classes):
        if cm[i, :].sum() > 0:
            class_acc = cm[i, i] / cm[i, :].sum()
            class_accs.append(class_acc)
    
    mean_acc = np.mean(class_accs) if class_accs else 0.0
    
    ious = []
    for i in range(num_classes):
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - intersection
        if union > 0:
            iou = intersection / union
            ious.append(iou)
    
    mean_iou = np.mean(ious) if ious else 0.0
    
    return {
        'accuracy': overall_acc,
        'macc': mean_acc,
        'miou': mean_iou,
        'ious': ious,
        'class_accs': class_accs
    }


def create_custom_metric_calculator(metric_class):
    
    def metric_calculator(predictions, targets, num_classes):
        metric = metric_class(num_classes)
        metric.update(predictions.unsqueeze(0), targets.unsqueeze(0))
        
        acc, macc, miou, iou = metric.calc()
        
        return {
            'accuracy': acc,
            'macc': macc,
            'miou': miou,
            'ious': iou
        }
    
    return metric_calculator


def create_s3dis_metric_calculator():
    def s3dis_metric_calculator(predictions, targets, num_classes):
        from utils.metrics import Metric
        
        if hasattr(predictions, 'cpu'):
            predictions = predictions.cpu()
        if hasattr(targets, 'cpu'):
            targets = targets.cpu()
        
        metric = Metric(num_classes, device='cpu')
        
        batch_size = 8192  
        for i in range(0, len(predictions), batch_size):
            end_idx = min(i + batch_size, len(predictions))
            batch_pred = predictions[i:end_idx]
            batch_target = targets[i:end_idx]
            
            pred_logits = torch.zeros(batch_pred.shape[0], num_classes, dtype=torch.float32)
            pred_logits.scatter_(1, batch_pred.unsqueeze(1).long(), 1.0)
            
            batch_target = batch_target.long()
            
            metric.update(pred_logits, batch_target)
        
        acc, macc, miou, iou = metric.calc()
        
        return {
            'loss': 0.0,  
            'miou': miou,
            'macc': macc,
            'ious': iou,
            'accs': acc
        }
    
    return s3dis_metric_calculator

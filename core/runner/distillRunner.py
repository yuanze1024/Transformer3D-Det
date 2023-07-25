from core.utils.utils import save_checkpoint
from .runner_utils.utils import transform_input
from .runner_utils.testRunnerUtils import testmodel
import torch
import time
import traceback
from core.other.optimizer import get_optimizer
from torch.nn.functional import huber_loss

def print_grad(model, keyword=None):
    print('Calculate grad')
    for name, param in model.named_parameters():
        # if 'weight' not in name:
        #     continue
        if keyword is not None and keyword not in name:
            continue
        print(name, 'max: grad[%.5f] value[%.5f]' %(float(torch.max(param.grad).cpu()), float(torch.max(param.data).cpu())), end=' ; ')
        print('std: grad[%.5f] value[%.5f]' %(float(param.grad.detach().std().cpu()), float(param.data.detach().std().cpu())), 'shape', param.shape, flush=True)
        # print('real value', param.grad.cpu()[:3, :], param.data.cpu()[:3, :], flush=True)

def compute_distill_loss(refined_vote_xyz, refined_vote_features, vote_xyz, aligned_vote_feature):
    """
    refined_vote_xyz: [B, N, 3]
    refined_vote_features: [B, N, 288]
    vote_xyz: [B, M, 3]
    aligned_vote_feature: [B, M, 288]

    regard the refined_vote_xyz as the ground truth, find the nearest refined_vote for each vote_xyz,
    and calculate the mse loss between the aligned_vote_feature and corresponding refined_vote_features
    """
    B, N, _ = refined_vote_xyz.shape
    _, M, _ = vote_xyz.shape
    dist = torch.sum((refined_vote_xyz.view(B, N, 1, 3) - vote_xyz.view(B, 1, M, 3)) ** 2, dim=-1)  # [B, N, M]
    _, min_dist_ind = torch.min(dist, dim=1)  # [B, M]
    min_dist_ind = min_dist_ind.unsqueeze(-1).repeat(1, 1, 288)  # [B, M, 288]
    # standardize the refined_vote_features and aligned_vote_feature in the feature dimension
    refined_vote_features = refined_vote_features.div(torch.norm(refined_vote_features, dim=-1, keepdim=True, p=2))
    aligned_vote_feature = aligned_vote_feature.div(torch.norm(aligned_vote_feature, dim=-1, keepdim=True, p=2))
    refined_vote_features = torch.gather(refined_vote_features, dim=1, index=min_dist_ind)  # [B, M, 288]
    distill_loss = huber_loss(refined_vote_features, aligned_vote_feature) / B
    return distill_loss

def distillRunner(info):
    config = info['config']
    train_loader_iter = iter(info['traindataloader'])
    optimizer = info['optimizer']
    lr_scheduler = info['lr_scheduler']
    model = info['model']
    model_t = info['model_t']
    loggers = info['loggers']
    lowest_error = info['lowest_error']
    last_iter = info['last_iter']
    teacher_optimizer_config = info['teacher_optimizer_config']
    clip_grad_norm = config.get('clip_grad_norm', None)
    if clip_grad_norm is not None:
        print('CLIP GRAD NORM! MAX =', clip_grad_norm)
    t_start = time.time()
    T_START = time.time()
    model_t.test_mode()
    model_t.net.refine_module.train()
    optimizer_t = get_optimizer(teacher_optimizer_config, model_t.net.refine_module.parameters())
    if isinstance(model, torch.nn.DataParallel):
        model.module.train_mode()
    elif isinstance(model, torch.nn.Module):
        model.train_mode()  # change mode
    else:
        raise NotImplementedError(type(model))
    print('last_iter:', last_iter)
    max_tries = 3
    for iter_id in range(last_iter + 1, config.max_iter + 1):
        for tries in range(max_tries):
            try:
                input = next(train_loader_iter)
                break
            except Exception as e:
                if isinstance(e, StopIteration):
                    print('Start A New Epoch', flush=True)
                else:
                    if tries == max_tries - 1:
                        raise e
                    print('dataloader exception', str(e))
                    print(traceback.format_exc())
                train_loader_iter = iter(info['traindataloader'])
        input = transform_input(input)
        optimizer.zero_grad()
        optimizer_t.zero_grad()
        t_loader = time.time()
        output_t = model_t(input)
        # 'aggregated_vote_xyz', 'aggregated_vote_features', 'aggregated_vote_inds'
        refined_vote_xyz = output_t['aggregated_vote_xyz']
        refined_vote_features = output_t['refined_vote_feature']
        output = model(input, output_t)  # also could backward inside
        vote_xyz = output['vote_xyz']
        aligned_vote_feature = output['aligned_features']
        t_forward = time.time()
        if isinstance(model, torch.nn.DataParallel):
            # mutli-batch; for data-parallel-model use
            for key, value in output.items():
                if 'loss' in key:
                    output[key] = torch.mean(value, dim=0)
        assert 'loss' in output.keys(), 'Key "loss" should in output.keys'
        loss = output['loss']
        loss += compute_distill_loss(refined_vote_xyz, refined_vote_features, vote_xyz, aligned_vote_feature)
        # print(loss)
        loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            # print('clip_max_norm', clip_grad_norm, flush=True)
            pass
        # model.average_gradients()  # multi card sync
        # print_grad(model, 'weight')
        # if iter_id % 1000 == 0:# or True:# and False:  # just print
        # if True:# and False:  # just print
        #     print_grad(model, '.0.weight')  # conv_first
        optimizer.step()
        optimizer_t.step()
        # print('backward okay') # for test
        output['iteration'] = [iter_id, config.max_iter, (iter_id + 1) / len(info['traindataloader'])]
        output['loader_time'] = t_loader - t_start
        output['forward_time'] = t_forward - t_loader
        t_tmp = time.time()
        output['update_time'] = t_tmp - t_forward
        output['time'] = t_tmp - t_start
        output['mean_time_iter'] = (t_tmp - T_START) / (iter_id - last_iter)
        t_start = t_tmp
        output['lr'] = lr_scheduler.get_lr()[0]
        with torch.no_grad(): # 省着切换train和val模式了
            if iter_id != -1 and (iter_id % config.test_freq == 0 or iter_id % config.save_freq == 0):
                if isinstance(model, torch.nn.DataParallel):
                    model.module.val_mode()
                elif isinstance(model, torch.nn.Module):
                    model.val_mode()  # change mode
                else:
                    raise NotImplementedError(type(model))
                output_error = {}
                error, weight, test_time = [], [], 0.
                for testset_name, loader in info['testdataloaders'].items():
                    _error, _weight = testmodel(model, loader, loggers, config.log_freq, testset_name, iter_id)
                    error.append(_error)
                    weight.append(_weight)
                    test_time += time.time() - t_start
                    t_start = time.time()
                    output_error[testset_name + '_error'] = _error
                error_final = sum(error) / sum(weight)  # calculate mean
                # for logger
                output_error['time'] = test_time
                output_error['test_time'] = test_time
                output_error['error'] = error_final
                output_error['prev_lowest_error'] = lowest_error
                output_error['flush'] = True
                output_error['n_count'] = 1
                loggers.update_error(output_error, True)  # similiar as model.val
                is_best = error_final < lowest_error
                if is_best or iter_id % config.save_freq == 0:
                    if is_best:
                        lowest_error = error_final
                    save_checkpoint({
                        'step': iter_id,
                        'state_dict': model.state_dict(),
                        'lowest_error': lowest_error,
                        'optimizer': optimizer.state_dict(),
                    }, is_best, config.snapshot_save_path + '/ckpt' + '_' + str(iter_id))
                if isinstance(model, torch.nn.DataParallel):
                    model.module.train_mode()
                elif isinstance(model, torch.nn.Module):
                    model.train_mode()  # change mode
                else:
                    raise NotImplementedError(type(model))
        lr_scheduler.step()
        loggers.update_loss(output, iter_id % config.log_freq == 0)  # TODO
    print('training: done')

from core.utils.utils import save_checkpoint
from .runner_utils.utils import transform_input
from .runner_utils.testRunnerUtils import testmodel
import torch
import time
import traceback
from torch.nn.functional import huber_loss

from ray.air import session

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

# def get_seed_foreground_mask(seed_inds, vote_label_mask) -> torch.Tensor:
#     """ 
#     返回前景的mask。
#     return:
#     mask: (batch_size, n_point)
#     """
#     # index 需要是 int64类型，but why?
#     seed_inds = seed_inds.long()
#     foreground_mask = torch.gather(vote_label_mask, 1, seed_inds).float()
#     zero_tensor = (foreground_mask == 0)
#     num_bg = torch.sum(zero_tensor, axis=1)
#     for i_batch in range(zero_tensor.shape[0]):
#         if num_bg[i_batch] != 0:
#             # 令背景的权重为每个batch中背景点个数的倒数
#             foreground_mask[i_batch][foreground_mask[i_batch] == 0] = 1. / num_bg[i_batch]
#     return foreground_mask

SEED_DISTILL_WEIGHT = 1.
VOTE_DISTILL_WEIGHT = 1.

# def compute_seed_distill_loss(pred_s, pred_t, foreground_mask=1.):
#     # pred_s, pred_t (B, C, npoint)
#     global SEED_DISTILL_WEIGHT
#     batch_size, _, num_tokens = pred_s.shape

#     mse_loss = torch.sum(((pred_s-pred_t)*foreground_mask)**2)
#     if type(foreground_mask) == float:
#         mse_loss /= (batch_size * num_tokens)
#     else:
#         mse_loss /= (torch.sum(foreground_mask) + 1e-7)
#     return mse_loss*SEED_DISTILL_WEIGHT
def get_valid_vote_mask(vote_xyz, gt_box_center, gt_box_size):
    """
    vote_xyz: [B, N, 3]
    gt_box_center: [B, O, 3]
    gt_box_size: [B, O, 3]
    return: [B, N]
    """
    B, N, _ = vote_xyz.shape
    _, O, _ = gt_box_center.shape
    vote_xyz = vote_xyz.unsqueeze(2).repeat(1, 1, O, 1)  # [B, N, O, 3]
    gt_box_center = gt_box_center.unsqueeze(1).repeat(1, N, 1, 1)  # [B, N, O, 3]
    # mask = torch.abs(vote_xyz - gt_box_center) < gt_box_size.unsqueeze(1).repeat(1, N, 1, 1) / 2  # [B, N, O, 3]
    # mask = torch.all(mask, dim=-1)  # [B, N, O]
    # mask = torch.any(mask, dim=-1)  # [B, N]
    # return mask, 100 * torch.sum(mask) / (B * N * O * 3)
    gt_box_size = gt_box_size.unsqueeze(1).repeat(1, N, 1, 1)   # [B, N, O, 3]
    dist_ratio = torch.abs(vote_xyz - gt_box_center) / (gt_box_size / 2 + 1e-7) # [B, N, O, 3]
    mask = 1.0 / (1 + torch.sum(dist_ratio**2, dim=-1))   # [B, N, O]
    # mask = 1 - torch.sum(dist_ratio**2, dim=-1)
    mask, _ = torch.max(mask, dim=-1)   # [B, N]，得到每个refined_vote在O个目标框中最靠近gt_center的
    mask = mask * (mask >= 0).float()   # 将比例平方和大于1的点mask置为0
    percent = torch.sum(mask >= 0.5) / (B * N) # 每个refined_vote落在检测框中（内接椭球）的比例。
    return mask, percent

def compute_vote_distill_loss(refined_vote_xyz, refined_vote_features, vote_xyz, aligned_vote_feature, vote_mask):
    """
    refined_vote_xyz: [B, N, 3]
    refined_vote_features: [B, N, 288]
    vote_xyz: [B, M, 3]
    aligned_vote_feature: [B, M, 288]

    regard the refined_vote_xyz as the ground truth, find the nearest refined_vote for each vote_xyz,
    and calculate the mse loss between the aligned_vote_feature and corresponding refined_vote_features
    """
    global VOTE_DISTILL_WEIGHT
    # vote_mask_index = torch.argmax(vote_mask.half(), dim=1)  # [B]
    # valid_points = refined_vote_xyz[torch.arange(vote_mask_index.shape[0]), vote_mask_index, :]
    # valid_points = valid_points.unsqueeze(1).repeat(1, refined_vote_xyz.shape[1], 1) 
    # vote_xyz_mask = vote_mask.unsqueeze(-1).repeat(1, 1, 3)
    # refined_vote_xyz = torch.where(vote_xyz_mask, refined_vote_xyz, valid_points) # [B, N, 3]
    # vote_feature_mask = vote_mask.unsqueeze(-1).repeat(1, 1, 288)
    # valid_features = refined_vote_features[torch.arange(vote_mask_index.shape[0]), vote_mask_index, :]
    # valid_features = valid_features.unsqueeze(1).repeat(1, refined_vote_xyz.shape[1], 1)
    # refined_vote_features = torch.where(vote_feature_mask, refined_vote_features, valid_features)
    B, N, _ = refined_vote_xyz.shape
    # _, M, _ = vote_xyz.shape
    # dist = torch.sum(torch.abs(refined_vote_xyz.view(B, N, 1, 3) - vote_xyz.view(B, 1, M, 3)), dim=-1)  # [B, N, M]
    # _, min_dist_ind = torch.min(dist, dim=1)  # [B, M]
    # min_dist_ind = min_dist_ind.unsqueeze(-1).repeat(1, 1, 288)  # [B, M, 288]
    # # standardize the refined_vote_features and aligned_vote_feature in the feature dimension
    # refined_vote_features = refined_vote_features.div(torch.norm(refined_vote_features, dim=-1, keepdim=True, p=2))
    # aligned_vote_feature = aligned_vote_feature.div(torch.norm(aligned_vote_feature, dim=-1, keepdim=True, p=2))
    # refined_vote_features = torch.gather(refined_vote_features, dim=1, index=min_dist_ind)  # [B, M, 288]
    # distill_loss = huber_loss(refined_vote_features, aligned_vote_feature)
    # # diff = torch.abs(refined_vote_features - aligned_vote_feature) * spatial_attention_mask * channel_attention_mask
    # # l2_mask = diff < 1.
    # # distill_loss = 0
    # # distill_loss += torch.sum((diff ** 2) * l2_mask) / 2
    # # distill_loss += torch.sum((diff - 0.5) * ~l2_mask)
    # return distill_loss*VOTE_DISTILL_WEIGHT
    _, M, C = aligned_vote_feature.shape
    vote_xyz = vote_xyz.unsqueeze(2).repeat(1, 1, N, 1) # [B, M, N, 3]
    refined_vote_xyz = refined_vote_xyz.unsqueeze(1).repeat(1, M, 1, 1) # [B, M, N, 3]
    dist = torch.abs(vote_xyz - refined_vote_xyz)   # [B, M, N, 3]
    dist = torch.sum(dist ** 2, dim=-1)   # [B, M, N]
    min_dist_index = torch.argmin(dist, dim=-1) # [B, M]，每个vote对应一个距离其最近的refined_vote

    # refined_vote_features的维度是[B, N, C]，在refined_vote_features上索引[B, M], 得到每个点对应的feature[B, M, C]
    corresponding_features = torch.gather(refined_vote_features, 1, min_dist_index.unsqueeze(-1).repeat(1, 1, refined_vote_features.shape[-1]))
    # 每个vote对应一个refined_vote，这个refined_vote对应的mask权重，再repeat C次得到[B, M, C]
    corresponding_mask = torch.gather(vote_mask, 1, min_dist_index).unsqueeze(-1).repeat(1, 1, C)
    # 计算
    # distill_loss = torch.sum(((aligned_vote_feature - corresponding_features) * corresponding_mask) ** 2) / (B * M * C)
    corresponding_features = corresponding_features.div(torch.norm(corresponding_features, dim=-1, keepdim=True, p=2))
    aligned_vote_feature = aligned_vote_feature.div(torch.norm(aligned_vote_feature, dim=-1, keepdim=True, p=2))

    distill_loss = torch.sum(((aligned_vote_feature - corresponding_features) * corresponding_mask) ** 2) 
    return distill_loss / (B * M * C) * VOTE_DISTILL_WEIGHT

def compute_refine_loss(output_t, vote_mask, input):
    """ 
    vote_mask: [B, N]
    """
    GT_VOTE_FACTOR = 3
    batch_size, num_vote = vote_mask.shape
    aggregated_vote_inds = output_t['aggregated_vote_inds']  # TODO
    aggregated_seed_xyz = torch.gather(output_t['seed_xyz'], 1, aggregated_vote_inds[...,None].long().repeat(1,1,3))
    refined_xyz = output_t['transformer_weighted_xyz']
    gt_point_xyz = torch.gather(input['vote_label'], 1, output_t['seed_inds'].unsqueeze(-1).repeat(1,1,9).long())
    gt_vote_xyz = torch.gather(gt_point_xyz, 1, aggregated_vote_inds.unsqueeze(-1).repeat(1,1,9).long()) # [B, N, 3 * GT_NUM_VOTE]
    gt_vote_xyz += aggregated_seed_xyz.repeat(1,1,3)

    vote_xyz_reshape = refined_xyz.repeat(1,1,3).view(batch_size*num_vote, -1, 3)
    seed_gt_votes_reshape = gt_vote_xyz.view(batch_size*num_vote, GT_VOTE_FACTOR, 3)

    from nn_distance import nn_distance
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1)
    votes_dist = votes_dist.view(batch_size, num_vote)
    return torch.sum(votes_dist*vote_mask.float())/(torch.sum(vote_mask.float())+1e-6)
    # diff = torch.abs(refined_xyz - aggregated_seed_xyz - )
    # distill_loss = 0
    # distill_loss += torch.sum((diff ** 2) * vote_mask) / 2
    # distill_loss += torch.sum((diff - 0.5) * ~vote_mask)

    # # no vote loss
    # seed_dist = aggregated_seed_xyz - refined_xyz
    # from nn_distance import huber_loss
    # seed_dist = torch.mean(huber_loss(seed_dist, delta=1.0), dim=-1)
    # novote_mask = ~vote_mask
    # no_vote_loss = torch.sum(seed_dist*novote_mask.float()) / (torch.sum(novote_mask.float())+1e-6)
    # return no_vote_loss

def distillRunner(info, hyper_search=False):
    config = info['config']
    train_loader_iter = iter(info['traindataloader'])
    optimizer = info['optimizer']
    lr_scheduler = info['lr_scheduler']
    model = info['model']
    model_t = info['model_t']
    loggers = info['loggers']
    lowest_error = info['lowest_error']
    last_iter = info['last_iter']
    optimizer_t = info['teacher_optimizer']
    clip_grad_norm = config.get('clip_grad_norm', None)
    if clip_grad_norm is not None:
        print('CLIP GRAD NORM! MAX =', clip_grad_norm)
    t_start = time.time()
    T_START = time.time()
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
        # foreground_mask = get_seed_foreground_mask(output['seed_inds'], point_object_mask)
        vote_mask, percent = get_valid_vote_mask(refined_vote_xyz, input['center_label'], input['box_size'])
        output['percent_not_loss'] = percent
        output['vote_distill_loss'] = compute_vote_distill_loss(refined_vote_xyz, refined_vote_features, vote_xyz, aligned_vote_feature, vote_mask)
        output['refine_loss'] = compute_refine_loss(output_t, torch.gather(input['vote_label_mask'], 1, output_t['aggregated_vote_inds'].long()), input)
        loss += output['vote_distill_loss'] + output['refine_loss']
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
                    model_t.module.net.refine_module.eval()
                elif isinstance(model, torch.nn.Module):
                    model.val_mode()  # change mode
                    model_t.net.refine_module.eval()
                else:
                    raise NotImplementedError(type(model))
                output_error = {}
                error, weight, test_time = [], [], 0.
                for testset_name, loader in info['testdataloaders'].items():
                    _error, _weight, mAP25 = testmodel(model, loader, loggers, config.log_freq, testset_name, iter_id)
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
                    model_t.module.net.refine_module.train()
                elif isinstance(model, torch.nn.Module):
                    model.train_mode()  # change mode
                    model_t.net.refine_module.train()
                else:
                    raise NotImplementedError(type(model))
        lr_scheduler.step()
        loggers.update_loss(output, iter_id % config.log_freq == 0)  # TODO
        if hyper_search:
            session.report({'mAP': mAP25, 'loss': output['loss']})
    save_checkpoint({'teacher': model_t.state_dict()}, False, config.snapshot_save_path + '/teacher')
    print('training: done')

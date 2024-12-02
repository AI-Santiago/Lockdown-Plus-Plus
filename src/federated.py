import utils
import models
import math
import copy
import numpy as np
from agent import Agent
from agent_sparse import Agent as Agent_s
from options import args_parser
from aggregation import Aggregation
import torch
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
import logging
from sklearn.cluster import KMeans
import torch
import copy
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import torch
import wandb
from sklearn.cluster import SpectralClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.pairwise import cosine_distances

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    args = args_parser()
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    if not args.debug:
        logPath = "logs"
        if args.mask_init == "uniform":
            fileName = "uniformAckRatio{}_{}_Method{}_data{}_alpha{}_Epoch{}_inject{}_dense{}_Agg{}_same_mask{}_noniid{}_maskthreshold{}_attack{}_af{}.pt.pt".format(
                args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, args.local_ep, args.poison_frac,
                args.dense_ratio, args.aggr, args.same_mask, args.non_iid, args.theta, args.attack,args.anneal_factor)
        elif args.dis_check_gradient == True:
            fileName = "NoGradientAckRatio{}_{}_Method{}_data{}_alpha{}_Epoch{}_inject{}_dense{}_Agg{}_same_mask{}_noniid{}_maskthreshold{}_attack{}_af{}.pt.pt".format(
                args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, args.local_ep, args.poison_frac,
                args.dense_ratio, args.aggr, args.same_mask, args.non_iid, args.theta, args.attack,args.anneal_factor)
        else:
            fileName = "AckRatio{}_{}_Method{}_data{}_alpha{}_Epoch{}_inject{}_dense{}_Agg{}_same_mask{}_noniid{}_maskthreshold{}_attack{}_endpoison{}_af{}.pt".format(
                args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, args.local_ep, args.poison_frac,
                args.dense_ratio, args.aggr, args.same_mask, args.non_iid, args.theta, args.attack,
                args.cease_poison, args.anneal_factor)
        fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)
    logging.info(args)

    # Initialize wandb
    wandb.init(project="federated_learning", config=args)

    cum_poison_acc_mean = 0

    # load dataset and user groups (i.e., user to data mapping)
    train_dataset, val_dataset = utils.get_datasets(args.data)
    if args.data == "cifar100":
        num_target = 100
    elif args.data == "tinyimagenet":
        num_target = 200
    else:
        num_target = 10
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                            pin_memory=False)
    if args.non_iid:
        user_groups = utils.distribute_data_dirichlet(train_dataset, args)
    else:
        user_groups = utils.distribute_data(train_dataset, args, n_classes=num_target)
        # print(user_groups)
    idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
    # logging.info(idxs)
    if args.data != "tinyimagenet":
        # poison the validation dataset
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    else:
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args)

    poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                                     pin_memory=False)
    if args.data != "tinyimagenet":
        idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
        poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(poisoned_val_set_only_x.dataset, args, idxs, poison_all=True, modify_label=False)
    else:
        poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args, modify_label =False)

    poisoned_val_only_x_loader = DataLoader(poisoned_val_set_only_x, batch_size=args.bs, shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=False)

    # initialize a model, and the agents
    global_model = models.get_model(args.data).to(args.device)
    global_mask = {}
    neurotoxin_mask = {}
    updates_dict = {}
    n_model_params = len(parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]))
    params = {name: copy.deepcopy(global_model.state_dict()[name]) for name in global_model.state_dict()}
    if args.method == "lockdown":
        sparsity = utils.calculate_sparsities(args, params, distribution=args.mask_init)
        mask = utils.init_masks(params, sparsity)
    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_agents):
        if args.method == "lockdown":
            if args.same_mask==0:
                agent = Agent_s(_id, args, train_dataset, user_groups[_id], mask=utils.init_masks(params, sparsity))
            else:
                agent = Agent_s(_id, args, train_dataset, user_groups[_id], mask=mask)
        else:
            agent = Agent(_id, args, train_dataset, user_groups[_id])
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

        # aggregation server and the loss function

    aggregator = Aggregation(agent_data_sizes, n_model_params, poisoned_val_loader, args, None)
    criterion = nn.CrossEntropyLoss().to(args.device)
    agent_updates_list = []
    worker_id_list = []
    agent_updates_dict = {}
    mask_aggrement = []

    acc_vec = []
    asr_vec = []
    pacc_vec = []
    per_class_vec = []

    clean_asr_vec = []
    clean_acc_vec = []
    clean_pacc_vec = []
    clean_per_class_vec = []

    for rnd in range(1, args.rounds + 1): 
        logging.info("--------round {} ------------".format(rnd))
        # mask = torch.ones(n_model_params)
        rnd_global_params = parameters_to_vector([ copy.deepcopy(global_model.state_dict()[name]) for name in global_model.state_dict()])
        agent_updates_dict = {}
        chosen = np.random.choice(args.num_agents, math.floor(args.num_agents * args.agent_frac), replace=False)
        if args.method == "lockdown" or args.method == "fedimp":
            old_mask = [copy.deepcopy(agent.mask) for agent in agents]
        for agent_id in chosen:
            # logging.info(torch.sum(rnd_global_params))
            global_model = global_model.to(args.device)
            if args.method == "lockdown":
                update = agents[agent_id].local_train(global_model, criterion, rnd, global_mask=global_mask, neurotoxin_mask = neurotoxin_mask, updates_dict=updates_dict)
            else:
                update = agents[agent_id].local_train(global_model, criterion, rnd, neurotoxin_mask=neurotoxin_mask)
            agent_updates_dict[agent_id] = update
            utils.vector_to_model(copy.deepcopy(rnd_global_params), global_model)

        # aggregate params obtained by agents and update the global params
        if args.method == "lockdown":
            updates_dict,neurotoxin_mask = aggregator.aggregate_updates(global_model, agent_updates_dict)
        else:
            updates_dict,neurotoxin_mask = aggregator.aggregate_updates(global_model, agent_updates_dict)
        worker_id_list.append(agent_id + 1)



        # inference in every args.snap rounds
        if rnd % args.snap == 0:
            val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(global_model, criterion, val_loader,
                                                                                  args, rnd, num_target)
            logging.info(f'| Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
            logging.info(f'| Val_Per_Class_Acc: {val_per_class_acc} ')
            acc_vec.append(val_acc)
            per_class_vec.append(val_per_class_acc)


            poison_loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(global_model, criterion,
                                                                            poisoned_val_loader, args, rnd, num_target)
            cum_poison_acc_mean += asr
            asr_vec.append(asr)
            logging.info(f'| Attack Loss/Attack Success Ratio: {poison_loss:.3f} / {asr:.3f} |')

            poison_loss, (poison_acc, _), fail_samples = utils.get_loss_n_accuracy(global_model, criterion,
                                                                                   poisoned_val_only_x_loader, args,
                                                                                   rnd, num_target)
            pacc_vec.append(poison_acc)
            logging.info(f'| Poison Loss/Poison accuracy: {poison_loss:.3f} / {poison_acc:.3f} |')

            if args.method == "lockdown" :
                test_model = copy.deepcopy(global_model) 
                for name, param in test_model.named_parameters():
                    mask = 0 ######Actually it still turns into a tensor later on
                    # print('done')
                    for id, agent in enumerate(agents):
                        mask += old_mask[id][name].to(args.device) 
                        # print('old_mask',id, agent, old_mask[id][name].shape)
                        # print('mask',mask.shape) #torch.Size([64])
                        # break

                    param.data = torch.where(mask.to(args.device) >= args.theta, param,
                                             torch.zeros_like(param))   
                    logging.info(torch.sum(mask.to(args.device) >= args.theta) / torch.numel(mask)) ###### mostly 1 and generally 0.6-0.8 






            ######The following code is trying to cluster based on mask, and exp shows they are useless
            # if args.method == "lockdown":
            #     test_model = copy.deepcopy(global_model)
            #     for name, param in test_model.named_parameters():
            #         masks = []
            #         backup=0
            #         for id, agent in enumerate(agents):
            #             mask = old_mask[id][name].to(args.device)
            #             masks.append(mask.flatten())
            #             backup += old_mask[id][name].to(args.device)
            #         masks_tensor = torch.stack(masks) ###### Shape would be (num_clients, num_elements)
            #         num_top = 16
            #         processed_masks = []
            #         for client_mask in masks_tensor:
            #             top_values, top_indices = torch.topk(client_mask, num_top)
            #             new_mask = torch.zeros_like(client_mask)
            #             new_mask[top_indices] = top_values
            #             processed_masks.append(new_mask)
                        
            #         processed_masks_tensor = torch.stack(processed_masks)
            #         masks_np = processed_masks_tensor.cpu().numpy()


            #         ###### Jaccard 
            #         # distance_matrix = pairwise_distances(masks_np, metric='jaccard')
            #         ###### Use spectral clustering, good in theory, in practice you get disconnected graphs and it's too slow
            #         # num_clusters = 2  
            #         # spectral_clustering = SpectralClustering(
            #         #     n_clusters=num_clusters,
            #         #     affinity='precomputed',
            #         #     random_state=0
            #         # )
            #         # labels = spectral_clustering.fit_predict(distance_matrix)


            #         ###### K-Means 
            #         num_clusters = 2  
            #         # kmeans = KMeans(n_clusters=num_clusters, random_state=42, init='k-means++')
            #         # labels = kmeans.fit_predict(masks_np)  
            #         distance_matrix = cosine_distances(masks_np)

            #         # Initialize KMedoids with the desired number of clusters and precomputed distance matrix
            #         kmedoids = KMedoids(n_clusters=num_clusters, metric='precomputed', random_state=42, init='k-medoids++')

            #         # Fit KMedoids on the distance matrix and predict the cluster labels
            #         labels = kmedoids.fit_predict(distance_matrix)
            #         unique_labels, counts = np.unique(labels, return_counts=True)

            #         if len(unique_labels) == 1:
            #                 param.data = torch.where(backup.to(args.device) >= args.theta, param,
            #                                         torch.zeros_like(param)) 
            #                 logging.info(torch.sum(mask.to(args.device) >= args.theta) / torch.numel(mask)) 

            #         else:    

            #             smallest_cluster_label = unique_labels[np.argmin(counts)]
            #             other_cluster_label = unique_labels[np.argmax(counts)]  
            #             positions_in_smallest_cluster = np.where(labels == smallest_cluster_label)[0]
            #             positions_in_other_cluster = np.where(labels == other_cluster_label)[0]
            #             print('positions_in_smallest_cluster',positions_in_smallest_cluster)
            #             # print('positions_in_other_cluster',positions_in_other_cluster)
            #             # break
            #             masks_in_smallest_cluster = masks_np[positions_in_smallest_cluster]
            #             ###### See if at least one client is set to 1 #
            #             # mask_sums_small = np.any(masks_in_smallest_cluster != 0, axis=0).astype(int)
            #             masks_in_other_cluster = masks_np[labels == other_cluster_label]
            #             # mask_sums_other = np.any(masks_in_other_cluster != 0, axis=0).astype(int)
            #             mask_sums_small = np.sum(masks_in_smallest_cluster != 0, axis=0)
            #             mask_sums_other = np.sum(masks_in_other_cluster != 0, axis=0)
            #             positions_to_zero = np.where(mask_sums_small !=0)[0]
            #             param_data_flat = param.data.view(-1)
            #             param_data_flat[positions_to_zero] = 0
            #             param.data = param_data_flat.view(param.data.shape)    
            #             logging.info(
            #             f"{torch.sum(param.data == 0).item() / param.data.numel()}"
            #             )








                val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(test_model, criterion,
                                                                                      val_loader,
                                                                                      args, rnd, num_target)
                # writer.add_scalar('Clean Validation/Loss', val_loss, rnd)
                # writer.add_scalar('Clean Validation/Accuracy', val_acc, rnd)
                logging.info(f'| Clean Val_Loss/Val_Acc: {val_loss:.3f} / {val_acc:.3f} |')
                logging.info(f'| Clean Val_Per_Class_Acc: {val_per_class_acc} ')
                clean_acc_vec.append(val_acc)
                clean_per_class_vec.append(val_per_class_acc)

                poison_loss, (poison_acc, _), _ = utils.get_loss_n_accuracy(test_model, criterion,
                                                                            poisoned_val_loader,
                                                                            args, rnd, num_target)
                clean_asr_vec.append(poison_acc)
                cum_poison_acc_mean += poison_acc
                logging.info(f'| Clean Attack Success Ratio: {poison_loss:.3f} / {poison_acc:.3f} |')


                wandb.log({
                    "round": rnd,
                    "clean_val_loss": val_loss,
                    "clean_val_acc": val_acc,
                    "clean_val_per_class_acc": val_per_class_acc,
                    "clean_attack_loss": poison_loss,
                    "clean_attack_success_ratio": poison_acc})

                poison_loss, (poison_acc, _), fail_samples = utils.get_loss_n_accuracy(test_model, criterion,
                                                                                       poisoned_val_only_x_loader, args,
                                                                                       rnd, num_target)
                clean_pacc_vec.append(poison_acc)
                logging.info(f'| Clean Poison Loss/Clean Poison accuracy: {poison_loss:.3f} / {poison_acc:.3f} |')

                wandb.log({
                    "clean_poison_loss": poison_loss,
                    "clean_poison_accuracy": poison_acc,
                    })
                # ask the guys to finetune the classifier







                del test_model

        save_frequency = 25
        if rnd % save_frequency == 0:
            if args.mask_init == "uniform":
                PATH = "checkpoint/uniform_AckRatio{}_{}_Method{}_data{}_alpha{}_Rnd{}_Epoch{}_inject{}_dense{}_Agg{}_se_threshold{}_noniid{}_maskthreshold{}_attack{}.pt".format(
                    args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, rnd, args.local_ep,
                    args.poison_frac, args.dense_ratio, args.aggr, args.se_threshold, args.non_iid,
                    args.theta, args.attack)
            elif args.dis_check_gradient == True:
                PATH = "checkpoint/NoGradient_AckRatio{}_{}_Method{}_data{}_alpha{}_Rnd{}_Epoch{}_inject{}_dense{}_Agg{}_se_threshold{}_noniid{}_maskthreshold{}_attack{}.pt".format(
                    args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, rnd, args.local_ep,
                    args.poison_frac, args.dense_ratio, args.aggr, args.se_threshold, args.non_iid,
                    args.theta, args.attack)
            else:
                PATH = "checkpoint/AckRatio{}_{}_Method{}_data{}_alpha{}_Rnd{}_Epoch{}_inject{}_dense{}_Agg{}_se_threshold{}_noniid{}_maskthreshold{}_attack{}.pt".format(
                    args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, rnd, args.local_ep,
                    args.poison_frac, args.dense_ratio, args.aggr, args.se_threshold, args.non_iid,
                    args.theta, args.attack)
            if args.method == "lockdown" or args.method == "fedimp":
                torch.save({
                    'option': args,
                    'model_state_dict': global_model.state_dict(),
                    'masks': [agent.mask for agent in agents],
                    'acc_vec': acc_vec,
                    "asr_vec": asr_vec,
                    'pacc_vec ': pacc_vec,
                    "per_class_vec": per_class_vec,
                    "clean_asr_vec": clean_asr_vec,
                    'clean_acc_vec': clean_acc_vec,
                    'clean_pacc_vec ': clean_pacc_vec,
                    'clean_per_class_vec': clean_per_class_vec,
                }, PATH)
            else:
                torch.save({
                    'option': args,
                    'model_state_dict': global_model.state_dict(),
                    'acc_vec': acc_vec,
                    "asr_vec": asr_vec,
                    'pacc_vec ': pacc_vec,
                    "per_class_vec": per_class_vec,
                    'neurotoxin_mask': neurotoxin_mask
                }, PATH)

    logging.info('Training has finished!')

    # Finish wandb run
    wandb.finish()

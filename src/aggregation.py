import copy

import torch
import models
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import numpy as np
from copy import deepcopy
from torch.nn import functional as F
import logging
from utils import name_param_to_array,  vector_to_model, vector_to_name_param
from sklearn.cluster import SpectralClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans

class Aggregation():
    def __init__(self, agent_data_sizes, n_params, poisoned_val_loader, args, writer):
        self.agent_data_sizes = agent_data_sizes
        self.args = args
        self.writer = writer
        self.server_lr = args.server_lr
        self.n_params = n_params
        self.poisoned_val_loader = poisoned_val_loader
        self.cum_net_mov = 0
        
         
    def aggregate_updates(self, global_model, agent_updates_dict):
        lr_vector = torch.Tensor([self.server_lr]*self.n_params).to(self.args.device)
        if self.args.method != "rlr":
            lr_vector=lr_vector
        else:
            lr_vector, _ = self.compute_robustLR(agent_updates_dict)
        # mask = torch.ones_like(agent_updates_dict[0])
        aggregated_updates = 0
        cur_global_params = parameters_to_vector(
            [global_model.state_dict()[name] for name in global_model.state_dict()]).detach()
        if self.args.aggr=='avg':          
            aggregated_updates = self.agg_avg(agent_updates_dict) 
        if self.args.aggr== "clip_avg":
            for _id, update in agent_updates_dict.items():
                weight_diff_norm = torch.norm(update).item()
                logging.info(weight_diff_norm)
                update.data = update.data / max(1, weight_diff_norm / 2)
            aggregated_updates = self.agg_avg(agent_updates_dict)
            logging.info(torch.norm(aggregated_updates))
        elif self.args.aggr=='comed':
            aggregated_updates = self.agg_comed(agent_updates_dict)
        elif self.args.aggr == 'sign':
            aggregated_updates = self.agg_sign(agent_updates_dict)
        elif self.args.aggr == "krum":
            aggregated_updates = self.agg_krum(agent_updates_dict)
        elif self.args.aggr == "gm":
            aggregated_updates = self.agg_gm(agent_updates_dict,cur_global_params)
        elif self.args.aggr == "tm":
            aggregated_updates = self.agg_tm(agent_updates_dict)
        neurotoxin_mask = {} 
        updates_dict = vector_to_name_param(aggregated_updates, copy.deepcopy(global_model.state_dict())) ###### Convert the aggregated updated vectors into a named parameter dictionary and make a deep copy of the global model's state dictionary.
        for name in updates_dict: ###### processes the absolute value of each update to create a mask that identifies the significant portion of the update.
            updates = updates_dict[name].abs().view(-1)
            gradients_length = torch.numel(updates)
            _, indices = torch.topk(-1 * updates, int(gradients_length * self.args.dense_ratio))
            mask_flat = torch.zeros(gradients_length)
            mask_flat[indices.cpu()] = 1
            neurotoxin_mask[name] = (mask_flat.reshape(updates_dict[name].size()))

        cur_global_params = parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]).detach() 
        # print('cur_global_params', cur_global_params.type, cur_global_params.shape, cur_global_params)
        new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float() ###### Calculate new global parameters, add current parameters to aggregation update
        vector_to_model(new_global_params, global_model) ###### update
        return    updates_dict, neurotoxin_mask 


    def compute_robustLR(self, agent_updates_dict):

        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]  
        sm_of_signs = torch.abs(sum(agent_updates_sign))
        mask=torch.zeros_like(sm_of_signs)
        mask[sm_of_signs < self.args.theta] = 0
        mask[sm_of_signs >= self.args.theta] = 1
        sm_of_signs[sm_of_signs < self.args.theta] = -self.server_lr
        sm_of_signs[sm_of_signs >= self.args.theta] = self.server_lr
        return sm_of_signs.to(self.args.device), mask


    def agg_krum(self, agent_updates_dict):
        krum_param_m = 1
        def _compute_krum_score( vec_grad_list, byzantine_client_num):
            krum_scores = []
            num_client = len(vec_grad_list)
            for i in range(0, num_client):
                dists = []
                for j in range(0, num_client):
                    if i != j:
                        dists.append(
                            torch.norm(vec_grad_list[i]- vec_grad_list[j])
                            .item() ** 2
                        )
                dists.sort()  # ascending
                score = dists[0: num_client - byzantine_client_num - 2]
                krum_scores.append(sum(score))
            return krum_scores

        # Compute list of scores
        __nbworkers = len(agent_updates_dict)
        krum_scores = _compute_krum_score(agent_updates_dict, self.args.num_corrupt)
        score_index = torch.argsort(
            torch.Tensor(krum_scores)
        ).tolist()  # indices; ascending
        score_index = score_index[0: krum_param_m]
        return_gradient = [agent_updates_dict[i] for i in score_index]
        return sum(return_gradient)/len(return_gradient)

    def agg_avg(self, agent_updates_dict):
        """ classic fed avg """

        ###### All client parameters
        all_updates = []
        for _id, update in agent_updates_dict.items():
            all_updates.append(update.view(-1).cpu().numpy())
        all_updates = np.array(all_updates)

        ###### KMeans clustering with 2 clusters. take topk, switching distance function is not useful.
        kmeans = KMeans(n_clusters=2, random_state=42, init='k-means++')
        labels = kmeans.fit_predict(all_updates)

        ###### Labels for the smallest clusters
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) == 2:
            smallest_cluster_label = unique_labels[np.argmin(counts)]

            ###### number of clients in the smallest cluster
            num_clients_in_smallest_cluster = counts[np.argmin(counts)]
            print(f"Number of clients in the smallest cluster: {num_clients_in_smallest_cluster}")

            ###### Index of parameter positions belonging to the smallest group
            positions_in_smallest_cluster = np.where(labels == smallest_cluster_label)[0]

            ###### Calculate the sum of the smallest group's n_agent_data for later average calculation
            n_agent_data_smallest_cluster = sum(self.agent_data_sizes[list(agent_updates_dict.keys())[idx]] for idx in positions_in_smallest_cluster)

            ###### Setting the parameters of these positions to zero, they are not involved in aggregation
            for idx in positions_in_smallest_cluster:
                update = agent_updates_dict[list(agent_updates_dict.keys())[idx]]
                update_flat = update.view(-1)
                update_flat[:] = 0
                update.data = update_flat.view(update.data.shape)

        ###### still FedAvg 
        sm_updates, total_data = 0, 0
        for _id, update in agent_updates_dict.items():
            n_agent_data = self.agent_data_sizes[_id]
            sm_updates += n_agent_data * update
            total_data += n_agent_data

        ###### Note this
        total_data -= n_agent_data_smallest_cluster

        return sm_updates / total_data
    
    def agg_comed(self, agent_updates_dict):
        agent_updates_col_vector = [update.view(-1, 1) for update in agent_updates_dict.values()]
        concat_col_vectors = torch.cat(agent_updates_col_vector, dim=1)
        return torch.median(concat_col_vectors, dim=1).values
    
    def agg_sign(self, agent_updates_dict):
        """ aggregated majority sign update """
        agent_updates_sign = [torch.sign(update) for update in agent_updates_dict.values()]
        sm_signs = torch.sign(sum(agent_updates_sign))
        return torch.sign(sm_signs)




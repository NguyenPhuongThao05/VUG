r"""
VUG_BiTGCF
################################################
Reference:
    Combining VUG (Virtual User Generation) with BiTGCF (Bi-partite Graph Convolutional Framework)
    for enhanced cross-domain recommendation with virtual user embeddings and graph convolution.
    
Description:
    This model combines:
    1. VUG's virtual user generation capability using attention mechanisms
    2. BiTGCF's graph convolutional networks and bidirectional transfer learning
    
    Key innovations:
    - Uses VUG generator to create virtual embeddings for target-only users
    - Employs BiTGCF's GCN layers with transfer learning between domains
    - Enhanced user representation through virtual user augmentation in graph space
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import EmbLoss
from recbole.utils import InputType
from .attention import CDUserItemAttention


class VUG_BiTGCF(CrossDomainRecommender):
    r"""VUG_BiTGCF combines virtual user generation with bi-directional graph convolution.
    
    It uses:
    - A dual-attention generator (from VUG) to create virtual source embeddings
    - Graph convolutional layers (from BiTGCF) for feature propagation
    - Bidirectional transfer learning between source and target domains
    - Enhanced training with virtual user augmentation for non-overlapping users
    """
    
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(VUG_BiTGCF, self).__init__(config, dataset)
        
        # Load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field
        self.device = config['device']

        # Load BiTGCF parameters
        self.latent_dim = config['embedding_size']  # embedding dimension
        self.n_layers = config['n_layers']  # number of GCN layers
        self.reg_weight = config['reg_weight']  # regularization weight
        self.domain_lambda_source = config['lambda_source']  # source transfer weight
        self.domain_lambda_target = config['lambda_target']  # target transfer weight
        self.drop_rate = config['drop_rate']  # dropout rate
        self.connect_way = config['connect_way']  # layer aggregation method
        self.is_transfer = config['is_transfer']  # enable transfer learning
        
        # Load VUG generator parameters
        self.gen_weight = config['gen_weight']  # weight for generator loss
        self.enhance_mode = config['enhance_mode']  # how to use generated embeddings
        self.enhance_weight = config['enhance_weight']  # weight for enhancement
        self.user_weight_attn = config['user_weight_attn']  # attention weight
        self.use_virtual = config['use_virtual']  # whether to use virtual embeddings
        
        # Define BiTGCF embeddings (separate for source and target)
        self.source_user_embedding = nn.Embedding(self.total_num_users, self.latent_dim)
        self.target_user_embedding = nn.Embedding(self.total_num_users, self.latent_dim)
        self.source_item_embedding = nn.Embedding(self.total_num_items, self.latent_dim)
        self.target_item_embedding = nn.Embedding(self.total_num_items, self.latent_dim)
        
        # Store overlapped counts for use in initialization and transfer
        self.overlapped_num_users = self.num_overlap_user  
        self.overlapped_num_items = self.num_overlap_item
        
        # Initialize embeddings for non-overlapping entities as zeros (BiTGCF style)
        with torch.no_grad():
            self.source_user_embedding.weight[self.overlapped_num_users: self.target_num_users].fill_(0)
            self.source_item_embedding.weight[self.overlapped_num_items: self.target_num_items].fill_(0)
            self.target_user_embedding.weight[self.target_num_users:].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)
        
        # Define VUG generator for creating virtual source embeddings
        self.source_generator = CDUserItemAttention(self.latent_dim)
        
        # Loss functions and layers
        self.dropout = nn.Dropout(p=self.drop_rate)
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()
        self.reg_loss = EmbLoss()
        self.gen_loss = nn.MSELoss()
        
        # Generate interaction matrices and adjacency matrices
        self.source_interaction_matrix = dataset.inter_matrix(
            form='coo', value_field=None, domain='source'
        ).astype(np.float32)
        self.target_interaction_matrix = dataset.inter_matrix(
            form='coo', value_field=None, domain='target'
        ).astype(np.float32)
        
        self.source_norm_adj_matrix = self.get_norm_adj_mat(
            self.source_interaction_matrix, self.total_num_users, self.total_num_items
        ).to(self.device)
        self.target_norm_adj_matrix = self.get_norm_adj_mat(
            self.target_interaction_matrix, self.total_num_users, self.total_num_items
        ).to(self.device)
        
        # Degree counts for transfer learning
        self.source_user_degree_count = torch.from_numpy(
            self.source_interaction_matrix.sum(axis=1)
        ).to(self.device)
        self.target_user_degree_count = torch.from_numpy(
            self.target_interaction_matrix.sum(axis=1)
        ).to(self.device)
        self.source_item_degree_count = torch.from_numpy(
            self.source_interaction_matrix.sum(axis=0)
        ).transpose(0, 1).to(self.device)
        self.target_item_degree_count = torch.from_numpy(
            self.target_interaction_matrix.sum(axis=0)
        ).transpose(0, 1).to(self.device)
        
        # Build mask for target-only users
        self.target_nonoverlap_mask = torch.zeros(
            (self.target_user_embedding.weight.size(0), 1),
            dtype=torch.bool,
            device=self.device
        )
        self.target_nonoverlap_mask[
            self.num_overlap_user : self.num_overlap_user + self.num_target_only_user
        ] = True
        self.target_nonoverlap_mask = self.target_nonoverlap_mask.expand(
            -1, self.latent_dim
        ).to(self.target_user_embedding.weight.dtype)
        
        # Storage for cached embeddings
        self.generated_source_emb_cache = None
        self.target_restore_user_e = None
        self.target_restore_item_e = None
        
        # Training phase
        self.phase = 'BOTH'
        
        # Parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['target_restore_user_e', 'target_restore_item_e']

    def set_phase(self, phase):
        """Set training phase (BOTH/TARGET)"""
        self.phase = phase

    def get_norm_adj_mat(self, interaction_matrix, n_users=None, n_items=None):
        r"""Build normalized adjacency matrix for graph convolution."""
        if n_users is None or n_items is None:
            n_users, n_items = interaction_matrix.shape
            
        # Create bipartite adjacency matrix
        A = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        
        # User-item and item-user connections
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        
        # Normalize adjacency matrix
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        
        # Convert to sparse tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self, domain='source', use_enhancement=True):
        r"""Get initial embeddings with optional virtual enhancement."""
        if domain == 'source':
            user_embeddings = self.source_user_embedding.weight
            item_embeddings = self.source_item_embedding.weight
            norm_adj_matrix = self.source_norm_adj_matrix
        else:
            user_embeddings = self.target_user_embedding.weight
            item_embeddings = self.target_item_embedding.weight
            norm_adj_matrix = self.target_norm_adj_matrix
            
            # Apply virtual enhancement for target domain
            if (use_enhancement and self.use_virtual and self.training):
                generated_emb = self._get_or_generate_virtual_embeddings()
                
                if self.enhance_mode == "add":
                    user_embeddings = user_embeddings + self.enhance_weight * self.target_nonoverlap_mask * generated_emb
                elif self.enhance_mode == "replace":
                    user_embeddings = torch.where(self.target_nonoverlap_mask, generated_emb, user_embeddings)
        
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings, norm_adj_matrix

    def _get_or_generate_virtual_embeddings(self):
        r"""Generate or retrieve cached virtual source embeddings for all users."""
        if self.generated_source_emb_cache is None:
            _, generated_emb, _, _ = self.generate_source_embeddings()
            self.generated_source_emb_cache = generated_emb
        return self.generated_source_emb_cache
    
    def clear_virtual_cache(self):
        r"""Clear cached virtual embeddings."""
        self.generated_source_emb_cache = None

    def graph_layer(self, adj_matrix, all_embeddings):
        r"""Graph convolutional layer (BiTGCF style)."""
        side_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
        new_embeddings = side_embeddings + torch.mul(all_embeddings, side_embeddings)
        new_embeddings = all_embeddings + new_embeddings
        new_embeddings = self.dropout(new_embeddings)
        return new_embeddings

    def transfer_layer(self, source_all_embeddings, target_all_embeddings):
        r"""Bidirectional transfer learning layer (BiTGCF style)."""
        source_user_embeddings, source_item_embeddings = torch.split(
            source_all_embeddings, [self.total_num_users, self.total_num_items]
        )
        target_user_embeddings, target_item_embeddings = torch.split(
            target_all_embeddings, [self.total_num_users, self.total_num_items]
        )

        # Lambda-weighted transfer
        source_user_embeddings_lam = (
            self.domain_lambda_source * source_user_embeddings + 
            (1 - self.domain_lambda_source) * target_user_embeddings
        )
        target_user_embeddings_lam = (
            self.domain_lambda_target * target_user_embeddings + 
            (1 - self.domain_lambda_target) * source_user_embeddings
        )
        source_item_embeddings_lam = (
            self.domain_lambda_source * source_item_embeddings + 
            (1 - self.domain_lambda_source) * target_item_embeddings
        )
        target_item_embeddings_lam = (
            self.domain_lambda_target * target_item_embeddings + 
            (1 - self.domain_lambda_target) * source_item_embeddings
        )

        # Laplacian-weighted transfer
        source_user_laplace = self.source_user_degree_count
        target_user_laplace = self.target_user_degree_count
        user_laplace = source_user_laplace + target_user_laplace + 1e-7
        source_user_embeddings_lap = (
            (source_user_laplace * source_user_embeddings + 
             target_user_laplace * target_user_embeddings) / user_laplace
        )
        target_user_embeddings_lap = source_user_embeddings_lap
        
        source_item_laplace = self.source_item_degree_count
        target_item_laplace = self.target_item_degree_count
        item_laplace = source_item_laplace + target_item_laplace + 1e-7
        source_item_embeddings_lap = (
            (source_item_laplace * source_item_embeddings + 
             target_item_laplace * target_item_embeddings) / item_laplace
        )
        target_item_embeddings_lap = source_item_embeddings_lap

        # Combine overlapping and specific embeddings
        source_specific_user_embeddings = source_user_embeddings[self.overlapped_num_users:]
        target_specific_user_embeddings = target_user_embeddings[self.overlapped_num_users:]
        source_specific_item_embeddings = source_item_embeddings[self.overlapped_num_items:]
        target_specific_item_embeddings = target_item_embeddings[self.overlapped_num_items:]
        
        source_overlap_user_embeddings = (
            source_user_embeddings_lam[:self.overlapped_num_users] + 
            source_user_embeddings_lap[:self.overlapped_num_users]
        ) / 2
        target_overlap_user_embeddings = (
            target_user_embeddings_lam[:self.overlapped_num_users] + 
            target_user_embeddings_lap[:self.overlapped_num_users]
        ) / 2
        source_overlap_item_embeddings = (
            source_item_embeddings_lam[:self.overlapped_num_items] + 
            source_item_embeddings_lap[:self.overlapped_num_items]
        ) / 2
        target_overlap_item_embeddings = (
            target_item_embeddings_lam[:self.overlapped_num_items] + 
            target_item_embeddings_lap[:self.overlapped_num_items]
        ) / 2
        
        # Reconstruct full embeddings
        source_transfer_user_embeddings = torch.cat(
            [source_overlap_user_embeddings, source_specific_user_embeddings], dim=0
        )
        target_transfer_user_embeddings = torch.cat(
            [target_overlap_user_embeddings, target_specific_user_embeddings], dim=0
        )
        source_transfer_item_embeddings = torch.cat(
            [source_overlap_item_embeddings, source_specific_item_embeddings], dim=0
        )
        target_transfer_item_embeddings = torch.cat(
            [target_overlap_item_embeddings, target_specific_item_embeddings], dim=0
        )

        source_alltransfer_embeddings = torch.cat(
            [source_transfer_user_embeddings, source_transfer_item_embeddings], dim=0
        )
        target_alltransfer_embeddings = torch.cat(
            [target_transfer_user_embeddings, target_transfer_item_embeddings], dim=0
        )
        
        return source_alltransfer_embeddings, target_alltransfer_embeddings

    def forward(self):
        r"""Forward pass through graph convolution layers."""
        # Clear cache at start
        self.clear_virtual_cache()
        
        # Get initial embeddings
        source_all_embeddings, source_norm_adj_matrix = self.get_ego_embeddings(
            domain='source', use_enhancement=False
        )
        target_all_embeddings, target_norm_adj_matrix = self.get_ego_embeddings(
            domain='target', use_enhancement=True
        )

        # Store embeddings from each layer
        source_embeddings_list = [source_all_embeddings]
        target_embeddings_list = [target_all_embeddings]
        
        # Graph convolution layers
        for layer_idx in range(self.n_layers):
            source_all_embeddings = self.graph_layer(source_norm_adj_matrix, source_all_embeddings)
            target_all_embeddings = self.graph_layer(target_norm_adj_matrix, target_all_embeddings)

            # Transfer learning when in BOTH phase
            if self.phase == 'BOTH' and self.is_transfer:
                source_all_embeddings, target_all_embeddings = self.transfer_layer(
                    source_all_embeddings, target_all_embeddings
                )

            # Normalize embeddings
            source_norm_embeddings = nn.functional.normalize(source_all_embeddings, p=2, dim=1)
            target_norm_embeddings = nn.functional.normalize(target_all_embeddings, p=2, dim=1)
            source_embeddings_list.append(source_norm_embeddings)
            target_embeddings_list.append(target_norm_embeddings)

        # Aggregate embeddings from all layers
        if self.connect_way == 'concat':
            source_final_embeddings = torch.cat(source_embeddings_list, 1)
            target_final_embeddings = torch.cat(target_embeddings_list, 1)
        elif self.connect_way == 'mean':
            source_final_embeddings = torch.stack(source_embeddings_list, dim=1)
            source_final_embeddings = torch.mean(source_final_embeddings, dim=1)
            target_final_embeddings = torch.stack(target_embeddings_list, dim=1)
            target_final_embeddings = torch.mean(target_final_embeddings, dim=1)

        # Split user and item embeddings
        source_user_all_embeddings, source_item_all_embeddings = torch.split(
            source_final_embeddings, [self.total_num_users, self.total_num_items]
        )
        target_user_all_embeddings, target_item_all_embeddings = torch.split(
            target_final_embeddings, [self.total_num_users, self.total_num_items]
        )

        return (source_user_all_embeddings, source_item_all_embeddings, 
                target_user_all_embeddings, target_item_all_embeddings)

    def calculate_loss(self, interaction):
        r"""Calculate main loss (BiTGCF only) - generator loss handled separately."""
        
        if self.phase == 'BOTH':
            return self.calculate_joint_loss(interaction)
        elif self.phase == 'TARGET':
            return self.calculate_target_loss(interaction)
        else:
            raise NotImplementedError(f"Phase {self.phase} not implemented")

    def calculate_joint_loss(self, interaction):
        r"""Calculate joint loss for both domains."""
        # Get embeddings
        (source_user_all_embeddings, source_item_all_embeddings, 
         target_user_all_embeddings, target_item_all_embeddings) = self.forward()

        # Extract interaction data
        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]
        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        # Get user and item embeddings for current batch
        source_u_embeddings = source_user_all_embeddings[source_user]
        source_i_embeddings = source_item_all_embeddings[source_item]
        target_u_embeddings = target_user_all_embeddings[target_user]
        target_i_embeddings = target_item_all_embeddings[target_item]

        # Calculate BCE losses
        source_output = self.sigmoid(torch.mul(source_u_embeddings, source_i_embeddings).sum(dim=1))
        source_bce_loss = self.bce_loss(source_output, source_label)

        target_output = self.sigmoid(torch.mul(target_u_embeddings, target_i_embeddings).sum(dim=1))
        target_bce_loss = self.bce_loss(target_output, target_label)

        # Calculate regularization losses
        u_ego_embeddings_s = self.source_user_embedding(source_user)
        i_ego_embeddings_s = self.source_item_embedding(source_item)
        source_reg_loss = self.reg_loss(u_ego_embeddings_s, i_ego_embeddings_s)

        u_ego_embeddings_t = self.target_user_embedding(target_user)
        i_ego_embeddings_t = self.target_item_embedding(target_item)
        target_reg_loss = self.reg_loss(u_ego_embeddings_t, i_ego_embeddings_t)

        # Combined loss
        total_loss = (
            source_bce_loss + self.reg_weight * source_reg_loss +
            target_bce_loss + self.reg_weight * target_reg_loss
        )

        return total_loss

    def calculate_target_loss(self, interaction):
        r"""Calculate loss for target domain only."""
        (_, _, target_user_all_embeddings, target_item_all_embeddings) = self.forward()

        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        target_u_embeddings = target_user_all_embeddings[target_user]
        target_i_embeddings = target_item_all_embeddings[target_item]

        target_output = self.sigmoid(torch.mul(target_u_embeddings, target_i_embeddings).sum(dim=1))
        target_bce_loss = self.bce_loss(target_output, target_label)

        u_ego_embeddings = self.target_user_embedding(target_user)
        i_ego_embeddings = self.target_item_embedding(target_item)
        target_reg_loss = self.reg_loss(u_ego_embeddings, i_ego_embeddings)

        total_loss = target_bce_loss + self.reg_weight * target_reg_loss
        return total_loss

    def calculate_gen_loss(self, interaction):
        r"""Calculate generator loss for virtual user generation."""
        
        if not self.use_virtual:
            dummy_loss = 0.0 * sum(p.sum() for p in self.source_generator.parameters())
            return dummy_loss
        
        user, generated_s_emb, overlap_indices, non_overlap_indices = \
            self.generate_source_embeddings(interaction)

        if overlap_indices.shape[0] == 0:
            reg_loss = 0.0001 * sum(p.pow(2).sum() for p in self.source_generator.parameters())
            return reg_loss

        # Real source embeddings for overlapping users
        real_source_emb = self.source_user_embedding(user[overlap_indices])
        gen_for_overlapped = generated_s_emb[overlap_indices]

        # MSE loss + regularization
        mse_loss = self.gen_loss(gen_for_overlapped, real_source_emb)
        reg_loss = 0.0001 * sum(p.pow(2).sum() for p in self.source_generator.parameters())
        
        total_gen_loss = self.gen_weight * mse_loss + reg_loss
        return total_gen_loss

    def generate_source_embeddings(self, interaction=None):
        r"""Generate virtual source embeddings using dual-attention mechanism."""
        
        if interaction is None:
            user = torch.arange(self.total_num_users, device=self.target_user_embedding.weight.device)
        else:
            user = interaction[self.TARGET_USER_ID]

        # Get target user embeddings as queries
        user_t_emb = self.target_user_embedding(user)
        Q_item = user_t_emb  # Simplified
        
        # Identify overlapping users
        overlapped_mask = (user < self.overlapped_num_users)
        overlap_indices = torch.nonzero(overlapped_mask).flatten()
        non_overlap_indices = torch.nonzero(~overlapped_mask).flatten()

        if overlap_indices.shape[0] == 0:
            generated_s_emb = torch.zeros_like(user_t_emb)
            dummy_gen = 0.0 * sum(p.sum() for p in self.source_generator.parameters())
            generated_s_emb = generated_s_emb + dummy_gen
            return user, generated_s_emb, overlap_indices, non_overlap_indices

        # Build keys and values from overlapping users
        overlapped_users = user[overlap_indices]
        K_user = self.target_user_embedding(overlapped_users)
        K_item = K_user
        V_user = self.source_user_embedding(overlapped_users)

        # Apply dual-attention generator
        generated_s_emb, attn_weights = self.source_generator(
            Q_user=user_t_emb,
            K_user=K_user,
            V_user=V_user,
            Q_item=Q_item,
            K_item=K_item,
            alpha=self.user_weight_attn
        )

        return user, generated_s_emb, overlap_indices, non_overlap_indices

    def predict(self, interaction):
        r"""Predict for target domain."""
        (_, _, target_user_all_embeddings, target_item_all_embeddings) = self.forward()
        
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]

        u_embeddings = target_user_all_embeddings[user]
        i_embeddings = target_item_all_embeddings[item]

        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        r"""Full sort prediction for target domain."""
        user = interaction[self.TARGET_USER_ID]

        restore_user_e, restore_item_e = self.get_restore_e()
        u_embeddings = restore_user_e[user]
        i_embeddings = restore_item_e[:self.target_num_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores.view(-1)

    def init_restore_e(self):
        r"""Clear stored embeddings when training."""
        if self.target_restore_user_e is not None or self.target_restore_item_e is not None:
            self.target_restore_user_e, self.target_restore_item_e = None, None

    def get_restore_e(self):
        r"""Get cached embeddings for evaluation."""
        if self.target_restore_user_e is None or self.target_restore_item_e is None:
            (_, _, self.target_restore_user_e, self.target_restore_item_e) = self.forward()
        return self.target_restore_user_e, self.target_restore_item_e
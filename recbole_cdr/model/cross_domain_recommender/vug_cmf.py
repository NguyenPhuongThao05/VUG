r"""
VUG_CMF
################################################
Reference:
    Combining VUG (Virtual User Generation) with CMF (Collective Matrix Factorization)
    for enhanced cross-domain recommendation with virtual user embeddings.
    
Description:
    This model combines:
    1. VUG's virtual user generation capability using attention mechanisms
    2. CMF's shared embedding space across domains
    
    The key idea is to use VUG's generator to create virtual source embeddings
    for non-overlapping target users, then use these enhanced embeddings in 
    a CMF-style joint factorization framework.
"""

import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.loss import EmbLoss

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from .attention import CDUserItemAttention


class VUG_CMF(CrossDomainRecommender):
    r"""VUG_CMF combines virtual user generation with collective matrix factorization.
    
    It uses:
    - A dual-attention generator (from VUG) to create virtual source embeddings
    - Shared user/item embeddings (from CMF) for joint factorization
    - Enhanced training with virtual user augmentation
    """
    
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(VUG_CMF, self).__init__(config, dataset)
        
        # Load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        # Load CMF parameters
        self.embedding_size = config['embedding_size']
        self.alpha = config['alpha']  # balance between source and target loss
        self.lamda = config['lambda']  # source regularization weight
        self.gamma = config['gamma']  # target regularization weight
        
        # Load VUG generator parameters
        self.gen_weight = config['gen_weight']  # weight for generator loss
        self.enhance_mode = config['enhance_mode']  # how to use generated embeddings
        self.enhance_weight = config['enhance_weight']  # weight for enhancement
        self.user_weight_attn = config['user_weight_attn']  # attention weight between user/item level
        self.use_virtual = config['use_virtual']  # whether to use virtual embeddings
        
        # Define shared embeddings (CMF style)
        self.user_embedding = nn.Embedding(self.total_num_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.total_num_items, self.embedding_size)
        
        # Define VUG generator for creating virtual source embeddings
        self.source_generator = CDUserItemAttention(self.embedding_size)
        
        # Loss functions
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()
        self.gen_loss = nn.MSELoss()
        self.source_reg_loss = EmbLoss()
        self.target_reg_loss = EmbLoss()
        
        # Build mask for target-only users
        self.target_nonoverlap_mask = torch.zeros(
            (self.user_embedding.weight.size(0), 1),
            dtype=torch.bool,
            device=config['device']
        )
        # Mark target-only users (between num_overlap_user and num_overlap_user + num_target_only_user)
        self.target_nonoverlap_mask[
            self.num_overlap_user : self.num_overlap_user + self.num_target_only_user
        ] = True
        
        # Expand mask to match embedding dimension
        self.target_nonoverlap_mask = self.target_nonoverlap_mask.expand(
            -1, self.embedding_size
        ).to(self.user_embedding.weight.dtype)
        
        # Storage for generated embeddings
        self.generated_source_emb_cache = None
        
        # Training phase
        self.phase = 'BOTH'
        
        # Parameters initialization
        self.apply(xavier_normal_initialization)

    def set_phase(self, phase):
        """Set training phase (BOTH/SOURCE/TARGET)"""
        self.phase = phase

    def get_user_embedding(self, user, use_enhancement=True):
        r"""Get user embeddings with optional virtual enhancement.
        
        Args:
            user: user IDs tensor
            use_enhancement: whether to apply virtual user enhancement
            
        Returns:
            Enhanced or original user embeddings
        """
        user_e = self.user_embedding(user)
        
        # Apply virtual enhancement for target-only users during training
        if use_enhancement and self.use_virtual and self.training:
            # Generate virtual source embeddings
            generated_emb = self._get_or_generate_virtual_embeddings()
            
            # Create mask for current batch
            batch_mask = self.target_nonoverlap_mask[user]
            
            # Enhance embeddings based on mode
            if self.enhance_mode == "add":
                # Add weighted virtual embeddings
                user_e = user_e + self.enhance_weight * batch_mask * generated_emb[user]
            elif self.enhance_mode == "replace":
                # Replace with virtual embeddings
                user_e = torch.where(batch_mask, generated_emb[user], user_e)
            elif self.enhance_mode == "concat":
                # Concatenate (requires adjustment in embedding_size)
                virtual_part = self.enhance_weight * batch_mask * generated_emb[user]
                user_e = user_e + virtual_part
                
        return user_e

    def get_item_embedding(self, item):
        r"""Get item embeddings."""
        return self.item_embedding(item)

    def _get_or_generate_virtual_embeddings(self):
        r"""Generate or retrieve cached virtual source embeddings for all users."""
        if self.generated_source_emb_cache is None:
            # Generate virtual embeddings for all users
            _, generated_emb, _, _ = self.generate_source_embeddings()
            self.generated_source_emb_cache = generated_emb
        return self.generated_source_emb_cache
    
    def clear_virtual_cache(self):
        r"""Clear cached virtual embeddings (call after each batch/epoch)."""
        self.generated_source_emb_cache = None

    def forward(self, user, item, use_enhancement=True):
        r"""Forward pass: compute prediction scores.
        
        Args:
            user: user IDs
            item: item IDs
            use_enhancement: whether to use virtual enhancement
            
        Returns:
            Predicted scores
        """
        user_e = self.get_user_embedding(user, use_enhancement)
        item_e = self.get_item_embedding(item)
        
        # Dot product + sigmoid
        scores = self.sigmoid(torch.mul(user_e, item_e).sum(dim=1))
        return scores

    def calculate_loss(self, interaction):
        r"""Calculate total loss combining CMF and VUG generator loss."""
        
        # Clear cache at the start of each batch
        self.clear_virtual_cache()
        
        if self.phase == 'BOTH':
            return self.calculate_joint_loss(interaction)
        elif self.phase == 'TARGET':
            return self.calculate_target_loss(interaction)
        else:
            raise NotImplementedError(f"Phase {self.phase} not implemented")

    def calculate_joint_loss(self, interaction):
        r"""Calculate joint loss for both domains with virtual enhancement."""
        
        # Extract source domain data
        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]

        # Extract target domain data
        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        # Forward pass for both domains
        p_source = self.forward(source_user, source_item, use_enhancement=False)
        p_target = self.forward(target_user, target_item, use_enhancement=True)

        # Source domain loss (standard CMF)
        loss_s = self.bce_loss(p_source, source_label) + \
                 self.lamda * self.source_reg_loss(
                     self.get_user_embedding(source_user, use_enhancement=False),
                     self.get_item_embedding(source_item)
                 )

        # Target domain loss (with virtual enhancement)
        loss_t = self.bce_loss(p_target, target_label) + \
                 self.gamma * self.target_reg_loss(
                     self.get_user_embedding(target_user, use_enhancement=True),
                     self.get_item_embedding(target_item)
                 )

        # VUG generator loss (train generator to match real source embeddings)
        loss_gen = self.calculate_gen_loss(interaction)

        # Combined loss
        total_loss = (
            self.alpha * loss_s +
            (1 - self.alpha) * loss_t +
            loss_gen
        )

        return total_loss

    def calculate_target_loss(self, interaction):
        r"""Calculate loss for target domain only."""
        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        p_target = self.forward(target_user, target_item, use_enhancement=True)

        loss_t = self.bce_loss(p_target, target_label) + \
                 self.gamma * self.target_reg_loss(
                     self.get_user_embedding(target_user, use_enhancement=True),
                     self.get_item_embedding(target_item)
                 )
        return loss_t

    def calculate_gen_loss(self, interaction):
        r"""Calculate generator loss: make generated embeddings match real source embeddings
        for overlapping users."""
        
        if not self.use_virtual:
            return torch.tensor(0.0, device=interaction[self.TARGET_USER_ID].device)
        
        # Generate embeddings for users in batch
        user, generated_s_emb, overlap_indices, non_overlap_indices = \
            self.generate_source_embeddings(interaction)

        # If no overlapping users in batch, skip generator loss
        if overlap_indices.shape[0] == 0:
            return torch.tensor(0.0, device=interaction[self.TARGET_USER_ID].device)

        # Real source embeddings for overlapping users
        real_source_emb = self.user_embedding(user[overlap_indices])
        
        # Generated embeddings for overlapping users
        gen_for_overlapped = generated_s_emb[overlap_indices]

        # MSE loss between generated and real
        loss = self.gen_weight * self.gen_loss(gen_for_overlapped, real_source_emb)

        return loss

    def generate_source_embeddings(self, interaction=None):
        r"""Generate virtual source embeddings using dual-attention mechanism.
        
        This method uses VUG's attention-based generator to create virtual source
        embeddings for all users (especially non-overlapping target users).
        
        Args:
            interaction: current batch interaction (optional)
            
        Returns:
            user: user IDs
            generated_s_emb: generated source embeddings
            overlap_indices: indices of overlapping users in batch
            non_overlap_indices: indices of non-overlapping users in batch
        """
        
        # Get user IDs from interaction or generate for all users
        if interaction is None:
            user = torch.arange(
                self.total_num_users,
                device=self.user_embedding.weight.device
            )
        else:
            user = interaction[self.TARGET_USER_ID]

        # Get target user embeddings as queries
        user_t_emb = self.user_embedding(user)  # Q_user

        # Compute item-level user embeddings as additional queries
        Q_item = user_t_emb  # Simplified: use same as user-level
        
        # Identify overlapping users in batch
        overlapped_mask = (user < self.overlapped_num_users)
        overlap_indices = torch.nonzero(overlapped_mask).flatten()
        non_overlap_indices = torch.nonzero(~overlapped_mask).flatten()

        # Build keys and values from overlapping users
        K_user = self.user_embedding(user[overlap_indices])  # target embeddings
        K_item = K_user  # Simplified
        V_user = self.user_embedding(user[overlap_indices])  # source embeddings (same in CMF)

        # If no overlapping users, return zero embeddings
        if overlap_indices.shape[0] == 0:
            generated_s_emb = torch.zeros_like(user_t_emb)
            return user, generated_s_emb, overlap_indices, non_overlap_indices

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
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        
        # Use enhancement during evaluation
        p = self.forward(user, item, use_enhancement=self.use_virtual)
        return p

    def full_sort_predict(self, interaction):
        r"""Full sort prediction for target domain."""
        user = interaction[self.TARGET_USER_ID]
        
        # Get enhanced user embeddings
        user_e = self.get_user_embedding(user, use_enhancement=self.use_virtual)
        
        # Get all target items
        all_item_e = self.item_embedding.weight[:self.target_num_items]
        
        # Compute scores
        scores = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return scores.view(-1)

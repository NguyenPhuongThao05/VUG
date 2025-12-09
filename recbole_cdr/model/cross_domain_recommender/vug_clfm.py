r"""
VUG_CLFM
################################################
Reference:
    Combining VUG (Virtual User Generation) with CLFM (Cross-domain Learning via Feature Mapping)
    for enhanced cross-domain recommendation with virtual user embeddings and shared latent factors.
    
Description:
    This model combines:
    1. VUG's virtual user generation capability using attention mechanisms
    2. CLFM's shared and domain-specific embedding structure
    
    Key innovations:
    - Uses VUG generator to create virtual embeddings for target-only users
    - Employs CLFM's factorization with shared and specific components
    - Enhanced user representation through virtual user augmentation
"""

import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.loss import EmbLoss

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from .attention import CDUserItemAttention


class VUG_CLFM(CrossDomainRecommender):
    r"""VUG_CLFM combines virtual user generation with cross-domain latent factor mapping.
    
    It uses:
    - A dual-attention generator (from VUG) to create virtual source embeddings
    - Shared and domain-specific embeddings (from CLFM) for better transfer learning
    - Enhanced training with virtual user augmentation for non-overlapping users
    """
    
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(VUG_CLFM, self).__init__(config, dataset)
        
        # Load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        # Load CLFM parameters (RecBole Config object style)
        self.user_embedding_size = config['user_embedding_size']
        self.source_item_embedding_size = config['source_item_embedding_size']
        self.target_item_embedding_size = self.source_item_embedding_size  # Same as CLFM
        self.share_embedding_size = config['share_embedding_size']
        self.alpha = config['alpha']  # balance between source and target loss
        self.reg_weight = config['reg_weight']  # regularization weight
        
        # Validation for shared dimension
        assert 0 <= self.share_embedding_size <= self.source_item_embedding_size and \
               0 <= self.share_embedding_size <= self.target_item_embedding_size, \
               f"Shared dimension {self.share_embedding_size} must be <= both source {self.source_item_embedding_size} and target {self.target_item_embedding_size} embedding sizes"
        
        # Load VUG generator parameters
        self.gen_weight = config['gen_weight']  # weight for generator loss
        self.enhance_mode = config['enhance_mode']  # how to use generated embeddings
        self.enhance_weight = config['enhance_weight']  # weight for enhancement
        self.user_weight_attn = config['user_weight_attn']  # attention weight
        self.use_virtual = config['use_virtual']  # whether to use virtual embeddings
        
        # Define CLFM embeddings
        self.source_user_embedding = nn.Embedding(self.total_num_users, self.user_embedding_size)
        self.target_user_embedding = nn.Embedding(self.total_num_users, self.user_embedding_size)
        
        self.source_item_embedding = nn.Embedding(self.total_num_items, self.source_item_embedding_size)
        self.target_item_embedding = nn.Embedding(self.total_num_items, self.target_item_embedding_size)
        
        # Define CLFM linear layers
        if self.share_embedding_size > 0:
            self.shared_linear = nn.Linear(self.user_embedding_size, self.share_embedding_size, bias=False)
        if self.source_item_embedding_size - self.share_embedding_size > 0:
            self.source_only_linear = nn.Linear(
                self.user_embedding_size, 
                self.source_item_embedding_size - self.share_embedding_size,
                bias=False
            )
        if self.target_item_embedding_size - self.share_embedding_size > 0:
            self.target_only_linear = nn.Linear(
                self.user_embedding_size, 
                self.target_item_embedding_size - self.share_embedding_size,
                bias=False
            )
        
        # Define VUG generator for creating virtual source embeddings
        self.source_generator = CDUserItemAttention(self.user_embedding_size)
        
        # Loss functions
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()
        self.gen_loss = nn.MSELoss()
        self.source_reg_loss = EmbLoss()
        self.target_reg_loss = EmbLoss()
        
        # Build mask for target-only users
        self.target_nonoverlap_mask = torch.zeros(
            (self.target_user_embedding.weight.size(0), 1),
            dtype=torch.bool,
            device=config['device']
        )
        # Mark target-only users
        self.target_nonoverlap_mask[
            self.num_overlap_user : self.num_overlap_user + self.num_target_only_user
        ] = True
        
        # Expand mask to match embedding dimension
        self.target_nonoverlap_mask = self.target_nonoverlap_mask.expand(
            -1, self.user_embedding_size
        ).to(self.target_user_embedding.weight.dtype)
        
        # Store overlapped_num_users for use in generation
        self.overlapped_num_users = self.num_overlap_user
        
        # Storage for generated embeddings
        self.generated_source_emb_cache = None
        
        # Training phase
        self.phase = 'BOTH'
        
        # Parameters initialization
        self.apply(xavier_normal_initialization)

    def set_phase(self, phase):
        """Set training phase (BOTH/TARGET)"""
        self.phase = phase

    def get_enhanced_user_embedding(self, user, domain='target', use_enhancement=True):
        r"""Get user embeddings with optional virtual enhancement.
        
        Args:
            user: user IDs tensor
            domain: 'source' or 'target'
            use_enhancement: whether to apply virtual user enhancement
            
        Returns:
            Enhanced or original user embeddings
        """
        if domain == 'source':
            user_e = self.source_user_embedding(user)
        else:
            user_e = self.target_user_embedding(user)
        
        # Apply virtual enhancement for target-only users during training
        if (use_enhancement and self.use_virtual and self.training and 
            domain == 'target'):
            
            # Generate virtual source embeddings
            generated_emb = self._get_or_generate_virtual_embeddings()
            
            # Create mask for current batch
            batch_mask = self.target_nonoverlap_mask[user]
            
            # Enhance embeddings based on mode
            if self.enhance_mode == "add":
                user_e = user_e + self.enhance_weight * batch_mask * generated_emb[user]
            elif self.enhance_mode == "replace":
                user_e = torch.where(batch_mask, generated_emb[user], user_e)
            elif self.enhance_mode == "concat":
                virtual_part = self.enhance_weight * batch_mask * generated_emb[user]
                user_e = user_e + virtual_part
                
        return user_e

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

    def source_forward(self, user, item, use_enhancement=False):
        r"""Forward pass for source domain."""
        user_embedding = self.get_enhanced_user_embedding(
            user, domain='source', use_enhancement=use_enhancement
        )
        item_embedding = self.source_item_embedding(item)
        
        # CLFM factorization
        factors = []
        if self.share_embedding_size > 0:
            share_factors = self.shared_linear(user_embedding)
            factors.append(share_factors)
        if self.source_item_embedding_size - self.share_embedding_size > 0:
            only_factors = self.source_only_linear(user_embedding)
            factors.append(only_factors)
        
        factors = torch.cat(factors, dim=1)
        output = self.sigmoid(torch.mul(factors, item_embedding).sum(dim=1))
        
        return output

    def target_forward(self, user, item, use_enhancement=True):
        r"""Forward pass for target domain with virtual enhancement."""
        user_embedding = self.get_enhanced_user_embedding(
            user, domain='target', use_enhancement=use_enhancement
        )
        item_embedding = self.target_item_embedding(item)
        
        # CLFM factorization
        factors = []
        if self.share_embedding_size > 0:
            share_factors = self.shared_linear(user_embedding)
            factors.append(share_factors)
        if self.target_item_embedding_size - self.share_embedding_size > 0:
            only_factors = self.target_only_linear(user_embedding)
            factors.append(only_factors)
        
        factors = torch.cat(factors, dim=1)
        output = self.sigmoid(torch.mul(factors, item_embedding).sum(dim=1))
        
        return output

    def calculate_loss(self, interaction):
        r"""Calculate main loss (CLFM only) - generator loss is handled separately by trainer."""
        
        # Clear cache at the start of each batch
        self.clear_virtual_cache()
        
        if self.phase == 'BOTH':
            return self.calculate_map_loss(interaction)
        elif self.phase == 'TARGET':
            return self.calculate_target_loss(interaction)
        else:
            raise NotImplementedError(f"Phase {self.phase} not implemented")

    def calculate_map_loss(self, interaction):
        r"""Calculate joint loss for both domains."""
        
        # Extract source domain data
        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]

        # Extract target domain data
        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        # Forward pass for both domains
        p_source = self.source_forward(source_user, source_item, use_enhancement=False)
        p_target = self.target_forward(target_user, target_item, use_enhancement=True)

        # Source domain loss
        loss_s = (
            self.bce_loss(p_source, source_label) + 
            self.reg_weight * self.source_reg_loss(
                self.source_user_embedding(source_user),
                self.source_item_embedding(source_item)
            )
        )

        # Target domain loss (with virtual enhancement)
        loss_t = (
            self.bce_loss(p_target, target_label) + 
            self.reg_weight * self.target_reg_loss(
                self.get_enhanced_user_embedding(target_user, 'target', True),
                self.target_item_embedding(target_item)
            )
        )

        # Combined CLFM loss (generator loss is handled separately)
        total_loss = loss_s * self.alpha + loss_t * (1 - self.alpha)

        return total_loss

    def calculate_target_loss(self, interaction):
        r"""Calculate loss for target domain only."""
        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        p_target = self.target_forward(target_user, target_item, use_enhancement=True)

        loss_t = (
            self.bce_loss(p_target, target_label) + 
            self.reg_weight * self.target_reg_loss(
                self.get_enhanced_user_embedding(target_user, 'target', True),
                self.target_item_embedding(target_item)
            )
        )
        return loss_t

    def calculate_gen_loss(self, interaction):
        r"""Calculate generator loss: make generated embeddings match real source embeddings
        for overlapping users."""
        
        if not self.use_virtual:
            # Create zero loss with gradient flow through generator
            dummy_loss = 0.0 * sum(p.sum() for p in self.source_generator.parameters())
            return dummy_loss
        
        # Generate embeddings for users in batch
        user, generated_s_emb, overlap_indices, non_overlap_indices = \
            self.generate_source_embeddings(interaction)

        # If no overlapping users in batch, return small generator regularization
        if overlap_indices.shape[0] == 0:
            reg_loss = 0.0001 * sum(p.pow(2).sum() for p in self.source_generator.parameters())
            return reg_loss

        # Real source embeddings for overlapping users
        real_source_emb = self.source_user_embedding(user[overlap_indices])
        gen_for_overlapped = generated_s_emb[overlap_indices]

        # MSE loss between generated and real + regularization
        mse_loss = self.gen_loss(gen_for_overlapped, real_source_emb)
        reg_loss = 0.0001 * sum(p.pow(2).sum() for p in self.source_generator.parameters())
        
        total_gen_loss = self.gen_weight * mse_loss + reg_loss

        return total_gen_loss

    def generate_source_embeddings(self, interaction=None):
        r"""Generate virtual source embeddings using dual-attention mechanism."""
        
        # Get user IDs from interaction or generate for all users
        if interaction is None:
            user = torch.arange(
                self.total_num_users,
                device=self.target_user_embedding.weight.device
            )
        else:
            user = interaction[self.TARGET_USER_ID]

        # Get target user embeddings as queries
        user_t_emb = self.target_user_embedding(user)  # Q_user
        Q_item = user_t_emb  # Simplified: use same as user-level
        
        # Identify overlapping users in batch
        overlapped_mask = (user < self.overlapped_num_users)
        overlap_indices = torch.nonzero(overlapped_mask).flatten()
        non_overlap_indices = torch.nonzero(~overlapped_mask).flatten()

        # If no overlapping users, return zeros with gradient flow
        if overlap_indices.shape[0] == 0:
            generated_s_emb = torch.zeros_like(user_t_emb)
            # Add small contribution from generator to maintain gradient flow
            dummy_gen = 0.0 * sum(p.sum() for p in self.source_generator.parameters())
            generated_s_emb = generated_s_emb + dummy_gen
            return user, generated_s_emb, overlap_indices, non_overlap_indices

        # Build keys and values from overlapping users
        overlapped_users = user[overlap_indices]
        K_user = self.target_user_embedding(overlapped_users)  # Target embeddings as keys
        K_item = K_user  # Simplified
        V_user = self.source_user_embedding(overlapped_users)  # Source embeddings as values

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
        p = self.target_forward(user, item, use_enhancement=self.use_virtual)
        return p

    def full_sort_predict(self, interaction):
        r"""Full sort prediction for target domain."""
        user = interaction[self.TARGET_USER_ID]
        
        # Get enhanced user embeddings
        user_embedding = self.get_enhanced_user_embedding(
            user, domain='target', use_enhancement=self.use_virtual
        )
        all_item_embedding = self.target_item_embedding.weight[:self.target_num_items]
        
        # CLFM factorization for full sort
        factors = []
        if self.share_embedding_size > 0:
            share_factors = self.shared_linear(user_embedding)
            factors.append(share_factors)
        if self.target_item_embedding_size - self.share_embedding_size > 0:
            only_factors = self.target_only_linear(user_embedding)
            factors.append(only_factors)
        
        factors = torch.cat(factors, dim=1)
        score = torch.matmul(factors, all_item_embedding.transpose(0, 1))
        return score.view(-1)
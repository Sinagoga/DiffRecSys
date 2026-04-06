# Adapted from https://github.com/liuzhao09/DiffGRM/

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.abstract_model import AbstractModel


############################################################
# From genrec.models.DIFF_GRM.model.py                     #
############################################################


def make_norm(norm_type: str, dim: int, eps: float):
    if (norm_type or "layernorm").lower() == "rmsnorm":
        return nn.RMSNorm(dim, eps=eps)
    return nn.LayerNorm(dim, eps=eps)


class MultiHeadAttention(nn.Module):

    def __init__(self, emb_dim, n_head, attn_drop=0.1, resid_drop=0.1):
        super().__init__()
        assert emb_dim % n_head == 0
        self.n_head = n_head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // n_head

        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim)

        self.attn_dropout = nn.Dropout(attn_drop)
        self.resid_dropout = nn.Dropout(resid_drop)

        # Initialize weights
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x, attention_mask=None, key_value=None, past_key_value=None, use_cache=False, is_decoder_self_attn=False):
        B, T, C = x.size()

        if key_value is not None:
            # Cross attention: Q from x, K,V from key_value
            q = self.qkv(x)[:, :, :self.emb_dim]  # Only take Q part
            k, v = key_value.chunk(2, dim=-1)  # key_value should be [B, T_enc, 2*emb_dim]
            T_kv = k.size(1)
        else:
            # Self attention
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            T_kv = T

        # Handle past key-value cache for incremental decoding
        if past_key_value is not None and use_cache and is_decoder_self_attn:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
            T_kv = k.size(1)

        # Save concatenated full k and v for cache (before reshape)
        k_for_cache = k
        v_for_cache = v

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T_kv, head_dim)
        v = v.view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T_kv, head_dim)

        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_head, T, T_kv)

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: (B, T, T_kv) or (B, 1, T, T_kv)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)  # Add head dimension
            att = att.masked_fill(attention_mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values
        y = torch.matmul(att, v)  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, emb_dim)

        # Output projection
        y = self.resid_dropout(self.proj(y))

        # Prepare cache for next iteration - preserve original 3D k and v
        present_key_value = (k_for_cache, v_for_cache) if use_cache else None

        return y, present_key_value


class FeedForward(nn.Module):

    def __init__(self, emb_dim, n_inner, resid_drop=0.1, act='gelu'):
        super().__init__()
        self.c_fc = nn.Linear(emb_dim, n_inner)
        self.c_proj = nn.Linear(n_inner, emb_dim)
        self.dropout = nn.Dropout(resid_drop)
        self.act = F.gelu if act == 'gelu' else F.relu

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return self.dropout(x)


class EncoderBlock(nn.Module):

    def __init__(self, emb_dim, n_head, n_inner, attn_drop=0.1, resid_drop=0.1, 
                 act='gelu', norm_type='layernorm', norm_eps=1e-6):
        super().__init__()
        self.ln_1 = make_norm(norm_type, emb_dim, norm_eps)
        self.attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        self.ln_2 = make_norm(norm_type, emb_dim, norm_eps)
        self.mlp = FeedForward(emb_dim, n_inner, resid_drop, act)

    def forward(self, x, attention_mask=None):
        # Self-attention + residual connection (non-decoder self-attention)
        attn_output, _ = self.attn(self.ln_1(x), attention_mask=attention_mask, is_decoder_self_attn=False)
        x = x + attn_output
        
        # Feed-forward network + residual connection
        x = x + self.mlp(self.ln_2(x))
        return x


class DecoderBlock(nn.Module):

    def __init__(self, emb_dim, n_head, n_inner, attn_drop=0.1, resid_drop=0.1, 
                 act='gelu', norm_type='layernorm', norm_eps=1e-6):
        super().__init__()
        self.ln_1 = make_norm(norm_type, emb_dim, norm_eps)
        self.self_attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        self.ln_2 = make_norm(norm_type, emb_dim, norm_eps)
        self.cross_attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        self.ln_3 = make_norm(norm_type, emb_dim, norm_eps)
        self.mlp = FeedForward(emb_dim, n_inner, resid_drop, act)

    def forward(self, x, encoder_hidden=None, attention_mask=None, 
                past_key_value=None, use_cache=False, cross_key_value=None):
        # Change: removed causal mask because diffusion models do not require strict sequential ordering
        # Self-attention (no causal mask)
        self_past_kv = None
        cross_past_kv = None
        if past_key_value is not None:
            if len(past_key_value) >= 1:
                self_past_kv = past_key_value[0]
            if len(past_key_value) >= 2:
                cross_past_kv = past_key_value[1]
        
        attn_output, present_key_value = self.self_attn(
            self.ln_1(x), 
            attention_mask=None,  # no causal mask
            past_key_value=self_past_kv,
            use_cache=use_cache,
            is_decoder_self_attn=True
        )
        x = x + attn_output

        # Cross attention
        if encoder_hidden is not None:
            if cross_key_value is not None:
                # Use precomputed KV to avoid redundant computation
                encoder_kv = cross_key_value
            else:
                # Fallback to old logic: recompute (used only in non-optimized path)
                encoder_kv = torch.cat([encoder_hidden, encoder_hidden], dim=-1)  # Concat K and V
            
            cross_attn_output, cross_present = self.cross_attn(
                self.ln_2(x),
                key_value=encoder_kv,
                past_key_value=cross_past_kv,
                use_cache=use_cache
            )
            x = x + cross_attn_output
            
            if use_cache:
                present_key_value = (present_key_value, cross_present)
        
        # Feed-forward network
        x = x + self.mlp(self.ln_3(x))
        
        return_dict = {}
        return_dict['hidden_states'] = x
        if use_cache:
            return_dict['present_key_value'] = present_key_value
        
        return return_dict


class Encoder(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        n_digit: int,

        n_embd: int,
        n_head: int,
        n_inner: int,
        dropout: float,

        attn_pdrop: float,
        resid_pdrop: float,

        norm_type: str | None = None,
        norm_eps: float | None = None,

        max_history_len: int | None = None,

        sid_offset: int = 3,
        codebook_size: int = 1000,

        encoder_n_layer: int = 6,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.n_digit = n_digit

        # Model dimensions
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_inner = n_inner
        self.dropout = dropout

        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop

        # Normalization configuration
        self.norm_type = norm_type
        if self.norm_type is None:
            self.norm_type = 'layernorm'
        else:
            self.norm_type = self.norm_type.lower()
        self.norm_eps  = norm_eps
        if self.norm_eps is None:
            self.norm_eps = 1e-6 if self.norm_type=='rmsnorm' else 1e-5
        else:
            self.norm_eps = float(self.norm_eps)

        self.tokenizer_sid_offset = sid_offset  # codebook token ID
        self.codebook_size = codebook_size
        
        # Encoder layers
        self.encoder_n_layer = encoder_n_layer
        
        
        # ==== Read new strategy ====
        # Embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.n_embd)
        
        # Add item_mlp consistent with RPG_ED: compress n_digit SID tokens into a single item token
        self.item_mlp = nn.Sequential(
            nn.Linear(self.n_digit * self.n_embd, self.n_embd),  # n_digit×d -> d
            nn.ReLU(),
            nn.Linear(self.n_embd, self.n_embd)
        )
        
        # Added: mask embedding table to represent masked positions
        self.mask_emb_table = nn.Embedding(self.n_digit, self.n_embd)
        
        # Positional encoding: add absolute positional encoding only for the encoder (consistent with RPG_ED)
        self.max_history_len = max_history_len if max_history_len is not None else 50  # default 50
        self.pos_emb_enc = nn.Embedding(self.max_history_len, self.n_embd)
        # Removed decoder positional encoding; decoder uses mask embeddings only
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                self.n_embd, self.n_head, self.n_inner,
                self.attn_pdrop, self.resid_pdrop,
                act='gelu',
                norm_type=self.norm_type, norm_eps=self.norm_eps
            )
            for _ in range(self.encoder_n_layer)
        ])
        
        # Layer normalization
        self.ln_f = make_norm(self.norm_type, self.n_embd, self.norm_eps)
        
        # Dropout
        self.drop = nn.Dropout(self.dropout)
    
    def get_digits_embeddings(self):
        """
        Get the full embedding matrix for all digit positions, used for efficient parallel decoding.

        Returns:
            A list of length n_digit, where each element is a tensor of shape (codebook_size, d_model)
            containing the embeddings for that digit position.
        """
        start = self.tokenizer_sid_offset
        end = start + self.n_digit * self.codebook_size
        E_sub = self.embedding.weight[start:end]
        
        return E_sub.reshape(self.n_digit, self.codebook_size, self.n_embd)  # (n_digit, codebook_size, d_model)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Diffusion training: process masked data and predict masked positions.

        Args:
            batch: dictionary containing the following fields:
                - history_sid: historical SID sequence [B, seq_len, n_digit]
                - decoder_input_ids: decoder inputs [B, n_digit]
                - decoder_labels: ground-truth labels [B, n_digit]
        """
        device = next(self.parameters()).device
        
        # --- Encoder ---
        history_sid = batch['history_sid'].to(device)  # [B, seq_len, n_digit]
        B, seq_len, n_digit = history_sid.shape
        
        # Assert: history_sid should be codebook id (0..K-1) or PAD (-1)
        valid_hist = ((history_sid == -1) | ((history_sid >= 0) & (history_sid < self.codebook_size))).all()
        assert bool(valid_hist), \
            f"history_sid must be codebook id (0..{self.codebook_size-1}) or -1 (PAD); found out-of-range values"
        
        # 1. Convert history SID into token IDs
        digit_indices = torch.arange(n_digit, device=device)
        history_tokens = history_sid + self.tokenizer_sid_offset + digit_indices * self.codebook_size

        history_tokens = torch.where(history_sid == -1, 0, history_tokens)
        history_tokens = torch.clamp(history_tokens, 0, self.vocab_size - 1)
        
        # 2. Get token embeddings
        tok_emb = self.embedding(history_tokens)  # [B, seq_len, n_digit, d]
        B, S, _, d = tok_emb.shape
        
        # 3. Reshape and compress via MLP: n_digit SID tokens -> 1 item token
        item_emb = tok_emb.reshape(B, S, self.n_digit * d)  # [B, S, n_digit*d]
        item_emb = self.item_mlp(item_emb)  # [B, S, d]
        
        # 4. Add positional embeddings (consistent with RPG_ED)
        pos_ids = torch.arange(S, device=item_emb.device)  # (S,)
        pos_emb = self.pos_emb_enc(pos_ids)  # (S, d)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, S, d)
        
        # 5. Add positional embedding to item_emb
        encoder_hidden = item_emb + pos_emb  # [B, S, d]
        encoder_hidden = self.drop(encoder_hidden)
        
        # 6. Handle attention mask for PAD positions
        if 'history_mask' in batch:
            history_mask = batch['history_mask'].to(device)  # [B, seq_len]
            # Create attention mask: True=valid position, False=PAD position
            attention_mask = history_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]
            attention_mask = attention_mask.expand(-1, -1, seq_len, -1)  # [B, 1, seq_len, seq_len]
        else:
            attention_mask = None
        
        # Pass through encoder blocks
        encoder_hidden = encoder_hidden
        for block in self.encoder_blocks:
            encoder_hidden = block(encoder_hidden, attention_mask=attention_mask)
        
        encoder_hidden = self.ln_f(encoder_hidden)  # [B, seq_len*n_digit, emb_dim]
        
        # >>> New: zero out encoder_hidden at PAD positions to prevent cross-attn from seeing invalid KV <<<
        if 'history_mask' in batch:
            history_mask = batch['history_mask'].to(device)  # [B, S], True=valid
            encoder_hidden = encoder_hidden * history_mask.unsqueeze(-1).float()

        return encoder_hidden
    
    def encode_input_ids(self, input_ids: torch.Tensor, mask_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Encode input token IDs into embeddings, used for the decoder input.

        Args:
            input_ids: [B, n_digit] token IDs for the decoder input
            mask_positions: [B, n_digit] boolean tensor indicating masked positions
        Returns:
            embeddings: [B, n_digit, d_model] corresponding embeddings
        """
        device = next(self.parameters()).device
        
        B, n_digit = input_ids.shape

        # Get mask positions; if not provided assume no positions are masked
        if mask_positions is None:
            mask_positions = torch.zeros(B, n_digit, device=device)

        digit_indices = torch.arange(n_digit, device=device) # [n_digit]

        # Convert to token IDs and clamp for safety
        token_ids = input_ids + self.tokenizer_sid_offset + digit_indices * self.codebook_size # [B, n_digit]
        token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)

        # Safe embedding lookup
        input_emb = self.embedding(token_ids)  # [B, n_digit, emb_dim]

        # Choose embedding based on mask_positions
        is_masked = mask_positions.bool() # [B, n_digit]
        input_emb[is_masked] = self.mask_emb_table.weight[digit_indices].expand(B, -1, -1)[is_masked]

        return input_emb  # [B, n_digit, emb_dim]
    

class Decoder(nn.Module):

    def __init__(
        self,

        n_embd: int,
        n_head: int,
        n_inner: int,
        dropout: float,

        attn_pdrop: float,
        resid_pdrop: float,

        norm_type: str | None = None,
        norm_eps: float | None = None,

        decoder_n_layer: int = 6,
    ):
        super().__init__()
        
        # Model dimensions
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_inner = n_inner
        self.dropout = dropout

        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop

        # Normalization configuration
        self.norm_type = norm_type
        if self.norm_type is None:
            self.norm_type = 'layernorm'
        else:
            self.norm_type = self.norm_type.lower()
        self.norm_eps  = norm_eps
        if self.norm_eps is None:
            self.norm_eps = 1e-6 if self.norm_type=='rmsnorm' else 1e-5
        else:
            self.norm_eps = float(self.norm_eps)
        
        # Encoder layers
        self.decoder_n_layer = decoder_n_layer
        
        # Decoder blocks  
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                self.n_embd, self.n_head, self.n_inner,
                self.attn_pdrop, self.resid_pdrop,
                act='gelu',
                norm_type=self.norm_type, norm_eps=self.norm_eps
            )
            for _ in range(self.decoder_n_layer)
        ])
        
        # Layer normalization
        self.ln_f = make_norm(self.norm_type, self.n_embd, self.norm_eps)
        
        # Dropout
        self.drop = nn.Dropout(self.dropout)

    def forward(self, encoder_hidden: torch.Tensor, decoder_emb: torch.Tensor, past_key_values=None, use_cache=False):
        """
        Run only the decoder part, used for iterative prediction during inference.

        Args:
            encoder_hidden: encoder outputs [B, seq_len, emb_dim]
            decoder_emb: decoder inputs [B, n_digit, emb_dim]
            past_key_values: cached key-value pairs to speed up inference
            use_cache: whether to use KV cache
        """
        present_key_values = []
        
        # Cross-KV cache optimization: compute once in the first step and reuse from past_key_values
        encoder_kv_list = None

        decoder_hidden = self.drop(decoder_emb)
        
        if past_key_values is None:  # NOTE: The early condition was (past_key_values is None and use_cache)
            # First step: precompute cross-attention KV for each layer
            encoder_kv_list = []
            for blk in self.decoder_blocks:
                # Perform W_k/W_v projection, consistent with inference
                kv_proj = blk.cross_attn.qkv(encoder_hidden)  # [B_expanded, seq_len, 3*emb_dim]
                # extract K and V parts (skip Q part)
                k = kv_proj[..., self.n_embd:2*self.n_embd]  # [B_expanded, seq_len, emb_dim]
                v = kv_proj[..., 2*self.n_embd:]              # [B_expanded, seq_len, emb_dim]
                # concatenate K and V
                layer_kv = torch.cat([k, v], dim=-1)  # [B_expanded, seq_len, 2*emb_dim]
                encoder_kv_list.append(layer_kv)
        elif past_key_values is not None:
            # Subsequent steps: extract cross-KV from past_key_values to realize true cache reuse
            encoder_kv_list = []
            for layer_cache in past_key_values:
                if layer_cache is not None and len(layer_cache) >= 2:
                    _, cross_kv = layer_cache
                    if cross_kv is not None:
                        cross_key, cross_value = cross_kv
                        layer_kv = torch.cat([cross_key, cross_value], dim=-1)
                        encoder_kv_list.append(layer_kv)
                    else:
                        encoder_kv_list.append(None)
                else:
                    encoder_kv_list.append(None)
        
        for i, block in enumerate(self.decoder_blocks):
            # Get the past_key_value for the current layer
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            # Pass precomputed cross-KV to enable cache reuse
            current_cross_kv = encoder_kv_list[i] if encoder_kv_list is not None else None
            
            block_output = block(
                decoder_hidden, 
                encoder_hidden=encoder_hidden,     # still pass H for fallback
                past_key_value=layer_past,
                use_cache=use_cache,
                cross_key_value=current_cross_kv   # only passed on the first call
            )
            decoder_hidden = block_output['hidden_states']
            
            # Collect new key-value cache
            if use_cache:
                layer_present = block_output.get('present_key_value')
                if layer_present is not None and len(layer_present) >= 2:
                    self_present, cross_present = layer_present
                    # ensure cross_present stores separated K and V for next cache use
                    if cross_present is not None:
                        # cross_present should be in (K, V) format
                        layer_kv = encoder_kv_list[i] if encoder_kv_list is not None else None
                        if layer_kv is not None:
                            k, v = layer_kv.chunk(2, dim=-1)  # separate K and V
                            cross_present = (k, v)  # save separated format for next cache use
                    present_key_values.append((self_present, cross_present))
                else:
                    present_key_values.append(layer_present)
        
        decoder_hidden = self.ln_f(decoder_hidden)  # [B, n_digit, emb_dim]
        
        # If not using cache, set to None
        if not use_cache:
            present_key_values = None
        
        return decoder_hidden, present_key_values


class ModelOutput:

    def __init__(self):
        self.loss = None
        self.logits = None
        self.hidden_states = None
        self.past_key_values = None


class DIFF_GRM(AbstractModel):

    def __init__(
        self,
        vocab_size: int = None,
        sid_offset: int = None,
        **config: dict,
    ):
        super().__init__(config)
        
        self.config = config
        self.tokenizer_sid_offset = sid_offset  # codebook token ID
        self.n_digit = config['n_digit']
        self.codebook_size = config['codebook_size']
        self.vocab_size = vocab_size

        # Model dimensions
        self.n_embd = config['n_embd']
        self.n_head = config['n_head']
        self.n_inner = config['n_inner']
        self.dropout = config['dropout']
        
        # Encoder layers
        self.encoder_n_layer = config['encoder_n_layer']
        self.decoder_n_layer = config['decoder_n_layer']
        
        # Normalization configuration
        self.norm_type = (config.get('norm_type', 'layernorm') or 'layernorm').lower()
        self.norm_eps  = float(config.get('norm_eps', 1e-6 if self.norm_type=='rmsnorm' else 1e-5))
        
        # ==== Read new strategy ====
        self.set_masking_mode(config)  # initialize masking mode and related parameters

        self.encoder = Encoder(
            vocab_size=self.vocab_size,
            n_digit=self.n_digit,
            n_embd=self.n_embd,
            n_head=self.n_head,
            n_inner=self.n_inner,
            dropout=self.dropout,
            attn_pdrop=config['attn_pdrop'],
            resid_pdrop=config['resid_pdrop'],
            norm_type=self.norm_type,
            norm_eps=self.norm_eps,
            max_history_len=config.get('max_history_len', 50),
            sid_offset=self.tokenizer_sid_offset,
            codebook_size=self.codebook_size,
            encoder_n_layer=self.encoder_n_layer
        )
        
        # Decoder blocks
        self.decoder = Decoder(
            n_embd=self.n_embd,
            n_head=self.n_head,
            n_inner=self.n_inner,
            dropout=self.dropout,
            attn_pdrop=config['attn_pdrop'],
            resid_pdrop=config['resid_pdrop'],
            norm_type=self.norm_type,
            norm_eps=self.norm_eps,
            decoder_n_layer=self.decoder_n_layer,
        )
        
        
        # -- 1.1 Remove old separate heads; use shared embedding dot-product --
        share_out = self.config.get('share_decoder_output_embedding', True)
        if share_out:
            # Direct weight-tying; no new parameters
            self.output_adapter = nn.Identity()
            print(f"[DIFF_GRM] Using shared embedding dot-product output layer")
        else:
            # Use this line if rolling back to an independent head
            self.output_adapter = nn.Linear(self.n_embd, self.n_embd, bias=False)
            print(f"[DIFF_GRM] Using independent Linear output adapter")
        # -------------------------------------------------------------
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            # LayerNorm: has bias; RMSNorm: only weight (no bias)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.ones_(module.weight)

    def resample_mask_prob_if_needed(self, strategy_config = None):
        """
        When using random + mask_prob_random=true, call at the start of each training epoch
        to resample the mask probability from the configured range and update the mask
        probability and loss scaling for the current epoch.
        """
        if self.masking_strategy != 'random' or not getattr(self, 'mask_prob_random', False):
            return  # only applies to random masking strategy with random mask probability enabled
        
        low = float((strategy_config or self.config).get('mask_prob_random_min', 0.0))
        high = float((strategy_config or self.config).get('mask_prob_random_max', 1.0))
        if not (0.0 <= low <= high <= 1.0):
            raise ValueError(
                f"mask_prob_random_min/max must satisfy 0.0 <= min <= max <= 1.0, got min={low}, max={high}"
            )
        
        sampled_prob = float(np.random.uniform(low, high))
        self.mask_probs = [sampled_prob]  # single view
        self.sampled_mask_prob = sampled_prob
        print(f"[MODEL] [Epoch-Resample] RANDOM masking prob resampled to {sampled_prob:.4f} (range [{low}, {high}]); augment_factor=1")

    def set_masking_mode(self, strategy_config):
        """
        Hot switch during training:
        - strategy: 'guided' | 'sequential' | 'random'
        - strategy_config: hyperparameters required by the corresponding strategy (see below)
        """
        self.masking_strategy = strategy_config.get('masking_strategy', 'random')

        if self.masking_strategy == 'sequential':
            # steps
            # Sequential multi-view
            seq_cfg = strategy_config.get('sequential_steps', 'auto')
            self.seq_steps = self.n_digit if seq_cfg in (None, 'auto') else int(seq_cfg)
            assert 1 <= self.seq_steps <= self.n_digit, \
                f"sequential_steps must be 1~{self.n_digit}, got {self.seq_steps}"
            
            # Added: support for multiple paths
            self.sequential_paths = int(strategy_config.get('sequential_paths', 1))
            assert self.sequential_paths >= 1, \
                f"sequential_paths must be >= 1, got {self.sequential_paths}"
            
            self.augment_factor = self.seq_steps * self.sequential_paths  # updated computation
            print(f"[MODEL] -> use SEQUENTIAL views: steps={self.seq_steps}, "
                  f"paths={self.sequential_paths}, augment_factor={self.augment_factor}")
            
            # Removed unnecessary mask_probs setting to save memory
            self.mask_probs = None

        elif self.masking_strategy == 'guided':
            # Confidence-guided sequential multi-view (model decides reveal order per batch)
            guided_cfg = strategy_config.get('guided_steps', 'auto')
            self.guided_steps = self.n_digit if guided_cfg in (None, 'auto') else int(guided_cfg)

            # Limit to at most 4 steps (e.g., n_digit=4)
            self.guided_steps = min(self.guided_steps, self.n_digit, 4)
            self.guided_conf_metric = strategy_config.get('guided_conf_metric', 'msp')
            assert self.guided_conf_metric in ('msp', 'entropy'), \
                f"guided_conf_metric must be one of ['msp','entropy'], got {self.guided_conf_metric}"

            # Added: option to reveal positions selected as 'most' or 'least' confident
            self.guided_select = strategy_config.get('guided_select', 'most')
            assert self.guided_select in ('most', 'least'), \
                f"guided_select must be one of ['most','least'], got {self.guided_select}"
            # Note: forward reads self.config['guided_refresh_each_step'], so sync this back to config
            self.config['guided_refresh_each_step'] = bool(strategy_config.get('guided_refresh_each_step', False))
            
            self.augment_factor = self.guided_steps
            print(f"[MODEL] -> GUIDED: steps={self.guided_steps}, metric={self.guided_conf_metric}, "
                  f"select={self.guided_select}, augment_factor={self.augment_factor}")

            # Removed unnecessary mask_probs setting to save memory
            self.mask_probs = None

        elif self.masking_strategy == 'random':
            # Legacy random mask branch (keep original logic)
            # Diffusion specific parameters - multi-probability mask configuration
            # New: support sampling a single mask probability from a range and repeat it via augment_factor
            self.mask_prob_random = bool(strategy_config.get('mask_prob_random', False))

            if self.mask_prob_random:
                self.resample_mask_prob_if_needed(strategy_config)  # sample initial mask probability

                self.augment_factor = 1

            elif strategy_config.get('mask_probs', None) is not None:
                # New method: directly specify multiple mask probabilities
                mask_probs_raw = strategy_config['mask_probs']

                if isinstance(mask_probs_raw, str):
                    # String format: "1.0,0.75,0.5,0.25"
                    self.mask_probs = [float(p.strip()) for p in mask_probs_raw.split(',')]
                elif isinstance(mask_probs_raw, (list, tuple)):
                    # List or tuple format: [1.0, 0.75, 0.5, 0.25]
                    self.mask_probs = [float(p) for p in mask_probs_raw]
                elif isinstance(mask_probs_raw, (int, float)):
                    # Single numeric value converted to a single-element list
                    self.mask_probs = [float(mask_probs_raw)]
                else:
                    # Other types: try converting to string and parsing
                    try:
                        mask_probs_str = str(mask_probs_raw)
                        self.mask_probs = [float(p.strip()) for p in mask_probs_str.split(',')]
                    except (ValueError, AttributeError):
                        raise ValueError(f"Cannot parse mask_probs: {mask_probs_raw} (type: {type(mask_probs_raw)}). "
                                       "Expected string like '1.0,0.75,0.5,0.25' or list like [1.0, 0.75, 0.5, 0.25]")
                
                self.augment_factor = len(self.mask_probs)  # auto-set augment factor
                print(f"[MODEL] Using multi-probability masking: {self.mask_probs}")

            else:
                # Legacy method: single mask probability + augment factor
                mask_prob = strategy_config.get('mask_prob', 0.5)
                self.augment_factor = strategy_config.get('augment_factor', 4)
                self.mask_probs = [float(mask_prob)] * self.augment_factor  # repeat same probability
                print(f"[MODEL] Using single-probability masking: {mask_prob} x {self.augment_factor}")
                
        
            # Validate mask probabilities validity (only for random strategy)
            if self.mask_probs is not None:
                for i, prob in enumerate(self.mask_probs):
                    if not (0.0 <= prob <= 1.0):
                        raise ValueError(f"mask_probs[{i}] = {prob} is not in valid range [0.0, 1.0]")

        else:
            raise ValueError(f"Unknown masking strategy: {self.masking_strategy}")

    def forward_decoder_only(self, decoder_input_ids: torch.Tensor, encoder_hidden: torch.Tensor,
                             mask_positions: torch.Tensor = None, digit=None,
                             past_key_values=None, use_cache=False):
        """
        Run only the decoder part, used for iterative prediction during inference.

        Args:
            batch: dictionary containing the following fields:
                - decoder_input_ids: decoder inputs [B, n_digit]
                - encoder_hidden: encoder outputs [B, seq_len, emb_dim]
                - mask_positions: mask positions [B, n_digit] (optional)
            digit: which digit position to predict
            past_key_values: cached key-value pairs to speed up inference
            use_cache: whether to use KV cache
        """
        _, n_digit = decoder_input_ids.shape
        
        # Construct decoder input embeddings
        decoder_emb = self.encoder.encode_input_ids(decoder_input_ids, mask_positions)  # [B, n_digit, emb_dim]
        
        # Pass through decoder blocks with KV cache support
        decoder_hidden, present_key_values = self.decoder.forward(
            encoder_hidden=encoder_hidden,
            decoder_emb=decoder_emb,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        # Compute logits for the specified digit
        E = self.encoder.get_digits_embeddings()  # (n_digit, codebook_size, d_model)
        if digit is not None:
            h = self.output_adapter(decoder_hidden[:, digit, :])  # (B, d_model)
            logits = torch.matmul(h, E[digit].T)
        else:
            h = self.output_adapter(decoder_hidden)  # (B, n_digit, d_model)
            logits = torch.einsum('bdm,dcm->bdc', h, E)  # [B, n_digit, codebook_size]
        
        return logits, present_key_values

    def calculate_loss(self, batch):
        """
        Diffusion training: process masked data and predict masked positions.

        Args:
            batch: dictionary containing the following fields:
                - history_sid: historical SID sequence [B, seq_len, n_digit]
                - decoder_input_ids: decoder inputs [B, n_digit]
                - decoder_labels: ground-truth labels [B, n_digit]
        """
        encoder_hidden = self.encoder(batch)  # [B, seq_len, emb_dim]
        
        device = next(self.parameters()).device
        
        # --- Multi-probability mask augmentation ---
        decoder_input_ids = batch['decoder_input_ids'].to(device)  # [B, n_digit]
        decoder_labels = batch['decoder_labels'].to(device)  # [B, n_digit]
        
        # Ensure decoder inputs are within valid range
        decoder_input_ids = torch.clamp(decoder_input_ids, 0, self.codebook_size - 1)
        decoder_labels = torch.clamp(decoder_labels, 0, self.codebook_size - 1)
        
        # ---------- Construct training views ----------
        all_masked_input_ids = []
        all_labels = []
        all_mask_positions = []
        all_encoder_hidden = []
        
        B = decoder_input_ids.shape[0]
        
        if self.masking_strategy == 'sequential':
            # Sequential multi-view: support multiple parallel paths
            for p in range(self.sequential_paths):  # generate multiple parallel paths
                # 1) Random order per sample for this path
                orders = torch.argsort(torch.rand(B, self.n_digit, device=device), dim=1)

                # step-0: all MASK
                full_mask = torch.ones(B, self.n_digit, dtype=torch.bool, device=device)
                inp0 = decoder_input_ids.new_zeros(B, self.n_digit)        # all 0 -> MASK
                all_masked_input_ids.append(inp0)
                all_labels.append(decoder_labels)
                all_mask_positions.append(full_mask.float())
                all_encoder_hidden.append(encoder_hidden)

                # step-1 ... step-(seq_steps-1): progressively reveal according to random order
                for reveal in range(1, self.seq_steps):        # 1 .. seq_steps-1
                    mask_pos = torch.ones_like(full_mask)      # start with all MASK

                    # orders[:, :reveal] shape (B, reveal)
                    reveal_idx = orders[:, :reveal]            # columns to reveal for each sample
                    mask_pos.scatter_(1, reveal_idx, 0)        # set 0 to indicate 'not masked'

                    inp = decoder_input_ids.clone()
                    inp[mask_pos] = 0                          # masked positions set to 0

                    all_masked_input_ids.append(inp)
                    all_labels.append(decoder_labels)
                    all_mask_positions.append(mask_pos.float())
                    all_encoder_hidden.append(encoder_hidden)
        elif self.masking_strategy == 'guided':
            B = decoder_labels.size(0)
            device = decoder_labels.device

            def score_with_mask(cur_mask: torch.Tensor):
                # cur_mask: [B, n_digit], True=masked (needs prediction)
                cur_inp = decoder_input_ids.new_zeros(B, self.n_digit, device=device)
                cur_inp[~cur_mask] = decoder_labels[~cur_mask]  # place true labels at unmasked positions

                _was_training = self.training
                self.eval()
                with torch.no_grad():
                    if B == 1:  # print only for single sample to avoid multi-worker spam
                        print(f"[GUIDED] scoring: self.training={self.training}")  # should be False here
                    logits, _ = self.forward_decoder_only(
                        decoder_input_ids=cur_inp,
                        encoder_hidden=encoder_hidden,
                        mask_positions=cur_mask.float(),
                        digit=None, use_cache=False
                    )  # [B, n_digit, K]
                if _was_training:
                    self.train()

                # Compute confidence (same as inference)
                probs = F.softmax(logits, dim=-1)
                if self.guided_conf_metric == 'entropy':
                    ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                    conf = -ent
                else:  # 'msp'
                    conf = probs.max(dim=-1).values  # note: use max(...).values here

                return conf  # [B, n_digit]

            refresh = str(self.config.get('guided_refresh_each_step', False)).lower() in ('1','true','yes','y')
            all_masked_input_ids, all_labels, all_mask_positions, all_encoder_hidden = [], [], [], []

            if not refresh:
                # ------- One-time sorting, no refresh -------
                full_mask = torch.ones(B, self.n_digit, dtype=torch.bool, device=device)
                conf = score_with_mask(full_mask)  # score with full mask to obtain ranking
                if self.guided_select == 'most':
                    order = torch.argsort(conf, 1, True)
                else:
                    order = torch.argsort(conf, 1, False)

                for t in range(1, self.guided_steps + 1):
                    cur_mask = torch.zeros(B, self.n_digit, dtype=torch.bool, device=device)
                    cols = order[:, :t]
                    cur_mask.scatter_(1, cols, True)

                    cur_inp = decoder_input_ids.new_zeros(B, self.n_digit)
                    cur_inp[~cur_mask] = decoder_labels[~cur_mask]

                    all_masked_input_ids.append(cur_inp)
                    all_labels.append(decoder_labels)
                    all_mask_positions.append(cur_mask.float())
                    all_encoder_hidden.append(encoder_hidden)
            else:
                # ------- Refresh each step -------
                cur_mask = torch.zeros(B, self.n_digit, dtype=torch.bool, device=device)
                for t in range(1, self.guided_steps + 1):
                    conf = score_with_mask(cur_mask)  # confidence for this step

                    # Columns already masked are not eligible
                    if self.guided_select == 'most':
                        conf = conf.masked_fill(cur_mask, -1e9)
                        cols = torch.argmax(conf, dim=1, keepdim=True)  # pick 1 column per sample
                    else:
                        conf = conf.masked_fill(cur_mask,  1e9)
                        cols = torch.argmin(conf, dim=1, keepdim=True)

                    cur_mask.scatter_(1, cols, True)

                    cur_inp = decoder_input_ids.new_zeros(B, self.n_digit)
                    cur_inp[~cur_mask] = decoder_labels[~cur_mask]

                    all_masked_input_ids.append(cur_inp)
                    all_labels.append(decoder_labels)
                    all_mask_positions.append(cur_mask.float())
                    all_encoder_hidden.append(encoder_hidden)
        
        else:
            # Legacy random mask branch (keep original logic)
            # LLaDA style: if mask_prob_random enabled, sample one mask probability per batch
            batch_mask_prob = None
            if getattr(self, 'mask_prob_random', False):
                low = float(self.config.get('mask_prob_random_min', 0.0))
                high = float(self.config.get('mask_prob_random_max', 1.0))
                # Use torch sampling for consistency with the global RNG seed
                batch_mask_prob = float(torch.empty(1).uniform_(low, high).item())
            for view_idx, mask_prob in enumerate(self.mask_probs):
                if batch_mask_prob is not None:
                    mask_prob = batch_mask_prob
                # Generate mask for the current mask probability
                mask_positions = torch.rand(B, self.n_digit, device=device) < mask_prob  # [B, n_digit]
                
                # Ensure each sample has at least one masked position
                no_mask_samples = ~mask_positions.any(dim=1)  # [B]
                if no_mask_samples.any():
                    # For samples without any mask, force-mask the first position
                    mask_positions[no_mask_samples, 0] = True
                
                # Apply mask: set masked positions to 0
                masked_input_ids = decoder_input_ids.clone()  # [B, n_digit]
                masked_input_ids[mask_positions] = 0
                
                # Store current view data
                all_masked_input_ids.append(masked_input_ids)
                all_labels.append(decoder_labels)  # labels remain unchanged
                all_mask_positions.append(mask_positions.float())
                all_encoder_hidden.append(encoder_hidden)  # each view uses the same encoder output
        
        # Merge all views: [B*n_views, ...]
        decoder_input_ids = torch.cat(all_masked_input_ids, dim=0)  # [B*n_views, n_digit]
        decoder_labels = torch.cat(all_labels, dim=0)  # [B*n_views, n_digit]
        mask_positions = torch.cat(all_mask_positions, dim=0)  # [B*n_views, n_digit]
        encoder_hidden = torch.cat(all_encoder_hidden, dim=0)  # [B*n_views, seq_len*n_digit, emb_dim]
        
        # Update batch size and sanity-check shapes
        B_expanded = B * self.augment_factor
        
        # Shape verification
        assert decoder_input_ids.shape[0] == B_expanded, f"decoder_input_ids shape mismatch: {decoder_input_ids.shape[0]} vs {B_expanded}"
        assert decoder_labels.shape[0] == B_expanded, f"decoder_labels shape mismatch: {decoder_labels.shape[0]} vs {B_expanded}"
        assert mask_positions.shape[0] == B_expanded, f"mask_positions shape mismatch: {mask_positions.shape[0]} vs {B_expanded}"
        assert encoder_hidden.shape[0] == B_expanded, f"encoder_hidden shape mismatch: {encoder_hidden.shape[0]} vs {B_expanded}"
        
        # Consistency check: guided views should monotonically increase masked count
        if self.masking_strategy == 'guided':
            m = mask_positions.view(B, self.augment_factor, self.n_digit).sum(-1)  # [B, 4]
            assert torch.all(m[:, 1:] >= m[:, :-1]), "guided views should increase masked count monotonically"
        
        # --- Decoder (training mode) ---

        logits, _ = self.forward_decoder_only(
            decoder_input_ids=decoder_input_ids,
            encoder_hidden=encoder_hidden,
            mask_positions=mask_positions,
            past_key_values=None,
            digit=None, use_cache=False
        )

        # Compute loss
        if self.masking_strategy == 'random' and getattr(self, 'mask_prob_random', False):
            # LLaDA style: aggregate per-sample loss across masked positions then multiply by 1/t
            # to suppress scaling differences caused by varying mask rates. Here t is the
            # actual mask rate per sample (not the sampled parameter), avoiding huge weights
            # when t is extremely small.
            per_sample_loss = torch.zeros(B_expanded, device=device)
            for d in range(self.n_digit):
                logits_d = logits[:, d, :]  # [B_expanded, codebook_size]
                labels_d = decoder_labels[:, d]
                mask_d = mask_positions[:, d].float()
                loss_d = F.cross_entropy(
                    logits_d, labels_d, reduction='none',
                    label_smoothing=self.config.get('label_smoothing', 0.1)
                )
                per_sample_loss += loss_d * mask_d  # only count masked positions
            # actual mask rate t_i: proportion of masked tokens per sample
            t_actual = mask_positions.float().mean(dim=1)  # [B_expanded]
            t_actual = torch.clamp(t_actual, min=1e-6)
            total_loss = (per_sample_loss / t_actual).mean()  # average across the batch
        else:
            # Original logic: compute loss only on masked positions and average by number of masked tokens
            total_loss = 0.0
            total_weight = 0.0
            for d in range(self.n_digit):
                logits_d = logits[:, d, :]  # [B_expanded, codebook_size]
                labels_d = decoder_labels[:, d]
                mask_d = mask_positions[:, d].float()
                loss_d = F.cross_entropy(
                    logits_d, labels_d, reduction='none',
                    label_smoothing=self.config.get('label_smoothing', 0.1)
                )
                total_loss += (loss_d * mask_d).sum()
                total_weight += mask_d.sum()
            if total_weight > 0:
                total_loss = total_loss / total_weight
            else:
                total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss

    def encode(self, batch):
        return self.encoder(batch)
    
    def decode(self, batch, digit=None, past_key_values=None, use_cache=False):
        device = next(self.parameters()).device

        return self.forward_decoder_only(
            decoder_input_ids=batch['decoder_input_ids'].to(device),
            encoder_hidden=batch['encoder_hidden'].to(device),
            mask_positions=batch['mask_positions'].to(device) if batch.get('mask_positions', None) is not None else None,
            digit=digit, past_key_values=past_key_values, use_cache=use_cache)
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import math
import numpy as np
import torch
import torch.nn.functional as F
#from hdbscan import HDBSCAN
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from torch import nn

from knowledge_graph.set_utils import load_bitmap_strings

#from knowledge_graph.ner import apply_filter
#from knowledge_graph.set_utils import load_bitmaps, load_bitmap_strings, load_mapping, jaccard_dist, \
#    k_medoids, write_records

logger = logging.getLogger(__name__)


class Router:
    def get_experts(self, batch: torch.Tensor) -> List[int]:
        return [0] * len(batch)
    def set_current_ids(self, args):
        pass


class GradientRouter(Router):
    def __init__(self, ema_decay: float = 0.9, proj_dim: int = 128):
        self.ema_decay = ema_decay
        self.global_ema_gradient = None  # Dict[str, Tensor]
        self.gradient_history = {}       # Dict[sample_id, Dict[str, Tensor]]
        self.current_ids = []
        self.proj_dim = proj_dim
        self.proj_matrices = {}

    def init_proj_matrix(self, full_gradient_dim, device=torch.device('cpu')):
        self.proj_matrices[full_gradient_dim] = torch.randn(full_gradient_dim, self.proj_dim, device=device) / (self.proj_dim ** 0.5)

    def set_current_ids(self, current_ids):
        self.current_ids = current_ids

    def flatten_and_project(self, gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
        gradient_list = []
        for gradient in gradients.values():
            batch = gradient.shape[0]
            flat = gradient.detach().view(batch, -1)
            grad_dim = flat.shape[1]
            if grad_dim not in self.proj_matrices:
                self.init_proj_matrix(grad_dim, gradient.device)
            gradient_list.append(flat @ self.proj_matrices[grad_dim])
        return torch.cat(gradient_list, dim=1)

    def update(self, current_gradients: Dict[str, torch.Tensor]):
        # Update global EMA
        batch_proj_gradients = self.flatten_and_project(current_gradients)

        batch_mean_proj_grad = batch_proj_gradients.mean(dim=0)

        if self.global_ema_gradient is None:
            self.global_ema_gradient = batch_mean_proj_grad
        else:
            self.global_ema_gradient.mul_(self.ema_decay).add_(batch_mean_proj_grad, alpha=(1 - self.ema_decay))

        # Update sample EMA
        for idx, sid in enumerate(self.current_ids):
            sample_proj_grad = batch_proj_gradients[idx]

            if sid not in self.gradient_history:
                self.gradient_history[sid] = sample_proj_grad
            else:
                self.gradient_history[sid].mul_(self.ema_decay).add_(
                    sample_proj_grad, alpha=(1 - self.ema_decay)
                )

    def get_experts(self, batch: torch.Tensor) -> List[int]:
        expert_indices = []
        similarities = []

        if self.global_ema_gradient is None:
            return [0] * len(batch)

        global_ema = self.global_ema_gradient
        global_ema = global_ema / (global_ema.norm() + 1e-8)

        for sid in self.current_ids:
            sample_ema = self.gradient_history.get(sid, global_ema)
            sample_ema = sample_ema / (sample_ema.norm() + 1e-8)

            similarity = F.cosine_similarity(global_ema.unsqueeze(0), sample_ema.unsqueeze(0)).item()

            prob_noisy = 0.5 * (1 - similarity)
            expert_idx = 1 if random.random() < prob_noisy else 0

            similarities.append(similarity)
            expert_indices.append(expert_idx)


        print(f"sims: {similarities}")
        print(f"routings: {expert_indices}")

        return expert_indices



class ClusterRouter(Router):
    def __init__(self, d, d_reduced, alpha=0.9, warmup_steps=1000, init_sample_count=5000):
        self.d = d
        self.d_reduced = d_reduced
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.init_sample_count = init_sample_count
        self.M = np.random.randn(d_reduced, d)  # Gaussian projection matrix
        self.step = 0
        self.initialized = False
        self.embeddings_buffer = []

        self.centers = []
        self.radii = []

    def _project(self, v):
        return self.M @ v

    def _initialize_clusters(self):
        print("Initializing clusters with DBSCAN...")
        projected = np.array([self._project(v) for v in self.embeddings_buffer])
        #clustering = HDBSCAN(metric="cosine", min_samples=10, eps=1).fit(projected)
        clustering = None
        labels = clustering.labels_
        unique_labels = set(labels) - {-1}  # remove noise
        self.centers = []
        self.radii = []

        for lbl in unique_labels:
            cluster_points = projected[labels == lbl]
            center = np.mean(cluster_points, axis=0)
            radius = np.mean(np.linalg.norm(cluster_points - center, axis=1))
            self.centers.append(center)
            self.radii.append(radius)

        self.centers = np.array(self.centers)
        self.radii = np.array(self.radii)
        self.initialized = True
        print(f"Initialized {len(self.centers)} clusters.")

    def _get_experts_helper(self, v):
        self.step += 1

        if self.step <= self.warmup_steps:
            self.embeddings_buffer.append(v)
            return None

        if not self.initialized and len(self.embeddings_buffer) >= self.init_sample_count:
            self._initialize_clusters()

        if not self.initialized:
            self.embeddings_buffer.append(v)
            return None

        v_proj = self._project(v)
        distances = np.linalg.norm(self.centers - v_proj, axis=1) / self.radii
        expert_idx = np.argmin(distances)
        self.centers[expert_idx] = self.alpha * self.centers[expert_idx] + (1 - self.alpha) * v_proj

        return expert_idx

    def get_experts(self, batch):
        batch_size = batch.shape[0]
        expert_indices = np.full(batch_size, -1, dtype=int)

        for i in range(batch_size):
            expert_indices[i] = self._get_experts_helper(batch[i])

        return expert_indices





class TextClusterRouter:
    def __init__(self, d_text, d_reduced, alpha=0.9):
        self.embeddings = None
        self.embedding_label_lookup = None
        self.d = d_text
        self.d_reduced = d_reduced
        self.alpha = alpha
        self.projection_matrix = torch.randn(d_reduced, d_text)
        self.initialized = False
        self.embedding_label_lookup = None
        self.expert_assignments = None  # Shape: [batch_size]
        self.centers = []
        self.radii = []
        self.labels = None

    def set_routing(self, text_embeddings: torch.Tensor):
        """
        Perform clustering over fixed text embeddings and assign experts.

        :param text_embeddings: Tensor of shape [B, d_text], assumed detached from gradient flow
        """
        if text_embeddings.ndim != 2:
            text_embeddings = torch.mean(text_embeddings, dim=1)

        projected_full = torch.matmul(text_embeddings, self.projection_matrix.T.to(text_embeddings.device))
        projected_full = F.normalize(projected_full).cpu().numpy()



        logger.info("start clustering...")
        clustering = DBSCAN(eps=0.5, min_samples=10).fit(projected_full)
        logger.info("clustering finished")

        self.labels = clustering.labels_
        unique_labels = sorted(set(self.labels))

        if not unique_labels:
            raise RuntimeError("DBSCAN found no clusters.")

        # Compute cluster centers and radii
        self.centers = []
        self.radii = []

        for label in unique_labels:
            cluster_points = projected_full[self.labels == label]
            center = np.mean(cluster_points, axis=0)
            radius = np.mean(np.linalg.norm(cluster_points - center, axis=1))
            self.centers.append(center)
            self.radii.append(radius)

        self.centers = np.array(self.centers)
        self.radii = np.array(self.radii) + 1e-8  # Avoid division by zero


        self.embeddings = text_embeddings
        self.num_experts = len(self.centers)
        self.initialized = True
        print(f"ClusterRouter initialized with {self.num_experts} experts.")

    def get_experts(self, _):
        '''
        embeddings = F.normalize(batch, dim=1)  # (B, D)
        centers_norm = F.normalize(torch.from_numpy(self.centers), dim=1)  # (K, D)
        cosine_dist = 1 - (embeddings @ centers_norm.T.to(embeddings.device))
        normed_dist = cosine_dist / torch.stack(self.radii)
        expert_indices = torch.argmin(normed_dist, dim=1)
        '''
        expert_indices = [self.embedding_label_lookup[current_id] for current_id in self.current_ids]
        return expert_indices


    def set_current_ids(self, current_ids):
        self.current_ids = current_ids

    def create_lookup(self, ids):
        self.embedding_label_lookup = dict(zip(ids, self.labels))




'''
class SetRouter:
    def __init__(self, bitmaps_path, id2cui_path, cui2sty_path, medoid_path, label_path, ids, fill_empty=False):
        self.filter = {"T047": 1, "T046": 1, "T033": 1, "T019": 0, "T037": 0,
                       "T017": 0, "T023": 0, "T029": 0, "T030": 0, "T082": 0}
        self.fill_empty = fill_empty
        self.current_expert_indices = None
        self.bitmap_lookup = None
        self.current_ids = None
        self.bitmaps = load_bitmaps(bitmaps_path)
        self.id2cui = load_bitmap_strings(id2cui_path)
        self.cui2sty = load_mapping(cui2sty_path)
        if Path(label_path).is_file():
            self.labels = load_bitmap_strings(label_path)
        else:
            self.labels = None
        self.id2sty = {id: self.cui2sty[cui] for id, cui in self.id2cui.items() if cui in self.cui2sty}
        self.bitmap_lookup = self._create_lookup(ids)
        self.bitmaps, self.group2index, self.index2group = self._deduplicate(self.bitmaps)
        if Path(medoid_path).is_file():
            self.medoids = load_bitmaps(medoid_path)
            logger.info(f"Medoids loaded from {medoid_path} - {len(self.medoids)} medoids found.")
        else:
            logger.warning(f"Medoids not found. Creating new medoids.")
            logger.info(f"Precomputing routing clusters...")
            self.labels, self.medoids = self._set_routing()
            Path(label_path).write_text("\n".join([str(label) for label in self.labels]), encoding="utf-8")
            logger.info(f"Wrote labels to {label_path}")
            with Path(medoid_path).open("wb") as medoid_fh:
                write_records(medoid_fh, [self.bitmaps[medoid] for medoid in self.medoids])
            logger.info(f"Wrote medoids to {medoid_path}")
            logger.info(f"Routing finished.")
        if self.medoids is not None:
            self.num_experts = len(self.medoids)
        else:
            self.num_experts = 1

    def set_current_expert_indices(self, expert_indices, device):
        self.current_expert_indices = torch.as_tensor(expert_indices, device=device)

    def _set_routing(self):
        for i, bitmap in enumerate(self.bitmaps):
            self.bitmaps[i] = apply_filter(self.id2sty, bitmap,  self.filter)


        labels, medoids = k_medoids(
            self.bitmaps,
            k=3,
            dist_fn=jaccard_dist,
        )
        return labels, medoids

    def _create_lookup(self, ids):
        return dict(zip(ids, self.bitmaps))

    def get_experts(self, _=None):
        bitmaps = [self.bitmap_lookup[current_id] for current_id in self.current_ids]
        num_experts = len(self.medoids)

        expert_indices = [None] * len(bitmaps)
        per_expert = [[] for _ in range(num_experts)]

        for i, bm in enumerate(bitmaps):
            dists = [jaccard_dist(bm, medoid) for medoid in self.medoids]
            closest_expert = min(range(num_experts), key=lambda k: dists[k])
            expert_indices[i] = closest_expert
            per_expert[closest_expert].append(i)

        if self.fill_empty:
            def largest_expert_idx():
                return max(range(num_experts), key=lambda k: len(per_expert[k]))

            empty_experts = [k for k in range(num_experts) if len(per_expert[k]) == 0]

            for empty_expert in empty_experts:
                donor = largest_expert_idx()
                if len(per_expert[donor]) == 0:
                    break

                target_medoid = self.medoids[empty_expert]
                donor_indices = per_expert[donor]
                best_i = min(donor_indices, key=lambda i: jaccard_dist(bitmaps[i], target_medoid))

                per_expert[donor].remove(best_i)
                per_expert[empty_expert].append(best_i)
                expert_indices[best_i] = empty_expert
        return expert_indices

    def set_current_ids(self, current_ids):
        self.current_ids = current_ids

    def _deduplicate(self, bitmaps):
        key_to_group = {}
        unique, group2index, index2group = [], [], [None] * len(bitmaps)

        for i, bm in enumerate(bitmaps):
            group = key_to_group.get(bm)
            if group is None:
                group = len(unique)
                key_to_group[bm] = group
                unique.append(bm)
                group2index.append([i])
            else:
                group2index[group].append(i)
            index2group[i] = group
        return unique, group2index, index2group

    def visualize_clusters(self, n_sample=2500, seed=0, method="mds", show_silhouette=True):
        """
        Visualize set clusters via a 2D embedding of the Jaccard distance matrix.

        Parameters
        ----------
        n_sample : int
            Max number of points to visualize (computes O(n^2) distances).
        seed : int
            RNG seed for the sampling and embedding.
        method : {"mds", "tsne", "umap"}
            Embedding method. Falls back to MDS if others unavailable.
        show_silhouette : bool
            If True, compute/print silhouette on the sampled set (precomputed distances).
        """
        rng = np.random.default_rng(seed)

        X = self.bitmaps
        y = np.asarray(self.labels)
        N = len(X)
        if N == 0:
            raise ValueError("No bitmaps available.")

        # --- sample indices to keep runtime/memory reasonable ---
        m = min(n_sample, N)
        idx = rng.choice(N, size=m, replace=False)
        Xs = [X[i] for i in idx]
        ys = y[idx]

        # --- pairwise Jaccard distance matrix (symmetric) ---
        D = np.zeros((m, m), dtype=np.float32)
        for i in range(m):
            D[i, i] = 0.0
            xi = Xs[i]
            for j in range(i + 1, m):
                d = jaccard_dist(xi, Xs[j])  # uses your existing distance
                D[i, j] = d
                D[j, i] = d

        # --- 2D embedding ---
        coords = None
        if method == "umap":
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=2, metric="precomputed", random_state=seed, n_neighbors=15, min_dist=0.1
                )
                coords = reducer.fit_transform(D)
            except Exception:
                method = "mds"  # fallback
        if method == "tsne" and coords is None:
            try:
                from sklearn.manifold import TSNE
                tsne = TSNE(
                    n_components=2, metric="precomputed", random_state=seed,
                    init="random", perplexity=max(5, min(50, m // 50))
                )
                coords = tsne.fit_transform(D)
            except Exception:
                method = "mds"  # fallback
        if coords is None:
            from sklearn.manifold import MDS
            mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed, n_init=1, max_iter=300)
            coords = mds.fit_transform(D)

        # --- plot ---
        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=ys, s=10, alpha=0.85)
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Cluster label")
        ax.set_title(f"Set clustering ({method.upper()} on Jaccard distances), n={m}")

        # mark medoids (only those that fell in the sample)
        if getattr(self, "medoids", None) is not None:
            idx_set = {int(i) for i in idx}
            pos = {int(i): k for k, i in enumerate(idx)}
            medoids_in_sample = [mi for mi in self.medoids if int(mi) in idx_set]
            for mi in medoids_in_sample:
                k = pos[int(mi)]
                ax.scatter(coords[k, 0], coords[k, 1],
                           marker="*", s=240, edgecolor="k", linewidths=1.0, facecolors="none")
            if medoids_in_sample:
                ax.legend(*ax.get_legend_handles_labels(), loc="best", frameon=True)

        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

        # --- silhouette on the same sample (precomputed distances) ---
        if show_silhouette and len(np.unique(ys)) > 1:
            try:
                from sklearn.metrics import silhouette_score
                sil = silhouette_score(D, ys, metric="precomputed")
                print(f"Silhouette (Jaccard, sample of {m}): {sil:.3f}")
            except Exception as e:
                print(f"Silhouette not available: {e}")

        return fig, ax
        
'''


class LabelRouter:
    def __init__(self, label_path, ids, permute_seed: Optional[int] = None):
        self.labels = load_bitmap_strings(label_path, return_type=int)
        label_values = [int(x) for x in self.labels.values()]
        if permute_seed is not None and int(permute_seed) >= 0:
            rng = random.Random(int(permute_seed))
            rng.shuffle(label_values)
            print(f"LabelRouter: shuffled label assignments with seed={int(permute_seed)}")
        self.label_lookup = self._create_lookup(ids, values=label_values)
        self.label2member_index = defaultdict(list)
        for i, label in enumerate(label_values):
            self.label2member_index[int(label)].append(i)
        self.current_expert_indices = None
        self.current_ids = None
        self.num_experts = len(set(label_values))
        self.test_mode_expert = -1
        print(f"Using LabelRouter with {self.num_experts} experts.")

    def _create_lookup(self, ids, values=None):
        if values is None:
            values = self.labels.values()
        return dict(zip(ids, values))

    def set_current_ids(self, current_ids):
        self.current_ids = current_ids

    def set_current_expert_indices(self, expert_indices, device):
        self.current_expert_indices = torch.as_tensor(expert_indices, device=device)

    def get_experts(self, _=None):
        if self.test_mode_expert < 0:
            labels = [self.label_lookup[current_id] for current_id in self.current_ids]
        else:
            labels = [self.test_mode_expert] * len(self.current_ids)
        return labels

    def test_mode(self, expert):
        self.test_mode_expert = expert


class FakeLabelRouter:
    def __init__(self, label_path, ids, permute_seed: Optional[int] = None):
        self.labels = load_bitmap_strings(label_path, return_type=int)
        label_values = [int(x) for x in self.labels.values()]
        if permute_seed is not None and int(permute_seed) >= 0:
            rng = random.Random(int(permute_seed))
            rng.shuffle(label_values)
            print(f"FakeLabelRouter: shuffled label assignments with seed={int(permute_seed)}")
        self.label_lookup = self._create_lookup(ids, values=label_values)
        self.label2member_index = defaultdict(list)
        for i, label in enumerate(label_values):
            self.label2member_index[int(label)].append(i)
        self.current_expert_indices = None
        self.current_ids = None
        self.num_experts = len(set(label_values))
        print(f"Using FAKELabelRouter with {self.num_experts} experts.")

    def _create_lookup(self, ids, values=None):
        if values is None:
            values = self.labels.values()
        return dict(zip(ids, values))

    def set_current_ids(self, current_ids):
        self.current_ids = current_ids

    def set_current_expert_indices(self, expert_indices, device):
        self.current_expert_indices = torch.as_tensor(expert_indices, device=device)

    def get_experts(self, _=None):
        labels = [self.label_lookup[current_id] for current_id in self.current_ids]
        return labels


import numpy as np
import torch
from collections import defaultdict

class TextLabelRouter:
    def __init__(self, ids, centers_path=None, features_path=None, label_path=None, embeddings=None):
        self.label_lookup = None
        self.id_lookup = self.build_index(ids)

        if centers_path is None and label_path is None:
            raise ValueError("Either centers_path or label_path must be specified.")

        if label_path is None:
            # Center-based routing
            self.centers = np.load(centers_path)["centroids"]                       # (K, D) numpy
            self.num_centers = len(self.centers)
            self.num_experts = self.num_centers
            if features_path is not None:
                self.features = torch.load(features_path, weights_only=False)
            else:
                self.features=embeddings

        else:
            # Label-based routing
            self.labels = load_bitmap_strings(label_path, return_type=int)  # ideally {id -> label}
            self.label_lookup = self._create_lookup(ids)
            self.label2member_index = defaultdict(list)
            # Build reverse index
            if isinstance(self.labels, dict):
                it = self.labels.items()
            else:
                it = enumerate(self.labels)
            for i, label in it:
                self.label2member_index[label].append(i)
            self.num_experts = len(set(self.label_lookup.values()))

        self.current_expert_indices = None  # used as sample indices in center-mode
        self.current_ids = None             # used as sample ids in label-mode

        print(f"Using TextLabelRouter with {self.num_experts} experts.")

    def _create_lookup(self, ids):
        return dict(zip(ids, self.labels.values()))

    def build_index(self, items):
        # Make sure we're iterating Python scalars
        try:  # PyTorch tensor
            items = items.detach().cpu().tolist()
        except AttributeError:
            try:  # NumPy array
                items = items.tolist()
            except AttributeError:
                pass  # already list/tuple or similar

        out = {}
        for i, x in enumerate(items):
            if x in out:
                raise ValueError(f"Duplicate id at position {i}: {x!r}")
            out[x] = i
        return out

    def indices_for(self, index, ids, *, missing="error", default=None):
        """
        index: dict from id -> position (from build_index)
        ids: iterable of ids to look up
        missing: "error" | "ignore" | "default"
        default: value to use when missing == "default"
        """
        out = []
        for k in ids:
            if k in index:
                out.append(index[k])
            elif missing == "error":
                raise KeyError(f"id not found: {k!r}")
            elif missing == "default":
                out.append(default)
            elif missing == "ignore":
                continue
            else:
                raise ValueError("missing must be 'error', 'ignore', or 'default'")
        return out

    def set_current_ids(self, current_ids):
        self.current_ids = current_ids

    def set_current_expert_indices(self, expert_indices, device=None):
        # In center-mode this is used as sample indices into self.features
        if device is None:
            self.current_expert_indices = torch.as_tensor(expert_indices)
        else:
            self.current_expert_indices = torch.as_tensor(expert_indices, device=device)

    def get_experts(self, _=None):
        """
        Returns a list[int] of expert indices for the current batch.
        - Label mode: looks up labels by id.
        - Center mode: assigns to nearest center (Euclidean distance).
        """
        if self.label_lookup is None:
            # ---- Center-based assignment ----
            if self.current_ids is None:
                raise ValueError("current_ids is not set. Call set_current_ids(...) with sample indices.")
            # Grab batch features
            feats = self.features
            if isinstance(feats, np.ndarray):
                feats = torch.from_numpy(feats)
            elif not torch.is_tensor(feats):
                raise TypeError("features must be a tensor or numpy array.")
            current_indices = self.indices_for(self.id_lookup, self.current_ids, missing="error")
            batch_feats = feats[current_indices]  # (B, D)
            if batch_feats.dtype != torch.float32 and batch_feats.dtype != torch.float64:
                batch_feats = batch_feats.float()

            # Prepare centers tensor on same device/dtype
            centers = self.centers
            if isinstance(centers, torch.Tensor):
                centers_t = centers.to(device=batch_feats.device, dtype=batch_feats.dtype)
            else:
                centers_t = torch.from_numpy(centers).to(device=batch_feats.device, dtype=batch_feats.dtype)

            # Compute pairwise Euclidean distances and take argmin
            # (B, K)
            dists = torch.cdist(batch_feats, centers_t)
            labels = torch.argmin(dists, dim=1).tolist()
            return labels
        else:
            # ---- Label-based assignment ----
            if self.current_ids is None:
                raise ValueError("current_ids is not set. Call set_current_ids(...) with sample ids.")
            labels = [self.label_lookup[current_id] for current_id in self.current_ids]
            return labels



class MoCLERouterBasic(nn.Module):
    """
    Faithful/basic MoCLE (Sec. 3.3) without the universal expert (Sec. 3.4) and without load balancing.

    New: centroid-based assignment mode.
      If `pooled_embeddings` is provided (plus `centers_path`), cluster ids are computed
      by nearest-centroid instead of using the lookup from `cluster_assign_path`.

    Gating (paper Eq. (1), simplified):
        G = topk( softmax( (W_gate * C[x]) / tau + eps ) )

    - C[x] ... learnable cluster embedding, shared (init from k-means centroids)
    - eps ~ N(0, 1/E) during training

    Public UX mirrors LabelRouter:
      • pass `ids`
      • call `set_current_ids(...)`
      • call `get_experts()` → returns top-1 expert ids (list[int])
    """

    def __init__(
        self,
        cluster_assign_path: Optional[str],         # can be None in centroid mode
        ids: List[str],                             # global sample ids, order as dataset
        cluster_centers_path: Optional[str],        # optional .npz with key "centroids", shape (K, D)
        *,
        num_experts: Optional[int] = None,          # if None → E = K
        top_k: int = 1,
        temperature: float = 0.05,
        add_noise: bool = True,
        pooled_embeddings: Optional[
            Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]]
        ] = None,                                   # NEW: per-id pooled embeddings
        cluster_embed_dim: int = 64,               # used when no centroid file is provided
        static_init: bool = False,                 # init gate to fixed cluster->expert (GRASP) assignment
    ):
        super().__init__()

        # ---------- 1) Load optional centroids ----------
        centers = None
        K = None
        D = int(cluster_embed_dim)
        if cluster_centers_path is not None:
            centers_npz = np.load(cluster_centers_path)
            if "centroids" not in centers_npz:
                raise KeyError(
                    f"{cluster_centers_path} must contain key 'centroids' with shape (K, D)."
                )
            centers_np = centers_npz["centroids"]  # (K, D)
            centers = torch.from_numpy(centers_np).float()
            K, D = centers.shape
        elif D <= 0:
            raise ValueError("cluster_embed_dim must be > 0 when cluster_centers_path is not provided.")

        # ---------- 2) Mode selection & lookups ----------
        self.ids = list(ids)  # keep a copy
        self.id_to_index = {cid: i for i, cid in enumerate(self.ids)}

        # Lookup mode (legacy)
        self.lookup_mode = pooled_embeddings is None
        if self.lookup_mode:
            if cluster_assign_path is None:
                raise ValueError(
                    "cluster_assign_path is required when pooled_embeddings is None (lookup mode)."
                )
            self.clusters = load_bitmap_strings(cluster_assign_path, return_type=int)
            cluster_values = [int(v) for v in self.clusters.values()]
            if not cluster_values:
                raise ValueError(f"No cluster assignments found in {cluster_assign_path}.")

            if K is None:
                uniq = sorted(set(cluster_values))
                remap = {raw: idx for idx, raw in enumerate(uniq)}
                cluster_values = [remap[v] for v in cluster_values]
                K = len(uniq)
            else:
                min_v, max_v = min(cluster_values), max(cluster_values)
                if min_v < 0 or max_v >= K:
                    uniq = sorted(set(cluster_values))
                    if len(uniq) == K:
                        remap = {raw: idx for idx, raw in enumerate(uniq)}
                        cluster_values = [remap[v] for v in cluster_values]
                        print("MoCLERouterBasic: remapped non-contiguous cluster ids to [0, K-1].")
                    else:
                        raise ValueError(
                            f"Cluster assignments in {cluster_assign_path} are outside [0, {K - 1}] "
                            f"and cannot be remapped to {K} centroids."
                        )

            self.cluster_lookup = dict(zip(ids, cluster_values))
        else:
            # Centroid mode (new)
            if centers is None:
                raise ValueError(
                    "cluster_centers_path is required in centroid mode (when pooled_embeddings is provided)."
                )
            # Normalize pooled_embeddings into a dict[str] -> 1D torch.FloatTensor(D)
            self.embed_store: Dict[str, torch.Tensor] = {}

            def _to_tensor(x) -> torch.Tensor:
                if isinstance(x, torch.Tensor):
                    return x.detach().float()
                return torch.from_numpy(np.asarray(x)).float()

            if isinstance(pooled_embeddings, dict):
                for k, v in pooled_embeddings.items():
                    t = _to_tensor(v).float()
                    if t.ndim != 1:
                        raise ValueError(f"Embedding for id '{k}' must be 1-D, got {t.shape}.")
                    self.embed_store[k] = t
            else:
                # Assume array aligned with ids
                t = _to_tensor(pooled_embeddings).float()
                if t.ndim != 2:
                    raise ValueError(f"pooled_embeddings must be 2-D if not a dict, got {t.shape}.")
                if t.shape[0] != len(ids):
                    raise ValueError(
                        f"pooled_embeddings first dim must equal len(ids) ({len(ids)}), got {t.shape[0]}."
                    )
                # store per id
                for cid, row in zip(ids, t):
                    self.embed_store[cid] = row

            # Basic shape sanity
            any_vec = next(iter(self.embed_store.values()))
            if any_vec.shape[-1] != D:
                raise ValueError(
                    f"Embedding dim ({any_vec.shape[-1]}) does not match centroid dim D={D}."
                )

        if centers is None:
            # No centroid file: initialize learnable cluster embeddings directly.
            centers = torch.randn(K, D, dtype=torch.float32) * 0.02
            if D >= K and K <= 512:
                # Deterministic one-hot init when feasible.
                centers.zero_()
                centers[:, :K] = torch.eye(K, dtype=torch.float32)

        # ---------- 3) Experts & gates ----------
        if num_experts is None:
            num_experts = K
        if int(top_k) < 1:
            raise ValueError("top_k must be >= 1.")

        self.num_experts = num_experts
        self.num_clusters = K
        self.embed_dim = D
        self.top_k = min(int(top_k), int(num_experts))
        self.temperature = temperature
        self.add_noise = add_noise

        # Shared cluster embedding table (init with centroids)
        self.cluster_embed = nn.Embedding(K, D)
        with torch.no_grad():
            self.cluster_embed.weight.copy_(centers)

        # Layer-local gate W_gate
        self.gate = nn.Linear(D, num_experts, bias=False)
        if static_init:
            # Reproduce the fixed cluster->expert (GRASP) assignment at init.
            # With one-hot cluster embeddings, gate(e_c) = W_gate[:, c]; set the diagonal
            # so argmax routes cluster c -> expert c at ~0.9 softmax confidence (after the
            # temperature scaling), leaving room for the gate to drift during finetuning.
            import math
            with torch.no_grad():
                self.gate.weight.zero_()
                alpha = float(temperature) * math.log(9.0 * max(int(num_experts) - 1, 1))
                kk = min(int(K), int(num_experts), int(D))
                for _i in range(kk):
                    self.gate.weight[_i, _i] = alpha
            print(f"MoCLERouterBasic: static (GRASP) gate init on {kk} clusters, "
                  f"alpha={alpha:.4f} (temp={temperature}) -> ~0.9 confidence per cluster.")

        # Keep centroids for nearest-centroid assignment
        self.register_buffer("centers", centers, persistent=False)  # (K, D)

        # runtime state
        self.current_ids: Optional[List[str]] = None
        self.current_expert_indices: Optional[torch.Tensor] = None
        self.current_topk: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.test_mode_expert: int = -1
        self.last_aux_loss: Optional[torch.Tensor] = None

        mode_str = "centroid" if not self.lookup_mode else "lookup"
        print(
            f"Using MoCLERouterBasic ({mode_str} assignment) with {self.num_clusters} clusters "
            f"and {self.num_experts} experts (top_k={self.top_k})."
        )

    # ---------- optional utility to refresh embeddings ----------
    def set_pooled_embeddings(
        self,
        pooled_embeddings: Union[np.ndarray, torch.Tensor, Dict[str, Union[np.ndarray, torch.Tensor]]]
    ):
        """Update/attach pooled embeddings; switches router to centroid mode."""
        self.lookup_mode = False
        self.embed_store = {}

        def _to_tensor(x) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                return x.detach().float()
            return torch.from_numpy(np.asarray(x)).float()

        if isinstance(pooled_embeddings, dict):
            for k, v in pooled_embeddings.items():
                t = _to_tensor(v)
                if t.ndim != 1:
                    raise ValueError(f"Embedding for id '{k}' must be 1-D, got {t.shape}.")
                self.embed_store[k] = t
        else:
            t = _to_tensor(pooled_embeddings)
            if t.ndim != 2:
                raise ValueError(f"pooled_embeddings must be 2-D if not a dict, got {t.shape}.")
            if t.shape[0] != len(self.ids):
                raise ValueError(
                    f"pooled_embeddings first dim must equal len(ids) ({len(self.ids)}), got {t.shape[0]}."
                )
            for cid, row in zip(self.ids, t):
                self.embed_store[cid] = row

        D = self.embed_dim
        any_vec = next(iter(self.embed_store.values()))
        if any_vec.shape[-1] != D:
            raise ValueError(
                f"Embedding dim ({any_vec.shape[-1]}) does not match centroid dim D={D}."
            )

    # ---------- label-like plumbing ----------
    def _create_lookup(self, ids: List[str]):
        # kept for backward compat; not used in centroid mode
        return dict(zip(ids, self.clusters.values()))

    def set_current_ids(self, current_ids: List[str]):
        self.current_ids = current_ids
        self.current_expert_indices = None
        self.current_topk = None

    def set_current_expert_indices(self, expert_indices, device):
        self.current_expert_indices = torch.as_tensor(expert_indices, device=device)

    def test_mode(self, expert: int):
        """Force routing to a single expert (like your LabelRouter)."""
        self.test_mode_expert = expert

    @torch.no_grad()
    def _nearest_centroid_ids(self, ids: List[str], device: torch.device) -> torch.LongTensor:
        if not hasattr(self, "embed_store"):
            raise RuntimeError(
                "No pooled embeddings present. Provide `pooled_embeddings` or call set_pooled_embeddings().")

        embs = []
        for cid in ids:
            if cid not in self.embed_store:
                raise KeyError(f"Missing pooled embedding for id '{cid}'.")
            embs.append(self.embed_store[cid])
        embs = torch.stack(embs, dim=0).to(device)  # (B, D)

        # Ensure identical dtype on both tensors (works with AMP/bfloat16 too)
        target_dtype = torch.float32
        embs = embs.to(dtype=target_dtype)
        centers = self.centers.to(device=device, dtype=target_dtype)

        dists = torch.cdist(embs, centers)  # (B, K) Euclidean
        cluster_ids = torch.argmin(dists, dim=-1)  # (B,)
        return cluster_ids

    # ---------- main forward ----------
    def forward(self, cluster_ids: torch.LongTensor):
        """
        cluster_ids: (B,) long, each entry in [0, K-1]
        returns: dict(logits, probs, topk_idx, topk_w, aux_loss=0.)
        """
        # C[x]: (B, D)
        c = self.cluster_embed(cluster_ids)  # learnable, init with centroids

        logits = self.gate(c)  # (B, E)
        logits = logits / max(self.temperature, 1e-6)

        if self.add_noise and self.training:
            eps = torch.randn_like(logits) / math.sqrt(self.num_experts)
            logits = logits + eps

        probs = F.softmax(logits, dim=-1)  # (B, E)

        if self.top_k < self.num_experts:
            topk_w, topk_idx = probs.topk(self.top_k, dim=-1)  # (B, top_k)
            if self.top_k > 1:
                topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-9)
        else:
            topk_idx = torch.arange(self.num_experts, device=probs.device).unsqueeze(0).expand_as(probs)
            topk_w = probs

        # Load-balancing auxiliary loss (Switch-Transformer form).
        # aux = E * sum_e f_e * P_e, with f_e the fraction of samples top-1 routed to
        # expert e (no gradient) and P_e the mean gate probability for e (gradient).
        # Uniform routing gives aux = 1.0. Disabled unless the training loop scales it
        # by a positive coefficient, so default behaviour is unchanged.
        with torch.no_grad():
            top1_hard = topk_idx[:, 0]
            frac = torch.zeros(self.num_experts, device=probs.device, dtype=probs.dtype)
            frac.scatter_add_(0, top1_hard, torch.ones_like(top1_hard, dtype=probs.dtype))
            frac = frac / max(int(top1_hard.numel()), 1)
        mean_p = probs.mean(dim=0)
        aux_loss = float(self.num_experts) * (frac * mean_p).sum()

        return {
            "logits": logits,
            "probs": probs,
            "topk_idx": topk_idx,
            "topk_w": topk_w,
            "aux_loss": aux_loss,
        }

    # ---------- LabelRouter-like getter ----------
    def get_experts(self):
        """
        Returns top-1 experts as a Python list[int], like your old router.
        Priority:
          1) test_mode_expert
          2) externally set indices
          3) compute from current_ids (via lookup or nearest centroid)
        """
        if self.test_mode_expert >= 0:
            assert self.current_ids is not None, "call set_current_ids(...) first"
            self.current_topk = None
            return [self.test_mode_expert] * len(self.current_ids)

        if self.current_expert_indices is not None:
            return self.current_expert_indices.cpu().tolist()

        if self.current_ids is None:
            raise RuntimeError("current_ids is None; call set_current_ids(...) before get_experts().")

        device = self.cluster_embed.weight.device

        if self.lookup_mode:
            # map ids -> cluster ids via lookup
            cluster_ids = [self.cluster_lookup[cid] for cid in self.current_ids]
            cluster_ids = torch.tensor(cluster_ids, dtype=torch.long, device=device)
        else:
            # centroid mode: assign by nearest centroid from pooled embeddings
            cluster_ids = self._nearest_centroid_ids(self.current_ids, device)

        out = self.forward(cluster_ids)
        top1 = out["topk_idx"][:, 0]
        if self.training and torch.is_grad_enabled():
            self.last_aux_loss = out["aux_loss"]
        else:
            self.last_aux_loss = out["aux_loss"].detach()
        if self.training and torch.is_grad_enabled():
            self.current_topk = (out["topk_idx"], out["topk_w"])
        else:
            self.current_topk = (out["topk_idx"].detach(), out["topk_w"].detach())
        self.current_expert_indices = top1.detach()
        return top1.detach().cpu().tolist()













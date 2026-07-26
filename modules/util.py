import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Sequence, Union, Dict, Iterable, Any
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data

def split_train_test_masks(indices: List[int],
                           train_pct: float,
                           test_pct: float,
                           seed: int | None = None
                          ) -> Tuple[List[int], List[int]]:
    """
    Create train/test masks from a list of indices.

    Args:
        indices: list of dataset indices.
        train_pct: fraction or percent for training (0.8 or 80).
        test_pct: fraction or percent for testing (0.2 or 20).
        seed: optional random seed.

    Returns:
        (train_mask, test_mask) — lists of 0/1 aligned with `indices`.
    """
    if seed is not None:
        random.seed(seed)

    n = len(indices)
    if n == 0:
        return [], []

    total = train_pct + test_pct
    if abs(total - 1.0) > 1e-8 and abs(total - 100.0) > 1e-8:
        raise ValueError("train_pct + test_pct must sum to 1.0 or 100.0")

    # Normalize to fractions
    if abs(total - 100.0) < 1e-8:
        train_frac = train_pct / 100.0
        test_frac = test_pct / 100.0
    else:
        train_frac = train_pct
        test_frac = test_pct

    positions = list(range(n))
    random.shuffle(positions)

    n_train = int(round(train_frac * n))
    n_test = n - n_train  # absorb rounding

    train_pos = set(positions[:n_train])
    test_pos = set(positions[n_train:])

    train_mask = [1 if i in train_pos else 0 for i in range(n)]
    test_mask  = [1 if i in test_pos else 0 for i in range(n)]

    return train_mask, test_mask

def plot_latlon_target(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    target_col: str,
    figsize: tuple = (10, 6),
    cmap: str = "viridis",
    categorical_cmap: str = "tab10",
    point_size: float = 50.0,
    alpha: float = 0.9,
    show_colorbar: bool = True,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Scatter plot of points placed by longitude (x) and latitude (y) colored by target.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame indexed arbitrarily and containing latitude, longitude and target columns.
    lat_col : str
        Name of the latitude column in decimal degrees.
    lon_col : str
        Name of the longitude column in decimal degrees.
    target_col : str
        Column to color points by. Can be numeric or categorical.
    figsize : tuple
        Figure size passed to matplotlib when ax is None.
    cmap : str
        Colormap for numeric targets.
    categorical_cmap : str
        Colormap for categorical targets.
    point_size : float
        Base marker size for points.
    alpha : float
        Marker transparency.
    show_colorbar : bool
        If True and target is numeric, show a colorbar.
    ax : matplotlib.axes.Axes or None
        If provided, draw on this axis; otherwise a new figure/axis is created.
    title : str or None
        Optional plot title.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the plot.
    """
    # Basic validation
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"DataFrame must contain columns '{lat_col}' and '{lon_col}'")
    if target_col not in df.columns:
        raise ValueError(f"DataFrame must contain target column '{target_col}'")

    # Prepare axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Extract arrays
    lats = df[lat_col].to_numpy(dtype=float)
    lons = df[lon_col].to_numpy(dtype=float)
    target = df[target_col]

    # Decide numeric vs categorical
    if np.issubdtype(target.dtype, np.number):
        # Numeric target: continuous colormap
        sc = ax.scatter(
            lons,
            lats,
            c=target.to_numpy(dtype=float),
            cmap=cmap,
            s=point_size,
            alpha=alpha,
            edgecolors="k",
            linewidths=0.2,
            zorder=3,
        )
        if show_colorbar:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(str(target_col))
    else:
        # Categorical target: map categories to colors
        cats, codes = np.unique(target.astype(str), return_inverse=True)
        cmap_obj = plt.get_cmap(categorical_cmap)
        colors = cmap_obj(np.linspace(0, 1, len(cats)))
        node_colors = [colors[c] for c in codes]
        ax.scatter(
            lons,
            lats,
            c=node_colors,
            s=point_size,
            alpha=alpha,
            edgecolors="k",
            linewidths=0.2,
            zorder=3,
        )
        # create a compact legend
        handles = []
        for i, cat in enumerate(cats):
            handles.append(
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=colors[i], markersize=8, markeredgecolor="k")
            )
        ax.legend(handles, cats, title=str(target_col), loc="best", framealpha=0.9)

    # Axis labels and formatting
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    if title is not None:
        ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    return ax

def haversine_distance_matrix(df, lat_col='latitude', lon_col='longitude', as_dataframe=False):
    """
    Compute NxN great-circle distance matrix (meters) for points in df.
    df: pandas.DataFrame with latitude and longitude columns in decimal degrees.
    lat_col, lon_col: column names for lat/lon.
    as_dataframe: if True, return a pandas.DataFrame with same index/columns as df.index.
    Returns: numpy.ndarray shape (N, N) or pandas.DataFrame.
    """
    # Extract arrays and convert to radians
    lat = np.radians(df[lat_col].to_numpy(dtype=float))
    lon = np.radians(df[lon_col].to_numpy(dtype=float))
    # Earth radius in meters
    R = 6371000.0

    # Use broadcasting to compute pairwise differences
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]

    a = np.sin(dlat / 2.0)**2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0)**2
    # Numerical stability: clip a to [0,1]
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    D = R * c  # distances in meters

    if as_dataframe:
        return pd.DataFrame(D, index=df.index, columns=df.index)
    return D

def k_nearest_edges_from_distance_matrix(
    D: np.ndarray,
    k: int,
    epsilon: float = 1e-6,
    labels: Optional[Sequence] = None,
    return_dataframe: bool = True,
    tqdm_desc: str = "Building edges"
) -> Union[pd.DataFrame, list]:
    """
    Build an undirected k-nearest neighbor edge list from a pairwise distance matrix,
    showing progress with tqdm.auto.

    Parameters
    ----------
    D : np.ndarray
        Square (n,n) distance matrix (meters).
    k : int
        Number of nearest neighbors to consider for each node (excluding self).
    epsilon : float
        Small positive constant added to distances when computing weight.
    labels : sequence or None
        Optional labels for nodes (length n). If None, integer indices 0..n-1 are used.
    return_dataframe : bool
        If True, returns a pandas.DataFrame with columns ['u','v','distance_m','weight'].
    tqdm_desc : str
        Description shown in the tqdm progress bar.

    Returns
    -------
    pandas.DataFrame or list
        Undirected edge list (each unordered pair appears once).
    """
    if not isinstance(D, np.ndarray):
        raise TypeError("D must be a numpy.ndarray")
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square (n,n) matrix")
    n = D.shape[0]
    if n == 0:
        return pd.DataFrame(columns=['u','v','distance_m','weight']) if return_dataframe else []
    if k < 1:
        raise ValueError("k must be >= 1")

    # Use integer labels if none provided
    if labels is None:
        labels = list(range(n))
    else:
        if len(labels) != n:
            raise ValueError("labels length must match matrix size")

    # Work on a copy to avoid mutating input
    Dcopy = D.astype(float, copy=True)

    # Ensure self-distances are not selected
    np.fill_diagonal(Dcopy, np.inf)

    # Effective k (can't exceed n-1)
    k_eff = min(k, n - 1)

    # For each row, find indices of k_eff nearest neighbors
    idx_part = np.argpartition(Dcopy, kth=k_eff, axis=1)[:, :k_eff]
    row_idx = np.arange(n)[:, None]
    order_within = np.argsort(Dcopy[row_idx, idx_part], axis=1)
    neighbors_idx = idx_part[row_idx, order_within]

    # Build undirected edge dictionary keyed by ordered label pair
    edges = {}
    # Wrap outer loop with tqdm for progress
    for i in tqdm(range(n), desc=tqdm_desc, unit="rows"):
        u = labels[i]
        for j in neighbors_idx[i]:
            v = labels[j]
            if u == v:
                continue
            # order labels deterministically using string comparison to handle mixed types
            key = (u, v) if (str(u) <= str(v)) else (v, u)
            dist = float(Dcopy[i, j])
            if not np.isfinite(dist):
                continue
            # keep the smallest distance if duplicate encountered
            if key not in edges or dist < edges[key]:
                edges[key] = dist

    # Convert to list with weights
    edge_list = [(u, v, d, 100.0 / (d + epsilon)) for (u, v), d in edges.items()]

    if return_dataframe:
        return pd.DataFrame(edge_list, columns=['u', 'v', 'distance_m', 'weight'])
    return edge_list

def plot_graph_by_component(
    G: nx.Graph,
    coords_df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    edge_weight_attr: Optional[str] = "weight",
    edge_width_scale: float = 1.0,
    show_labels: bool = False,
    figsize: tuple = (10, 6),
    ax: Optional[plt.Axes] = None,
    cmap: str = "tab20",
    node_size_base: float = 150.0,
    node_alpha: float = 0.9,
    edge_alpha: float = 0.7,
) -> plt.Axes:
    """
    Draw a NetworkX graph with nodes placed by (longitude, latitude),
    color nodes by connected component, no legend, and draw smaller components on top.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
    coords_df : pandas.DataFrame (indexed by node labels) with lat/lon columns
    edge_weight_attr : str or None
        If provided, scale edge widths by this attribute; if None, constant width.
    Returns
    -------
    matplotlib.axes.Axes
    """
    # Validate coords_df
    if lat_col not in coords_df.columns or lon_col not in coords_df.columns:
        raise ValueError(f"coords_df must contain columns '{lat_col}' and '{lon_col}'")

    # Build position dict: node -> (x=lon, y=lat)
    pos = {}
    missing_nodes = []
    for n in G.nodes():
        if n not in coords_df.index:
            missing_nodes.append(n)
        else:
            lat = float(coords_df.at[n, lat_col])
            lon = float(coords_df.at[n, lon_col])
            pos[n] = (lon, lat)

    if missing_nodes:
        raise KeyError(f"The following graph nodes are missing from coords_df index: {missing_nodes}")

    # Prepare axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Determine connected components (weakly for directed graphs)
    if G.is_directed():
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))

    # Map node -> component id (original index in components list)
    comp_of = {}
    for cid, comp in enumerate(components):
        for n in comp:
            comp_of[n] = cid

    # Sort components by size descending (largest first). We'll draw largest first,
    # so smaller components (drawn later) appear above larger ones.
    comps_sorted = sorted(components, key=lambda c: len(c), reverse=True)

    n_comps = max(1, len(components))
    cmap_obj = plt.get_cmap(cmap)
    color_vals = cmap_obj(np.linspace(0, 1, n_comps))
    comp_color_map = {cid: color_vals[cid] for cid in range(n_comps)}

    # Node sizes: base scaled by degree
    node_list = list(G.nodes())
    degrees = np.array([G.degree(n) for n in node_list], dtype=float)
    deg_min = float(degrees.min()) if degrees.size > 0 else 0.0
    deg_max = float(degrees.max()) if degrees.size > 0 else 0.0
    deg_range = deg_max - deg_min
    if deg_range == 0.0:
        sizes = np.full_like(degrees, fill_value=node_size_base)
    else:
        sizes = node_size_base * (1.0 + (degrees - deg_min) / deg_range)
    node_size_map = dict(zip(node_list, sizes))

    # Edge widths (optional)
    widths = None
    if edge_weight_attr is not None:
        weights = []
        edgelist = list(G.edges(data=True))
        for u, v, data in edgelist:
            w = data.get(edge_weight_attr, 1.0)
            try:
                weights.append(float(w))
            except Exception:
                weights.append(1.0)
        weights = np.array(weights, dtype=float) if len(weights) > 0 else np.array([], dtype=float)
        if weights.size > 0:
            w_min = float(weights.min())
            w_max = float(weights.max())
            if w_max == w_min:
                widths = np.full_like(weights, fill_value=1.0) * edge_width_scale
            else:
                widths = (0.5 + 4.5 * (weights - w_min) / (w_max - w_min)) * edge_width_scale
    else:
        # build edgelist for later use (constant width)
        edgelist = list(G.edges())

    # Draw all edges first (underneath nodes). Do not pass zorder to draw_networkx_edges.
    if edge_weight_attr is not None and widths is not None:
        # draw edges with widths aligned to edgelist order
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v, _ in edgelist],
                               width=list(widths), alpha=edge_alpha, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=edge_alpha)

    # Draw nodes component-by-component in sorted order (largest -> smallest).
    # Because we draw largest first and smallest last, nodes in smaller CCs appear above nodes in larger CCs.
    for draw_order_idx, comp in enumerate(comps_sorted):
        comp_nodes = list(comp)
        # color for this component (use original component id)
        orig_cid = comp_of[comp_nodes[0]]
        color = comp_color_map[orig_cid]
        xs = [pos[n][0] for n in comp_nodes]
        ys = [pos[n][1] for n in comp_nodes]
        node_sizes_list = [node_size_map.get(n, node_size_base) for n in comp_nodes]
        # draw with scatter so each component can have its own color and zorder
        ax.scatter(
            xs,
            ys,
            s=node_sizes_list,
            c=[color],
            alpha=node_alpha,
            edgecolors="k",
            linewidths=0.3,
            zorder=2 + draw_order_idx  # later (smaller) components have higher zorder
        )

    # Optionally draw labels (draw after nodes so labels are on top)
    if show_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)

    # Axis formatting
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    return ax

def build_pyg_data_from_networkx(
    G: nx.Graph,
    x: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    test_mask: Optional[Union[Iterable[bool], Iterable[int], Iterable[Any]]] = None,
    edge_attr_name: str = "weight",
    add_reverse_edges: bool = True,
    node_mapping: Optional[Dict[Any, int]] = None,
    device: str = "cpu"
) -> Tuple[Data, Dict[Any, int]]:
    """
    Convert a NetworkX graph to a torch_geometric.data.Data object and return the node mapping.

    - G: networkx Graph or DiGraph (nodes may be 0..N-1 or arbitrary labels).
    - x: node feature tensor [N, F]. If None, a zero tensor [N, 1] is created.
    - y: label tensor [N] aligned to nodes (if provided).
    - test_mask: either
        * boolean iterable length N aligned to node order (True for test nodes),
        * or iterable of node labels/indices to mark as test nodes.
    - edge_attr_name: edge attribute key to use as scalar edge weight (default "weight").
    - add_reverse_edges: if True and G is undirected, add both directions for each edge.
    - node_mapping: optional dict mapping original node labels -> contiguous indices 0..N-1.
    - device: torch device string.
    Returns (data, node_mapping).
    """
    # 1) Build or infer node mapping
    nodes = list(G.nodes())
    if node_mapping is None:
        n = len(nodes)
        if set(nodes) == set(range(n)):
            node_mapping = {i: i for i in range(n)}
        else:
            node_mapping = {orig: i for i, orig in enumerate(nodes)}
    else:
        missing = set(G.nodes()) - set(node_mapping.keys())
        if missing:
            raise ValueError(f"node_mapping missing nodes: {missing}")

    # 2) Relabel graph to contiguous indices
    G_relabeled = nx.relabel_nodes(G, node_mapping, copy=True)

    # 3) Build edge_index and edge_attr
    edges = []
    weights = []
    for u, v, data in G_relabeled.edges(data=True):
        w = data.get(edge_attr_name, 1.0)
        edges.append((u, v))
        weights.append(w)
        if add_reverse_edges and not G_relabeled.is_directed():
            edges.append((v, u))
            weights.append(w)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        edge_attr = torch.tensor(weights, dtype=torch.float, device=device).view(-1, 1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 1), dtype=torch.float, device=device)

    # 4) Node features x
    num_nodes = G_relabeled.number_of_nodes()
    if x is None:
        x = torch.zeros((num_nodes, 1), dtype=torch.float, device=device)
    else:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.shape[0] != num_nodes:
            raise ValueError(f"x has {x.shape[0]} rows but graph has {num_nodes} nodes.")
        x = x.to(device)

    # 5) Labels y
    if y is not None:
        if not isinstance(y, torch.Tensor):
            raise TypeError("y must be a torch.Tensor")
        if y.shape[0] != num_nodes:
            raise ValueError(f"y has {y.shape[0]} rows but graph has {num_nodes} nodes.")
        y = y.to(device)

    # 6) test_mask handling -> produce boolean tensor aligned to node indices
    test_mask_tensor = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if test_mask is not None:
        tm_list = list(test_mask)
        # If boolean list aligned to nodes
        if len(tm_list) == num_nodes and all(isinstance(v, bool) for v in tm_list):
            test_mask_tensor = torch.tensor([bool(v) for v in tm_list], dtype=torch.bool, device=device)
        else:
            # Interpret as iterable of node labels or indices
            for item in tm_list:
                if item in node_mapping:
                    idx = node_mapping[item]
                else:
                    try:
                        idx = int(item)
                    except Exception:
                        raise ValueError(f"test_mask item {item!r} is neither a node label nor an integer index.")
                if idx < 0 or idx >= num_nodes:
                    raise IndexError(f"test index {idx} out of range for {num_nodes} nodes.")
                test_mask_tensor[idx] = True
    else:
        # If no test_mask provided, default to all False (no held-out test)
        test_mask_tensor = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    # 7) train_mask is the complement of test_mask (True for training nodes)
    train_mask_tensor = (~test_mask_tensor).to(torch.bool)

    # 8) Build Data and attach fields
    edge_index = edge_index.long()
    edge_attr = edge_attr.float()
    x = x.float()
    if y is not None:
        y = y.float()

    test_mask_tensor = test_mask_tensor.bool()
    train_mask_tensor = train_mask_tensor.bool()

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        test_mask=test_mask_tensor,
        train_mask=train_mask_tensor
    )
    if y is not None:
        data.y = y

    return data, node_mapping   
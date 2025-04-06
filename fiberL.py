import os
import cv2
import time
import numpy as np
import pandas as pd
import random
import tempfile
import itertools
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm  # For colormap
from matplotlib.colors import Normalize
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from seaborn import boxplot, histplot, kdeplot
from sklearn.decomposition import PCA
from scipy.spatial import distance, KDTree
from scipy.optimize import curve_fit
from skimage import morphology
import streamlit as st
from stqdm import stqdm

def anisotropic_diffusion(img, niter=1, kappa=50, gamma=0.1, voxelspacing=None, option=1):
    r"""
    Edge-preserving, XD Anisotropic diffusion.
 
    To achieve the best effects, the image should be scaled to
    values between 0 and 1 beforehand.
 
 
    Parameters
    ----------
    img : array_like
        Input image (will be cast to numpy.float).
    niter : integer
        Number of iterations.
    kappa : integer
        Conduction coefficient, e.g. 20-100. ``kappa`` controls conduction
        as a function of the gradient. If ``kappa`` is low small intensity
        gradients are able to block conduction and hence diffusion across
        steep edges. A large value reduces the influence of intensity gradients
        on conduction.
    gamma : float
        Controls the speed of diffusion. Pick a value :math:`<= .25` for stability.
    voxelspacing : tuple of floats or array_like
        The distance between adjacent pixels in all img.ndim directions
    option : {1, 2, 3}
        Whether to use the Perona Malik diffusion equation No. 1 or No. 2,
        or Tukey's biweight function.
        Equation 1 favours high contrast edges over low contrast ones, while
        equation 2 favours wide regions over smaller ones. See [1]_ for details.
        Equation 3 preserves sharper boundaries than previous formulations and
        improves the automatic stopping of the diffusion. See [2]_ for details.
 
    Returns
    -------
    anisotropic_diffusion : ndarray
        Diffused image.
 
    Notes
    -----
    Original MATLAB code by Peter Kovesi,
    School of Computer Science & Software Engineering,
    The University of Western Australia,
    pk @ csse uwa edu au,
    <http://www.csse.uwa.edu.au>
 
    Translated to Python and optimised by Alistair Muldal,
    Department of Pharmacology,
    University of Oxford,
    <alistair.muldal@pharm.ox.ac.uk>
 
    Adapted to arbitrary dimensionality and added to the MedPy library Oskar Maier,
    Institute for Medical Informatics,
    Universitaet Luebeck,
    <oskar.maier@googlemail.com>
 
    June 2000  original version. -
    March 2002 corrected diffusion eqn No 2. -
    July 2012 translated to Python -
    August 2013 incorporated into MedPy, arbitrary dimensionality -
 
    References
    ----------
    .. [1] P. Perona and J. Malik.
       Scale-space and edge detection using ansotropic diffusion.
       IEEE Transactions on Pattern Analysis and Machine Intelligence,
       12(7):629-639, July 1990.
    .. [2] M.J. Black, G. Sapiro, D. Marimont, D. Heeger
       Robust anisotropic diffusion.
       IEEE Transactions on Image Processing,
       7(3):421-432, March 1998.
    """
    # define conduction gradients functions
    if option == 1:
 
        def condgradient(delta, spacing):
            return np.exp(-((delta / kappa) ** 2.0)) / float(spacing)
 
    elif option == 2:
 
        def condgradient(delta, spacing):
            return 1.0 / (1.0 + (delta / kappa) ** 2.0) / float(spacing)
 
    elif option == 3:
        kappa_s = kappa * (2**0.5)
 
        def condgradient(delta, spacing):
            top = 0.5 * ((1.0 - (delta / kappa_s) ** 2.0) ** 2.0) / float(spacing)
            return np.where(np.abs(delta) <= kappa_s, top, 0)
 
    # initialize output array
    out = np.array(img, dtype=np.float32, copy=True)
 
    # set default voxel spacing if not supplied
    if voxelspacing is None:
        voxelspacing = tuple([1.0] * img.ndim)
 
    # initialize some internal variables
    deltas = [np.zeros_like(out) for _ in range(out.ndim)]
 
    for _ in range(niter):
        # calculate the diffs
        for i in range(out.ndim):
            slicer = tuple(
                [slice(None, -1) if j == i else slice(None) for j in range(out.ndim)]
            )
            deltas[i][tuple(slicer)] = np.diff(out, axis=i)
 
        # update matrices
        matrices = [
            condgradient(delta, spacing) * delta
            for delta, spacing in zip(deltas, voxelspacing)
        ]
 
        # subtract a copy that has been shifted ('Up/North/West' in 3D case) by one
        # pixel. Don't as questions. just do it. trust me.
        for i in range(out.ndim):
            slicer = tuple(
                [slice(1, None) if j == i else slice(None) for j in range(out.ndim)]
            )
            matrices[i][tuple(slicer)] = np.diff(matrices[i], axis=i)
 
        # update the image
        out += gamma * (np.sum(matrices, axis=0))
 
    return out

def skel_unique(img):
    # Main loop for initial pass at skeletonization
    skel = np.zeros(img.shape, np.uint8)                              # Creates blank canvas to store skeletonized image
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))                  # Generates kernel to pass through each section of image
    while True:
        open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)     # Fit open-cross kernel to thresholded image
        temp = cv2.subtract(img, open_img)                            # Subtracts the above from the original image
        temp = temp.astype(skel.dtype)                                            # Convert temp from 32-bit to 8bit datatype
        skel = cv2.bitwise_or(skel, temp)                                         # aggregates and compares both original and above created image
        img = cv2.erode(img, element)                     # degrades the width of the image by 1 pixel
        if cv2.countNonZero(img) == 0:                                # conditional exits if there are no more remaining iterations to pass through
            break
    return skel

def find_tips(skel_img):
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skel_img, -1, kernel)
    tips = (neighbors == 11).astype(np.uint8) * 255
    return tips

def merge_close_branch_points(branch_pts, threshold=10):
    """
    Merge branch points that are within a close distance of each other.
    """
    if len(branch_pts) < 2:
        return branch_pts

    merged_points = []
    used = set()

    for i, pt1 in enumerate(branch_pts):
        if i in used:
            continue
        group = [pt1]

        for j, pt2 in enumerate(branch_pts):
            if j != i and np.linalg.norm(pt1 - pt2) < threshold:
                group.append(pt2)
                used.add(j)

        # Take the mean of close points to create a single merged point
        merged_point = np.mean(group, axis=0).astype(np.int32)
        merged_points.append(merged_point)

    return np.array(merged_points)

def cos_sim(vectors):
    smallest_similarity = float('inf')
    smallest_pair = None
    
    # Generate all combinations of two vectors
    for (key1, vec1), (key2, vec2) in itertools.combinations(vectors.items(), 2):
        # Compute cosine similarity as the dot product (vectors are normalized)
        similarity = np.dot(vec1[0], vec2[0])
        
        # Check if this is the smallest similarity so far
        if similarity < smallest_similarity:
            smallest_similarity = similarity
            smallest_pair = (key1, key2)
    
    return smallest_similarity, smallest_pair

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x

def sort_by_proximity(contour_points, branch_point):
    """
    Sorts a list of points (from cv2.findContours) starting from the point closest to the branch point.
    Then it orders the remaining points based on nearest-neighbor proximity.

    Parameters:
        contour_points (numpy.ndarray): Array of points of shape (n, 1, 2) as returned by cv2.findContours.
        branch_point (tuple or np.ndarray): The (x, y) coordinate from which to anchor the sorting.

    Returns:
        numpy.ndarray: Sorted array of points of shape (n, 1, 2) with minimal distance between neighbors,
                       starting from the point closest to the branch point.
    """
    # Reshape contour points to (n, 2) for processing if needed.
    if contour_points.ndim == 3:
        contour_points = contour_points.reshape(-1, 2)

    # Remove duplicate points.
    contour_points = np.unique(contour_points, axis=0)

    # Find the point in the contour closest to the branch point.
    distances_to_branch = np.linalg.norm(contour_points - branch_point, axis=1)
    start_index = np.argmin(distances_to_branch)
    sorted_points = [contour_points[start_index]]
    contour_points = np.delete(contour_points, start_index, axis=0)

    # Greedily add the nearest neighbor.
    while len(contour_points) > 0:
        last_point = sorted_points[-1]
        distances = np.linalg.norm(contour_points - last_point, axis=1)
        nearest_idx = np.argmin(distances)
        sorted_points.append(contour_points[nearest_idx])
        contour_points = np.delete(contour_points, nearest_idx, axis=0)

    # Return the sorted points as an array of shape (n, 1, 2)
    return np.array(sorted_points).reshape(-1, 1, 2)

def sort_points_global(points):
    """
    Sort a set of points in a nearest-neighbor path from one end to the other.
    points: (N, 2) array of [x, y].
    """
    # Remove duplicates
    points = np.unique(points, axis=0)

    # Pick an initial point: e.g., leftmost or topmost
    start_idx = np.argmin(points[:, 0])  # leftmost
    sorted_points = [points[start_idx]]
    points = np.delete(points, start_idx, axis=0)

    # Greedily add nearest neighbor
    while len(points) > 0:
        last = sorted_points[-1]
        dists = np.linalg.norm(points - last, axis=1)
        nn_idx = np.argmin(dists)
        sorted_points.append(points[nn_idx])
        points = np.delete(points, nn_idx, axis=0)

    return np.array(sorted_points)

def adaptive_sort(points):
    """
    Sorts an edge dynamically based on its primary orientation.
    - Uses PCA to determine whether to sort by X or Y.
    """
    points = np.unique(points, axis=0)  # Remove duplicates

    if len(points) < 2:
        return points  # Not enough points to sort

    # Perform PCA to find the principal direction
    pca = PCA(n_components=2)
    pca.fit(points)
    principal_direction = pca.components_[0]  # First principal component

    # Determine whether to sort by X or Y based on the principal direction
    if abs(principal_direction[0]) > abs(principal_direction[1]):
        # More horizontal → sort by X
        sorted_points = points[np.argsort(points[:, 0])]
    else:
        # More vertical → sort by Y
        sorted_points = points[np.argsort(points[:, 1])]

    return sorted_points

def greedy_sort(points):
    points = np.unique(points, axis=0)
    if len(points) == 0:
        return points

    ordered = [points[0]]
    points = np.delete(points, 0, axis=0)

    while len(points):
        dists = distance.cdist([ordered[-1]], points)[0]
        next_idx = np.argmin(dists)
        ordered.append(points[next_idx])
        points = np.delete(points, next_idx, axis=0)

    return np.array(ordered)

def pca_curvature_score(points):
    """
    Returns a curvature score: higher = straighter.
    Uses PCA explained variance ratio.
    """
    if len(points) < 5:
        return 0
    pca = PCA(n_components=2)
    pca.fit(points)
    return pca.explained_variance_ratio_[0]  # how much variance is in the primary direction

def unit_tangent_vectors(coords):
    diffs = np.diff(coords, axis=0)
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    return diffs / norms

def tangent_correlation(tangents, max_lag):
    corrs = []
    for lag in range(1, max_lag + 1):
        dots = [
            np.dot(tangents[i], tangents[i + lag])
            for i in range(len(tangents) - lag)
        ]
        corrs.append(np.mean(dots))
    return np.array(corrs)

def exp_decay(s, Lp):
    return np.exp(-s / Lp)

def estimate_persistence_length(coords):
    coords = np.array(coords).reshape(-1, 2)

    if coords.ndim != 2 or coords.shape[0] < 3:
        # Too short or not 2D: return np.nan or 0
        return np.nan

    tangents = unit_tangent_vectors(coords)
    if len(tangents) < 2:
        return np.nan

    max_lag = min(20, len(tangents) - 1)
    if max_lag < 1:
        return np.nan

    corrs = tangent_correlation(tangents, max_lag)

    if np.any(np.isnan(corrs)) or np.all(corrs == 0):
        return np.nan

    # Compute contour distances
    step_sizes = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    avg_step = np.mean(step_sizes)
    s_vals = np.arange(1, max_lag + 1) * avg_step

    try:
        popt, _ = curve_fit(exp_decay, s_vals, corrs, p0=[avg_step * 5])
        return popt[0]  # Lp
    except Exception:
        return np.nan

def mst_sort(points):
    points = np.unique(points, axis=0)
    if len(points) < 2:
        return points

    G = nx.Graph()
    for i, pt1 in enumerate(points):
        for j, pt2 in enumerate(points):
            if i < j:
                dist = np.linalg.norm(pt1 - pt2)
                G.add_edge(i, j, weight=dist)

    mst = nx.minimum_spanning_tree(G)
    path = list(nx.dfs_preorder_nodes(mst, source=0))  # or use longest path?
    return points[path]

class fiberL:
    def __init__(self, image, niter=50, kappa=50, gamma=0.2, option = 3, thresh_1 = 126, 
                 g_blur = 9, thresh_2 = 15, ksize = 5, min_prune = 5, max_node_dist=15, 
                 tip_distance_thresh = 25, cos_thresh = 0.85, curvature_thresh = 0.85, pixels_per_unit=1.0):
        self.image = image
        self.niter = niter
        self.kappa = kappa
        self.gamma=gamma
        self.option=option
        self.thresh_1 = thresh_1
        self.thresh_2 = thresh_2
        self.g_blur = g_blur
        self.ksize = ksize
        self.min_prune = min_prune
        self.max_node_dist = max_node_dist
        self.tip_distance_thresh = tip_distance_thresh
        self.cos_thresh = cos_thresh
        self.curvature_thresh = curvature_thresh
        self.pixels_per_unit = pixels_per_unit

    def preproc(self):
        if self.image is not None:
            start = time.time()
            print("🧼 Preprocessing started...")

            if len(self.image.shape) == 3:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = self.image.copy()

            print("  ⏳ Running anisotropic diffusion...")
            self.diff_img = anisotropic_diffusion(gray_image, niter=self.niter, kappa=self.kappa, gamma=self.gamma, option=self.option)
            self.binary_thresh = cv2.threshold(self.diff_img, self.thresh_1, 255, cv2.THRESH_BINARY)[1].astype('uint8')
            self.binary_canny = cv2.Canny((self.diff_img).astype(np.uint8), 20, 60)
            self.binary_image = cv2.bitwise_or(self.binary_thresh, self.binary_canny)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Try (5, 5) or adjust size
            # Close small gaps
            closed = cv2.morphologyEx(self.binary_image, cv2.MORPH_CLOSE, kernel)
            # Optional: dilate to strengthen weak lines
            closed = cv2.dilate(closed, kernel, iterations=1)
            # Optional: erode to reduce extra thickness
            # closed = cv2.erode(closed, kernel, iterations=1)
            self.binary_image = closed  # update before skeletonization
            self.skel1 = skel_unique(self.binary_image)
            self.blurred_image = cv2.GaussianBlur(self.skel1, (self.g_blur, self.g_blur), 0)
            _, self.final_binary = cv2.threshold(self.blurred_image, self.thresh_2, 255, cv2.THRESH_BINARY)
            self.sk_image = morphology.skeletonize(self.final_binary > 0).astype(np.uint8)

            print(f"✅ Preprocessing completed in {time.time() - start:.2f}s\n")

    def branch(self):
        start = time.time()
        print("🔍 Detecting branches...")

        skel_img = self.sk_image.copy()

        # Define the branch point detection kernels
        t1 = np.array([[-1, 1, -1],
                    [1, 1, 1],
                    [-1, -1, -1]], dtype=np.int8)

        t2 = np.array([[1, -1, 1],
                    [-1, 1, -1],
                    [1, -1, -1]], dtype=np.int8)

        y1 = np.array([[1, -1, 1],
                    [0,  1, 0],
                    [0,  1, 0]], dtype=np.int8)

        y2 = np.array([[-1,  1, -1],
                    [ 1,  1,  0],
                    [-1,  0,  1]], dtype=np.int8)

        plus = np.array([[-1,  1, -1],
                        [ 1,  1,  1],
                        [-1,  1, -1]], dtype=np.int8)

        self.kernels = ([t1, np.rot90(t1, 1), np.rot90(t1, 2), np.rot90(t1, 3)] +
                        [t2, np.rot90(t2, 1), np.rot90(t2, 2), np.rot90(t2, 3)] +
                        [y1, np.rot90(y1, 1), np.rot90(y1, 2), np.rot90(y1, 3)] +
                        [y2, np.rot90(y2, 1), np.rot90(y2, 2), np.rot90(y2, 3)] +
                        [plus])
        
        self.branch_pts_img = np.zeros(skel_img.shape, dtype=np.uint8)

        print("  🧠 Applying kernels...")
        for k in stqdm(self.kernels, desc="  ⏳ Hit-or-Miss Kernels", total=len(self.kernels)):
            hit_miss = cv2.morphologyEx(skel_img, cv2.MORPH_HITMISS, k, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            self.branch_pts_img = cv2.bitwise_or(self.branch_pts_img, hit_miss)

        branch_pts = np.argwhere(self.branch_pts_img != 0)
        branch_pts = np.array([[pt[1], pt[0]] for pt in branch_pts])
        self.branch_pts = merge_close_branch_points(branch_pts, threshold=self.max_node_dist)
        print(f"  📌 Found {len(self.branch_pts)} merged branch points.")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.ksize, self.ksize))
        self.bp = cv2.dilate(self.branch_pts_img, kernel, iterations=1)
        self.detached = cv2.subtract(skel_img, self.bp)

        contours, _ = cv2.findContours(self.detached, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]
        kept_segments = []
        deleted_segments = []

        print("  ✂️ Pruning small segments...")
        for contour in stqdm(contours, desc="  ⏳ Contour Pruning"):
            arc_length = cv2.arcLength(contour, closed=False)
            point_count = len(contour)
            if point_count > self.min_prune and arc_length > self.min_prune:
                kept_segments.append(contour)
            else:
                deleted_segments.append(contour)
                for point in contour:
                    skel_img[point[0][1], point[0][0]] = 0

        tips = find_tips(skel_img)
        edge_objects = [edge for edge in kept_segments if len(edge) >= self.min_prune]
        edge_points_list = [np.array([point[0] for point in edge]) for edge in edge_objects]

        self.skel_img = skel_img
        self.edge_points_list = edge_points_list
        self.tips = tips

        print(f"✅ Branch processing completed in {time.time() - start:.2f}s\n")
    
    def intersection_associate(self, proximity_threshold=10):
        """
        Calculate intersection associations by finding nearby edges for each branch point.

        Args:
            br_pts: List of branch points (x, y).
            edge_points_list: List of edge points for all edges.
            proximity_threshold: Maximum distance for association.

        Returns:
            intersection_associations: A dictionary mapping branch point indices to associated edge indices.
        """
        start = time.time()
        intersection_associations = {}
        # Precompute once: stack all branch points into a (N, 2) array
        branch_pts = np.array(self.branch_pts)
        if len(branch_pts) == 0:
            return
        elif branch_pts.ndim == 1:
            branch_pts = branch_pts.reshape(1, 2)  # if single branch point
        elif branch_pts.shape[-1] != 2:
            raise ValueError("Branch points are not shaped correctly. Expected Nx2.")

        # For each edge (which is (M, 2)), use broadcasting
        for edge_index, edge_points in stqdm(enumerate(self.edge_points_list), total=len(self.edge_points_list), desc="Finding fiber associations about intersections: "):
            edge_points = np.array(edge_points)
            if edge_points.ndim == 1:
                # This handles flat arrays like [x0, y0, x1, y1, ...]
                if edge_points.size % 2 != 0:
                    continue  # skip malformed edge
                edge_points = edge_points.reshape(-1, 2)
            elif edge_points.shape[-1] != 2:
                # Sometimes OpenCV gives shape (N,1,2); flatten it
                edge_points = edge_points.reshape(-1, 2)
            
            dists = np.linalg.norm(edge_points[:, None, :] - branch_pts[None, :, :], axis=2)
            close = np.any(dists <= proximity_threshold, axis=0)  # bool per branch point

            for br_index in np.where(close)[0]:
                if br_index not in intersection_associations:
                    intersection_associations[br_index] = []
                intersection_associations[br_index].append(edge_index)
        self.int_assoc = intersection_associations    
        print(f"✅ Association complete in {time.time() - start:.2f}s")

    def edge_connect(self):
        """
        Calculate edge connections based on cosine similarity to identify the most aligned pair of edges.
        """
        print("🔧 Connecting edge segments...")
        start = time.time()

        branch_connections = {}

        for br_index, associated_edges in stqdm(self.int_assoc.items(), desc="  🔗 Pairing edges at branch points", total=len(self.int_assoc)):
            vectors = {}
            edge_count = len(associated_edges)
            smallest_pair = None
            for edge_index in associated_edges:
                edge_points = sort_by_proximity(self.edge_points_list[edge_index], self.branch_pts[br_index])
                # Check edge length and skip invalid ones
                if len(edge_points) < 5:
                    # print(f"Skipping edge {edge_index} due to insufficient points: length={len(edge_points)}")
                    continue

                last_idx = len(edge_points) - 1
                close_idx = np.argmin(np.linalg.norm(edge_points - self.branch_pts[br_index], axis=2))

                if close_idx == 0:
                    # print("close index is: ", close_idx)
                    v = np.linalg.norm(self.branch_pts[br_index] - edge_points[4])
                    vectors[edge_index] = (self.branch_pts[br_index] - edge_points[4]) / v
                elif close_idx == last_idx:
                    # print("close index is: ", last_idx)
                    v = np.linalg.norm(self.branch_pts[br_index] - edge_points[last_idx - 4])
                    vectors[edge_index] = (self.branch_pts[br_index] - edge_points[last_idx - 4]) / v
                else:
                    near_dist = np.linalg.norm(edge_points[close_idx - 1])
                    far_dist = np.linalg.norm(edge_points[close_idx + 1])
                    # print("near_dist: ", near_dist, "far_dist: ", far_dist)
                    if near_dist < far_dist and close_idx >= 4:
                        v = np.linalg.norm(self.branch_pts[br_index] - edge_points[close_idx - 4])
                        vectors[edge_index] = (self.branch_pts[br_index] - edge_points[close_idx - 4]) / v
                    elif close_idx + 4 < len(edge_points):
                        v = np.linalg.norm(self.branch_pts[br_index] - edge_points[close_idx + 4])
                        vectors[edge_index] = (self.branch_pts[br_index] - edge_points[close_idx + 4]) / v
                # print(vectors[edge_index])
            
            # Find the smallest cosine similarity
            while edge_count > 1:
                _, smallest_pair = cos_sim(vectors)

                if smallest_pair:
                    if br_index not in branch_connections:
                        branch_connections[br_index] = []
                    branch_connections[br_index].append(smallest_pair)

                    # Safely remove both from the dict (if they exist)
                    for edge in smallest_pair:
                        vectors.pop(edge, None)
                    edge_count -= 2
                else:
                    # 🛑 No more pairs to match — break to avoid infinite loop
                    break
        self.branch_connections = branch_connections
        print(f"✅ Edge connection completed in {time.time() - start:.2f}s\n")

    def merge_edges(self):
        if not self.branch_connections:
            self.edges = self.edge_points_list  # Ensure self.edges is always set
            return

        uf = UnionFind()
        # branch_connections[br_index] is a list of pairs like (edgeA, edgeB)
        for pair_list in stqdm(self.branch_connections.values(), desc="  🔗 Merging Pairs", total=len(self.branch_connections.values())):
            for (edge1, edge2) in pair_list:
                # Union the two integer indices
                uf.union(edge1, edge2)

        grouped_edges = {}
        for idx, edge in enumerate(self.edge_points_list):
            root = uf.find(idx)
            if root not in grouped_edges:
                grouped_edges[root] = []
            grouped_edges[root].append(edge)

        new_edge_objects = []
        for group_edges in stqdm(grouped_edges.values(), desc = "Grouping Edges and ironing out points...", total=len(grouped_edges.values())):
            concatenated = group_edges[0]
            for next_edge in group_edges[1:]:
                # Check orientation before concatenating
                if np.linalg.norm(concatenated[-1] - next_edge[0]) > 1:
                    next_edge = np.flip(next_edge, axis=0)
                concatenated = np.vstack((concatenated, next_edge))
            # Choose a reference point, such as the first point in the merged edge
            reference_point = concatenated[0]  
            # Sort the merged edge based on proximity
            sorted_edge = greedy_sort(concatenated).reshape(-1, 1, 2)
            new_edge_objects.append(sorted_edge)
            
        self.edges = new_edge_objects

    def tip_association(self):
        """
        Post-process to merge edge tips that are spatially close and directionally aligned.
        """
        print("🔁 Merging close tips...")
        start = time.time()

        # Find tip coordinates from skeleton tips
        tip_coords = np.column_stack(np.where(self.tips > 0))  # (y, x)
        tip_coords = np.array([[x, y] for y, x in tip_coords])  # to (x, y)

        if len(tip_coords) < 2:
            print("  ⚠️ Not enough tips to process.")
            return

        # Build edge tips array
        start_points = []
        end_points = []
        edge_meta = []
        for edge_idx, edge in stqdm(enumerate(self.edges), desc="Finding edge start and end points...", total=len(self.edges)):
            if len(edge) >= 5:
                tip_start = edge[0, 0]
                tip_end = edge[-1, 0]
                start_points.append(tip_start)
                end_points.append(tip_end)
                edge_meta.append((edge_idx, 'start'))
                edge_meta.append((edge_idx, 'end'))

        tip_endpoints = np.vstack([start_points, end_points])
        tree = KDTree(tip_endpoints)

        dists, indices = tree.query(tip_coords, distance_upper_bound=2.0)

        tip_to_edge = []
        for i, (dist, idx) in stqdm(enumerate(zip(dists, indices)), desc="Associating tip points to edge indices",
                                    total=len(dists)):
            if np.isinf(dist):
                continue
            edge_idx, pos = edge_meta[idx]
            tip_to_edge.append((tuple(tip_coords[i]), edge_idx, pos))

        tip_points = np.array([tp[0] for tp in tip_to_edge])
        if tip_points.ndim != 2 or tip_points.shape[0] < 2:
            print("⚠️ Not enough tip points to build KDTree.")
            return

        tree = KDTree(tip_points)
        used = set()
        paired_edges = set()
        merged_edges = []

        # Precompute orientation and curvature
        orientations = {}
        curvatures = {}
        for idx, edge in stqdm(enumerate(self.edges), desc="Calculating local edge attitude and curvature...",
                               total=len(self.edges)):
            if len(edge) >= 2:
                orientations[idx] = PCA(n_components=1).fit(edge.reshape(-1, 2)).components_[0]
                curvatures[idx] = pca_curvature_score(edge[:, 0])

        for i, (tip1, edge1, pos1) in stqdm(enumerate(tip_to_edge), total=len(tip_to_edge), desc="Mending Fibers on Tip Logic"):
            if i in used:
                continue

            dist, j = tree.query(tip1, k=2)
            if len(dist) < 2 or dist[1] > self.tip_distance_thresh:
                continue

            _, edge2, pos2 = tip_to_edge[j[1]]
            if j[1] in used or edge1 == edge2:
                continue

            # Check orientation + curvature thresholds
            if edge1 not in orientations or edge2 not in orientations:
                continue
            score1 = curvatures[edge1]
            score2 = curvatures[edge2]
            if score1 <= self.curvature_thresh or score2 <= self.curvature_thresh:
                continue
            cos_sim = np.dot(orientations[edge1], orientations[edge2])
            if cos_sim <= self.cos_thresh:
                continue

            # Fetch and orient edges properly
            e1 = self.edges[edge1]
            e2 = self.edges[edge2]
            seg1 = e1[::-1] if pos1 == 'start' else e1
            seg2 = e2 if pos2 == 'start' else e2[::-1]
            combined = np.vstack((seg1, seg2))

            merged_edges.append(combined)
            used.add(i)
            used.add(j[1])
            paired_edges.add(edge1)
            paired_edges.add(edge2)

        # Append unmerged edges
        for edge_idx in range(len(self.edges)):
            if edge_idx not in paired_edges:
                merged_edges.append(self.edges[edge_idx])

        self.edges = [
            mst_sort(edge).reshape(-1, 1, 2) if edge.shape[0] >= 2 else edge.reshape(-1, 1, 2)
            for edge in stqdm(merged_edges, total=len(merged_edges), desc="Sorting final list of edges...")
        ]
        print(f"✅ Tip-to-tip merging done in {time.time() - start:.2f}s — final edge count: {len(self.edges)}")

    def viz_and_sum(self, thickness=2, mode="rand", text=False):
        """
        Visualizes the final results:
        - Top row: Original image, skeletonized image, colored edge network
        - Bottom row: Length distribution, rose diagram (orientation), angle vs length scatter
        """
        # Compute arc lengths for all edges
        self.arc_lengths = [cv2.arcLength(edge, closed=False) for edge in self.edges]

        # Compute angles using PCA (only if edge has at least 2 points)
        self.angles = [
            np.degrees(np.arctan2(*PCA(n_components=1).fit(edge.reshape(-1, 2)).components_[0][::-1])) % 180
            if len(edge) >= 2 else np.nan for edge in self.edges
        ]

        # Estimate persistence lengths if the arc length meets the minimum threshold; otherwise, assign NaN
        p_lengths = [
            estimate_persistence_length(edge) if self.arc_lengths[idx] >= self.min_prune else np.nan 
            for idx, edge in enumerate(self.edges)
        ]

        # Create a boolean mask for valid persistence lengths (i.e., not NaN)
        valid = ~np.isnan(p_lengths)

        # Filter persistence lengths and corresponding arc lengths
        self.pers_lengths = np.array(p_lengths)[valid]
        valid_arc_lengths = np.array(self.arc_lengths)[valid]

        # Compute the length coefficient (persistence / arc length) for valid cases
        self.log_len_coef = np.log(self.pers_lengths) / valid_arc_lengths

        self.stats = {
                    "Mean": [np.mean(self.arc_lengths), np.mean(self.angles), np.mean(self.pers_lengths), np.mean(self.log_len_coef)],
                    "Median": [np.median(self.arc_lengths), np.median(self.angles), np.median(self.pers_lengths), np.median(self.log_len_coef)],
                    "Minimum": [np.min(self.arc_lengths), np.min(self.angles), np.min(self.pers_lengths), np.min(self.log_len_coef)],
                    "Maximum": [np.max(self.arc_lengths), np.max(self.angles), np.max(self.pers_lengths), np.max(self.log_len_coef)],
                    "SD": [np.std(self.arc_lengths), np.std(self.angles), np.std(self.pers_lengths), np.std(self.log_len_coef)],
                    "n": [len(self.arc_lengths), len(self.angles), len(self.pers_lengths), len(self.log_len_coef)],
                }
        self.stats_df = pd.DataFrame.from_dict(
            self.stats,
            orient="index",
            columns=["Fiber Length", "Direction", "Persistent Length", "Lambda (Log(Persistent)/Actual)"]
        ).reset_index().rename(columns={"index": "Statistic"})

        norm = Normalize(vmin=min(self.arc_lengths), vmax=max(self.arc_lengths))
        colormap = cm.plasma
        color_image = np.zeros((self.skel_img.shape[0], self.skel_img.shape[1], 3), dtype=np.uint8)

        for idx, (edge, length) in enumerate(zip(self.edges, self.arc_lengths)):
            if not isinstance(edge, np.ndarray):
                continue
            edge = edge.astype(np.int32)
            color = (
                tuple(random.randint(0, 255) for _ in range(3)) if mode == "rand"
                else tuple(int(c * 255) for c in colormap(norm(length))[:3])
            )
            cv2.polylines(color_image, [edge], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
            if text and len(edge) > 0:
                mid = edge[len(edge) // 2]
                mid_x, mid_y = mid[0] if isinstance(mid[0], (np.ndarray, list, tuple)) else mid
                cv2.putText(color_image, str(idx), (mid_x, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        self.color_image = color_image

        fig = plt.figure(figsize=(20, 12), constrained_layout=True)
        spec = fig.add_gridspec(2, 3, height_ratios=[1, 1])

        ax_img = fig.add_subplot(spec[0, 0])
        ax_img.imshow(self.image, cmap='gray')
        ax_img.set_title("Original Image")
        ax_img.axis('off')

        ax_skel = fig.add_subplot(spec[0, 1])
        ax_skel.imshow(255 - self.skel_img, cmap='gray')
        ax_skel.set_title("Skeletonized Image")
        ax_skel.axis('off')

        ax_color = fig.add_subplot(spec[0, 2])
        ax_color.imshow(self.color_image)
        ax_color.set_title("Colored Edge Network")
        ax_color.axis('off')

        # Composite subplot (boxplot and histogram)
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        subspec = spec[1, 0].subgridspec(2, 1, height_ratios=[1, 3], hspace=0.01)
        ax_box = fig.add_subplot(subspec[0])
        ax_hist = fig.add_subplot(subspec[1], sharex=ax_box)

        for ax in [ax_box, ax_hist]:
            ax.set_facecolor('none')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1)
            ax.spines['bottom'].set_linewidth(1)

        boxplot(x=self.arc_lengths, ax=ax_box, color='lightblue', width=0.5, orient='h')
        ax_box.set_title("Fiber Length Boxplot", pad=2)
        ax_box.set_xlabel("")
        ax_box.tick_params(labelbottom=False, pad=2)
        ax_box.spines['bottom'].set_visible(False)
        ax_box.spines['left'].set_visible(False)
        ax_box.spines['top'].set_visible(False)
        ax_box.spines['right'].set_visible(False)

        histplot(self.arc_lengths, bins=30, kde=True, ax=ax_hist, color='steelblue', edgecolor='black')
        ax_hist.axvline(np.median(self.arc_lengths), color='orange', linestyle='--', label='Median')
        ax_hist.axvline(np.mean(self.arc_lengths), color='blue', linestyle='--', label='Mean')
        ax_hist.set_xlabel("Fiber Length")
        ax_hist.set_ylabel("Frequency")
        ax_hist.set_title("Length Distribution", pad=2)
        ax_hist.legend(frameon=False)

        # Rose plot
        valid_angles = np.array([a for a in self.angles if not np.isnan(a)])
        ax_rose = fig.add_subplot(spec[1, 1], projection='polar')
        ax_rose.hist(np.radians(valid_angles), bins=36, color="coral")
        ax_rose.set_title("Rose Diagram (Orientation)", pad=20)
        ax_rose.set_theta_zero_location("E")  # Rotate so 0° is right
        ax_rose.set_theta_direction(1)        # Clockwise
        ax_rose.set_thetamin(0)
        ax_rose.set_thetamax(180)

        # Filter for valid angles and corresponding length coefficients
        angles_valid = np.array(self.angles)[valid]
        len_coef_valid = self.len_coef

        # Create DataFrame for seaborn
        df_density = pd.DataFrame({
            "Angle": angles_valid,
            "Length Coefficient": len_coef_valid
        })

        # KDE plot
        ax_kde = fig.add_subplot(spec[1, 2])
        kdeplot(
            data=df_density,
            x="Angle", y="Length Coefficient",
            cmap="viridis",
            fill=True,
            thresh=0.02,
            bw_adjust=0.7,
            ax=ax_kde
        )
        ax_kde.set_title("KDE: Angle vs. Length Coefficient")
        ax_kde.set_xlabel("Angle (degrees)")
        ax_kde.set_ylabel("Persistence / Arc Length")

        avg_length = np.mean(self.arc_lengths) if self.arc_lengths else 0
        fig.suptitle(f"Average Fiber Length: {avg_length:.2f}", fontsize=16, y=1.02)

        plt.show()
        self.fig = fig
        self.axes = fig.axes

    def find_length(self):
        total_start = time.time()
        print("🚀 Fiber length analysis started...\n")

        self.preproc()
        self.branch()
        self.intersection_associate()
        self.edge_connect()
        self.merge_edges()
        self.tip_association()
        self.viz_and_sum()

        print(f"🎯 Fiber length analysis completed in {time.time() - total_start:.2f}s\n")

    def export_results(self, output_dir="fiberL_output", export_pdf=True, export_images=True):
        """
        Export results: saves images, fiber length data, and a PDF report including metadata and summary plots.
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"fiberL_{timestamp}"

        # 1. Save raw images
        if export_images:
            color_path = os.path.join(output_dir, f"{base_name}_colored_edges.png")
            skel_path = os.path.join(output_dir, f"{base_name}_skeleton.png")

            cv2.imwrite(color_path, cv2.cvtColor(self.color_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(skel_path, self.skel_img * 255 if self.skel_img.max() <= 1 else self.skel_img)
            print(f"🖼️  Saved edge and skeleton images to: {output_dir}")

        # 2. Save fiber lengths
        csv_path = os.path.join(output_dir, f"{base_name}_fiber_lengths.csv")
        np.savetxt(csv_path, self.arc_lengths, delimiter=",", header="fiber_length", comments="")
        print(f"📏 Fiber lengths saved to: {csv_path}")

        # 3. Generate summary PDF
        if export_pdf:
            pdf_path = os.path.join(output_dir, f"{base_name}_report.pdf")
            with PdfPages(pdf_path) as pdf:
                # Page 1: Main visualization
                pdf.savefig(self.fig)

                # Page 2: Parameters and statistics
                from matplotlib.figure import Figure
                fig2 = Figure(figsize=(8.5, 11))
                ax2 = fig2.subplots()

                stats = {
                    "Average length": np.mean(self.arc_lengths),
                    "Minimum length": np.min(self.arc_lengths),
                    "Maximum length": np.max(self.arc_lengths),
                    "Standard deviation": np.std(self.arc_lengths),
                    "Number of fibers": len(self.arc_lengths),
                }

                settings = {
                    "Preprocessing": f"Anisotropic diffusion (niter={self.niter}, kappa={self.kappa}, gamma={self.gamma})",
                    "Thresholds": f"Binary={self.thresh_1}, Final={self.thresh_2}",
                    "Gaussian blur": self.g_blur,
                    "Pruning": f"min_prune={self.min_prune}, node_dist={self.max_node_dist}",
                    "Exported on": timestamp,
                }

                lines = ["FIBER LENGTH ANALYSIS REPORT", "-"*40, "📌 Parameter Settings:"]
                lines += [f"{k}: {v}" for k, v in settings.items()]
                lines += ["\n📊 Summary Statistics:"]
                lines += [f"{k}: {v:.2f}" for k, v in stats.items()]

                ax2.axis("off")
                ax2.text(0.01, 0.99, "\n".join(lines), fontsize=11, va="top")

                pdf.savefig(fig2)

            print(f"📄 PDF report saved to: {pdf_path}")

    def export_results_streamlit(self):
        """
        Streamlit-compatible export: allows user to download images, CSV, and PDF via download buttons.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"fiberL_{timestamp}"

        # Use a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save colored edge image
            color_path = os.path.join(tmpdir, f"{base_name}_colored_edges.png")
            cv2.imwrite(color_path, cv2.cvtColor(self.color_image, cv2.COLOR_RGB2BGR))

            # Save skeleton image
            skel_path = os.path.join(tmpdir, f"{base_name}_skeleton.png")
            cv2.imwrite(skel_path, self.skel_img * 255 if self.skel_img.max() <= 1 else self.skel_img)

            # Save fiber lengths CSV
            csv_path = os.path.join(tmpdir, f"{base_name}_fiber_lengths.csv")
            np.savetxt(csv_path, self.arc_lengths, delimiter=",", header="fiber_length", comments="")

            # Save PDF report
            pdf_path = os.path.join(tmpdir, f"{base_name}_report.pdf")
            with PdfPages(pdf_path) as pdf:
                pdf.savefig(self.fig)

                fig2 = Figure(figsize=(8.5, 11))
                ax2 = fig2.subplots()
              
                settings = {
                    "Preprocessing": f"Anisotropic diffusion (niter={self.niter}, kappa={self.kappa}, gamma={self.gamma})",
                    "Thresholds": f"Binary={self.thresh_1}, Final={self.thresh_2}",
                    "Gaussian blur": self.g_blur,
                    "Pruning": f"min_prune={self.min_prune}, node_dist={self.max_node_dist}",
                    "Exported on": timestamp,
                }

                lines = ["FIBER LENGTH ANALYSIS REPORT", "-"*40, "📌 Parameter Settings:"]
                lines += [f"{k}: {v}" for k, v in settings.items()]
                lines += ["\n📊 Summary Statistics:"]
                for _, row in self.stats_df.iterrows():
                    stat = row["Statistic"]
                    vals = ", ".join([f"{col}: {row[col]:.2f}" for col in self.stats_df.columns[1:]])
                    lines.append(f"{stat}: {vals}")

                ax2.axis("off")
                ax2.text(0.01, 0.99, "\n".join(lines), fontsize=11, va="top")

                pdf.savefig(fig2)

            # Create download buttons
            with open(color_path, "rb") as f:
                st.download_button("📥 Download Colored Edges PNG", f, file_name=os.path.basename(color_path))

            with open(skel_path, "rb") as f:
                st.download_button("📥 Download Skeleton PNG", f, file_name=os.path.basename(skel_path))

            with open(csv_path, "rb") as f:
                st.download_button("📥 Download Fiber Lengths CSV", f, file_name=os.path.basename(csv_path))

            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download PDF Report", f, file_name=os.path.basename(pdf_path))

    def plot_edge(self, edge_no):
        x, y = self.edges[edge_no][:, 0, 1], self.edges[edge_no][:, 0, 0]
        plt.scatter(x,y)
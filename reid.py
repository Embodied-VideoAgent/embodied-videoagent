from __future__ import annotations

import networkx as nx
from networkx.algorithms.matching import max_weight_matching
from utils import check_dynamic
from object3d import Object3D


IOU_THRESHOLD_SCORE = 0.2
VISUAL_2D_THRESHOLD_SCORE = 0.45
BBOX_SIMILARITY_SCORE = 0.7


def static_object_reid(old_objects: list[Object3D], new_objects: list[Object3D]):
    """re-identifier algorithm for virtual 3D objects
    using spatial and visual similarity"""
    graph = nx.Graph()
    new_object_identifiers = []
    old_object_identifiers = []
    for o1 in new_objects:
        new_object_identifiers.append(o1.identifier)
    for o2 in old_objects:
        old_object_identifiers.append(o2.identifier)
    graph.add_nodes_from(new_object_identifiers, bipartite=0)
    graph.add_nodes_from(old_object_identifiers, bipartite=1)
    for o1 in new_objects:
        for o2 in old_objects:
            iou = o1.spatial_iou(o2)
            max_ios = o1.spatial_max_ios(o2)
            if iou > IOU_THRESHOLD_SCORE or (
                max_ios > IOU_THRESHOLD_SCORE and o1.category == o2.category
            ):
                category_match = 1.0 if o1.category == o2.category else 0.0
                score = (
                    1.0 * max(0, iou - 0.2)
                    + 1.0 * max(0, max_ios - 0.2) * category_match
                )
                graph.add_edge(o1.identifier, o2.identifier, weight=score)
    matching = max_weight_matching(graph, maxcardinality=True)
    new_matching = []
    for pair in matching:
        if pair[0] in new_object_identifiers and pair[1] in old_object_identifiers:
            new_matching.append(pair)
        else:
            new_matching.append((pair[1], pair[0]))
    return new_matching


def dynamic_object_reid(old_objects: list[Object3D], new_objects: list[Object3D]):
    """re-ID algorithm for dynamic 3D objects using 3D-bbox similarity and visual features."""
    graph = nx.Graph()
    new_object_identifiers = []
    old_object_identifiers = []
    for o1 in new_objects:
        new_object_identifiers.append(o1.identifier)
    for o2 in old_objects:
        old_object_identifiers.append(o2.identifier)

    graph.add_nodes_from(new_object_identifiers, bipartite=0)
    graph.add_nodes_from(old_object_identifiers, bipartite=1)
    for o1 in new_objects:
        for o2 in old_objects: 
            visual_similarity_score = o1.visual_similarity_score(o2)
            bbox_similarity_score = o1.spatial_vol_sim(o2)
            if visual_similarity_score >= VISUAL_2D_THRESHOLD_SCORE and bbox_similarity_score >= BBOX_SIMILARITY_SCORE:
                score = visual_similarity_score + bbox_similarity_score
                graph.add_edge(o1.identifier, o2.identifier, weight=score)
    matching = max_weight_matching(graph, maxcardinality=True)
    new_matching = []
    for pair in matching:
        if pair[0] in new_object_identifiers and pair[1] in old_object_identifiers:
            new_matching.append(pair)
        else:
            new_matching.append((pair[1], pair[0]))
    return new_matching
    

def split_static_dynamic_objects(old_objects: list[Object3D], depth, mask, pos, rot, hfov):
    dynamic_objects = []
    static_objects = []
    for i, obj in enumerate(old_objects):
        object_dynamic = check_dynamic(obj=obj, pos=pos, rmat=rot, hfov=hfov, depth=depth)
        if object_dynamic:
            dynamic_objects.append(obj)
        else:
            static_objects.append(obj)
    return static_objects, dynamic_objects



def remove_duplicate_objects(
    virtual_objects: list[Object3D], updated_indices: list[int]
):
    """remove duplicated virtual object detections"""

    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))

        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, x, y):
            fx, fy = self.find(x), self.find(y)
            if fx != fy:
                self.parent[fy] = fx

    def get_min_nodes_in_components(n, edges):
        uf = UnionFind(n)
        for u, v in edges:
            uf.union(u, v)
        comp_min = {}
        for i in range(n):
            root = uf.find(i)
            if root not in comp_min:
                comp_min[root] = i
            else:
                comp_min[root] = min(comp_min[root], i)
        return list(comp_min.values())

    edges = []
    # part_1_time_start = time.time()
    num = len(virtual_objects)
    for i in updated_indices:
        for j in range(num):
            o1 = virtual_objects[i]
            o2 = virtual_objects[j]
            iou = o1.spatial_iou(o2)
            max_ios = o1.spatial_max_ios(o2)
            if iou > IOU_THRESHOLD_SCORE or (
                max_ios > IOU_THRESHOLD_SCORE and o1.category == o2.category
            ):
                edges.append((i, j))
    # part_1_time_end = time.time()
    # print("part_1_time_cost: ", part_1_time_end-part_1_time_start)
    new_indices = get_min_nodes_in_components(num, edges)
    # part_2_time_end = time.time()
    # print("part_2_time_cost: ", part_2_time_end-part_1_time_end)
    new_existing_objects = [
        obj for idx, obj in enumerate(virtual_objects) if idx in new_indices
    ]

    # print("new virtual object len: ", len(new_existing_objects))
    return new_existing_objects

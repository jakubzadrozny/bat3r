from collections import defaultdict

from bat3r.skin import OneMeshGltf


def recover_parents(ancestors):
    parents = {}
    children = defaultdict(list)
    descendants = defaultdict(list)
    roots = []
    
    for j, A_j in ancestors.items():
        A_j = set(A_j)
        if len(A_j) == 0:
            parents[j] = None
            roots.append(j)
            continue
        
        best_a = None
        best_size = -1
        for a in A_j:
            # descendants[a].append(j)
            A_a = set(ancestors[a])
            if A_a.issubset(A_j):
                if len(A_a) > best_size:
                    best_size = len(A_a)
                    best_a = a
        
        parents[j] = best_a
        children[best_a].append(j)

    def get_descendants(v):
        for child in children[v]:
            get_descendants(child)
            for d, dist in descendants[child]:
                descendants[v].append((d, dist+1))
            descendants[v].append((child, 1))

    for root in roots:
        get_descendants(root)

    return parents, children, descendants, roots


def _topsort(v, children, order):
    order.append(v)
    for child in children[v]:
        order = _topsort(child, children, order)
    return order


def topsort(children, roots):
    return sum((_topsort(root, children, []) for root in roots), [])


def tree_sanity_check(mesh: OneMeshGltf, parents, children, roots):
    ancestors = defaultdict(list)
    
    def _get_ancestors(v, parents, children):
        if parents[v] is not None:
            ancestors[v].append(parents[v])
            ancestors[v] += ancestors[parents[v]]
        for child in children[v]:
            _get_ancestors(child, parents, children)
    
    for root in roots:
        _get_ancestors(root, parents, children)

    okay = all((set(mesh.nodes_parents_list[v]) == set(ancestors[v])) for v in mesh.joints.tolist())
    return okay

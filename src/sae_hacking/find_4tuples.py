#!/usr/bin/env python3


def find_pattern(graph):
    # Get bipartite partition
    X, Y = bipartite_partition(graph)

    results = []

    # Precompute positive and negative neighbors for each node
    pos_neighbors = {node: set() for node in graph}
    neg_neighbors = {node: set() for node in graph}

    for u in graph:
        for v, weight in graph[u].items():
            if weight > 0:
                pos_neighbors[u].add(v)
            else:
                neg_neighbors[u].add(v)

    # For each A in X and each C in X
    for A in X:
        # Find all B where AB is positive
        pos_B_from_A = pos_neighbors[A]

        for C in X:
            if A == C:
                continue

            # Find all B where BC is positive
            pos_B_from_C = pos_neighbors[C]

            # Find common B nodes: these can be the 'B' in our 4-tuple
            common_B = pos_B_from_A.intersection(pos_B_from_C)

            # Find all D where AD is positive and CD is negative
            potential_D = pos_neighbors[A].intersection(neg_neighbors[C])

            # Generate all valid 4-tuples
            for B in common_B:
                for D in potential_D:
                    if B != D:  # Ensure distinct nodes
                        results.append((A, B, C, D))

    return results

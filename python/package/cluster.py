
def cluster_graph(graph_dict, tol, verbose=False):
    nodes_set = set()
    for (n, n2) in graph_dict:
        nodes_set.add(n)
        nodes_set.add(n2)
    nodes = list(nodes_set)

    graph_ordered = sorted(graph_dict.items(), key= lambda x: x[1])

    node_cluster = {node: node for node in nodes}
    cluster_size = {node: 1 for node in nodes}
    cluster_nodes = {node: [node] for node in nodes}
    edges_solution = {}

    def find(node):
        return node_cluster[node]

    def union(cluster1, cluster2):
        if cluster_size[cluster1] < cluster_size[cluster2]:
            cluster1, cluster2 = cluster2, cluster1

        for node in cluster_nodes[cluster2]:
            node_cluster[node] = cluster1

        cluster_nodes[cluster1].extend(cluster_nodes[cluster2])
        cluster_size[cluster1] += cluster_size[cluster2]
        cluster_nodes.pop(cluster2)
        cluster_size.pop(cluster2)

    num_clusters = len(nodes)
    for (n1, n2), e in graph_ordered:
        if e >= tol:
            break
        cluster1 = find(n1)
        cluster2 = find(n2)
        if cluster1 == cluster2:
            if verbose:
                print('node {} and node {} have'
                      ' the same cluster {}'.format(
                    n1, n2, cluster2))
            continue
        union(cluster1, cluster2)
        num_clusters -= 1
        edges_solution[(n1, n2)] = e
        if verbose:
            print('cluster {} and cluster {} are'
                  ' being joined with cost {}'.format(
                cluster1, cluster2, e))

    cluster_nodes2 = {}
    for c, nodes in cluster_nodes.items():
        max_ = [0, 0]
        for n in nodes:
            for pos in range(2):
                if n[pos] > max_[pos]:
                    max_[pos] = n[pos]
        cluster_nodes2[max_[0], max_[1]] = nodes
    return cluster_nodes2

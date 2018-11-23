---
title:  ROADEF 2018 summary
author: Team S22
documentclass: report
header-includes: |
    \usepackage{booktabs}
...

## Introduction

## Definitions

* **Tree**: first node. Corresponds to a *Jumbo*. A node without a parent.
* **Node**: tree representation of an *Item*. It has a parent and can have children or be a *Leaf*.
* **Forest**: a list of *Trees*, corresponding to the solution for an instance or *Batch*.
* **Solution**: same as *Forest*. The result of an instance. Should comply with constraints.
* **Child**: a *Node* that is obtained by cutting a bigger node. All nodes are children of another node except the tree or first node.
* **Leaf**: a *Node* that has no *children*. Corresponds to an *Item* or to a *Waste*. Or to a residual.
* **Swap**: the act of taking two *Nodes* and switching their places, while taking into account the differences in space. The result needs to still keep the logic of the *Jumbo* / *Tree* structure.
* **Insert**: just like the *Swap* but only the first node is inserted just before the second node. The second node is not moved.
* **Level**: Nodes of level 1 are the nodes that have CUT parameter equal to 1. A level 1 swap is a swap between nodes of level 1.

## Data structures

The solution is stored as a forest.

### Trees and nodes

If a node is not a leaf, then it has been cut. Its children represent the number of cuts it has received, following the rules of guillotine cuts.
Each children node has a level that is 1 bigger than its parent. This way we can track the level a node is in (and the number of cuts we had to do to arrive to the node).

The information present in a node, besides the regular graph information, is:

* *Size*: `WIDTH`, `HEIGHT`
* *Position*: `PLATE_ID`, `X`, `Y`
* `CUT`: the number of cuts needed to make to arrive to the node.
* `TYPE`: following the specified format: waste, intermediate node or an item.

## The solving approach

At the moment, there are two algorithms that work together: a simulated annealing algorithm and a construction algorithm.

### Construction algorithm

An initial solution is done by ordering the nodes with many (random) feasible sequences and trying to insert them one after the other in the first available feasible whole in the jumbos. The best solution of all simulated is kept.

This procedure to "create jumbos" is ran later with subsets of contiguous jumbos as part of an existing solution in order to try to find better neighbours.

### Simulated annealing

The following changes are done at any of the three levels (between nodes of level 1, nodes of level 2 and nodes of level 3).

1. find better jumbos by reordering a sequence of jumbos of size 1 (or bigger).
1. try to merge wastes that are neighbours into one single waste.
1. create waste cuts following the positions of defects
1. push wastes to the right of the solution (to the end).
1. do swapping in same level to correct sequence.
1. do swapping to correct defects.
1. do swapping in the same level from random pairs of nodes.
1. delete empty cuts.
1. make inter-level swaps of 1 level difference (1 and 2, 2 and 3, 3 and 4).
1. make inter-level swaps of 2 level difference (1 and 3, 2 and 4).

### Swaps

Swaps can be "real swaps" or just "an insertion". We call everything swaps even if they are just inserts.
Also, they can include the rotation of none, the first, the second, or both nodes.
Finally, the nodes can be in the same level or in different levels.

#### Checking swaps

In order to make a feasible swap, it's necessary to check the following:

* *In a swap*: each node is smaller than the space left by the other node plus the available waste.
* *In an insert*: the inserted node needs to be smaller than the available waste next to the other node.

If there's space, in theory we should be able to do it.

To evaluate the improvement of the swap we need to:

1. Calculate the balance of sequence violations.
2. Calculate the balance of defects violations.
3. Calculate the balance of density of waste moved to the right.

These three terms, weighted by some parameters, will determine if the swap is done. Here, there is always an element of randomness based on the temperature.

#### Inserting a node somewhere

This is the basic logic to insert a node inside a destination node. This is use as the base for all the swaps mentioned above.

1. Take out children waste from each node. This is the waste that's at the end if any.
1. If only child, take out and decrease level.
2. If rotation is needed, rotate the whole node.
3. Insert node in its destination parent.
4. If the level of the node has changed, update it.
5. if the node corresponds to actually several nodes, get them out and re-insert them.
6. if needed, add a children waste on the node(s) that has /have just been inserted.
7. Since a node has just been inserted, modify the residual waste at the end of the destination node.

## Results

### Summary

\input{results.tex}

### Characteristics of the computer used

<!-- free -mh -->
<!-- lscpu -->

<!-- **prise-srv3**:

* 62 GB RAM
* 12 cores were used from 48
* CPU speed (in MHz): 2300.000
 -->

<!-- **serv-cluster1**: -->

* 62 GB RAM
* 12 cores.
* CPU speed (in MHz): 2927.000


# Asking Leetcode is stupid, so let's crack it fast.

Python-first playground for classic algorithms and modern ML experiments. The repo mixes concise script files (ideal for quick interview refreshers) with exploratory notebooks for deep dives.

## Repository Map

| Theme | Highlights | Entry Points |
| --- | --- | --- |
| Sorting & Search | quicksort, mergesort, heap sort, binary search, two-pointer tricks | `sorting.py`, `BinarySearch.py`, `twoPointer.py` |
| Trees & Linked Lists | traversal patterns, balanced-tree helpers, pointer rewiring | `tree_linkedlist.py` |
| Graph Algorithms | queue-based BFS, stack-based DFS, Dijkstra with `heapq`, Bellman-Ford, MST variants, topological sort | `BFSqueue.py`, `DFSstack.py`, `DijkstraHeapq.py`, `Bellman Ford.py`, `MST.py`, `topological.py` |
| Dynamic Programming | knapsack variants, Kadane, tabulation/ memoization templates | `dynamicPrg.py`, `Kadane_Knapsack.py` |
| Greedy / Backtracking | interval scheduling, coin change, recursion utilities | `Greedy.py`, `recursion_backtrack.py` |
| Matrix & Numerical | 2D sweep algorithms, prefix sums, grid traversal helpers | `matrix.py` |
| Systems & Misc | high-level system design sketches, utility grab bag | `system design.py`, `MISC.py`, `progressive.py` |
| Networking | bandwidth smoothing heuristics | `equalizeBandwidth.py` |
| Deep Learning | Transformer / ViT / BERT playground, classical DNN + CNN + KNN experiments | `Transformer_ViT_BERT.ipynb`, `DNN_CNN_KNN.ipynb` |

## How to Use

1. Ensure Python 3.10+ is available; install any ad-hoc deps referenced inside specific scripts (most rely only on the standard library).
2. Run individual scripts directly, e.g.:
   ```bash
   python /Users/junyang/Desktop/JOB/Algo/DijkstraHeapq.py
   ```
3. Open notebooks in VS Code, JupyterLab, or Colab for interactive experimentation with the Transformer/ViT/BERT stack or KNN baselines.

## Suggested Study Paths

- **Interview sprint**: skim `sorting.py`, `BinarySearch.py`, `twoPointer.py`, then iterate through `tree_linkedlist.py` and `dynamicPrg.py`.
- **Graphs masterclass**: queue → stack → priority queue by following `BFSqueue.py`, `DFSstack.py`, `DijkstraHeapq.py`, and `MST.py`.
- **ML refresh**: replicate the ViT vs. vanilla Transformer comparison in `Transformer_ViT_BERT.ipynb`, then contrast with `DNN_KFold_KNN.ipynb`.

## Contributing

1. Fork and create a topic branch.
2. Add a focused implementation or polish an existing one (tests or example invocations are appreciated).
3. Open a PR describing the algorithmic idea, complexity, and any references.

## License

MIT — see `LICENSE` (or add one if missing). Use freely for learning, interviews, or production inspiration.*** End Patch*** End Patch

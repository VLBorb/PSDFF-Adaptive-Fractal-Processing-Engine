# Implementation Details

This document covers specific details about the Python implementation of the Enhanced Fractal System.

## Core Data Structures

1.  **`CONFIG` (dict):**
    * Stores all global parameters, making them easily accessible and modifiable[cite: 3, 66, 411, 545, 619, 784].
    * Includes settings for network size, frequency encoding, resonance, combination logic, simulation length, cache/memory limits, threading, and adaptive behaviors.

2.  **`TimeController` (class):**
    * Uses `time.time()` to track elapsed real time and divides it by `self.tau = 1.0 / CONFIG["FRAMES_PER_SECOND"]` to derive discrete `T_epoch` and `T_fine` values[cite: 19, 20, 22, 68, 70, 412, 414, 496, 548, 665, 788].

3.  **`BLF` (class):**
    * `self.flows`: Stores a list of binary flow lists (e.g., `[[0,1,0,...], [1,1,0,...]]`)[cite: 22, 71, 415, 497, 549, 625, 790].
    * `self.frequencies`: A list of lists, where each inner list contains the numeric frequencies corresponding to a binary flow[cite: 23, 71, 415, 497, 549, 625, 790].
    * `self.average_freqs`: A list of mean frequencies, one for each flow, used in resonance checks[cite: 23, 71, 415, 497, 549, 625, 790].
    * `self.identity`: A UUIDv5 hex string representing the BLF's spectral hash[cite: 24, 72, 416, 499, 551, 666, 790].

4.  **`AdaptiveCombinerNode` (class):**
    * `self.activated_flows`: A `collections.deque` with `maxlen=CONFIG["MAX_NODE_CACHE"]`. It stores dictionaries representing activated flows: `{"freq": avg_freq, "flow_hash": blf.identity, "time": current_time}`[cite: 25, 27, 75, 76, 419, 421, 457, 461, 503, 555, 626, 629, 667, 791].
    * `self.combined_outputs`: A list that temporarily stores the results of `combine_flows` before they are collected for FractalMemory or chaining[cite: 30, 79, 423, 457, 556, 626, 666, 791].
    * `self.combination_cache`: A dictionary used to cache the output hashes of combined flow pairs to prevent redundant SHA256 computations. The key is a string derived from the source flow hashes and their frequencies (e.g., `f"{hash1}_{freq1:.2f}_{hash2}_{freq2:.2f}"`)[cite: 395, 402, 440, 457, 462, 463, 556, 557, 626, 630, 666, 791]. It has a simple LIFO-like eviction policy if `MAX_NODE_CACHE` is exceeded[cite: 440, 463, 556, 630].
    * `self.neighbors`: A list of node IDs representing the neighbors this node will feed its outputs to[cite: 626, 637, 666, 791].

5.  **`OptimizedFractalMemory` (class):**
    * `self.storage`: A dictionary where keys are output hashes of combined flows. Values are dictionaries containing `{"combined_freq", "source_hashes", "T_epoch", "occurrences"}`[cite: 31, 80, 424, 441, 464, 505, 561, 632, 633, 668, 669, 794, 795].
    * `self.max_size`: Limits the number of entries in `self.storage`. When exceeded, the entry with the oldest `T_epoch` is removed[cite: 450, 463, 505, 561, 633, 669, 812].

6.  **`AdaptiveFractalProcessor` (class):**
    * Stores `gamma1, gamma2, gamma3` coefficients for its equations[cite: 785].
    * Methods `compute_priority`, `adaptive_response`, `compute_vfe` implement the respective formulas using NumPy for array operations[cite: 785, 786, 787].
    * The `sigmoid` function is defined globally for use in `compute_priority`[cite: 785].

## Key Algorithms & Logic Flow

1.  **Frequency Encoding (`BLF.encode_frequencies`):**
    * Iterates through bits of a flow. If `bit == 1` at index `i`, frequency `CONFIG["BASE_FREQ"] + CONFIG["DELTA_F"] * i` is added. If `bit == 0`, `CONFIG["BASE_FREQ"]` is added[cite: 23, 24, 72, 73, 74, 417, 418, 498, 550, 551, 665, 666, 790].

2.  **Node Activation (`AdaptiveCombinerNode.check_activation`):**
    * Resets if `current_time["T_epoch"] != blf.timestamp["T_epoch"]`[cite: 27, 76, 421, 460, 503, 555, 629, 667, 792].
    * Calls `adapt_epsilon`[cite: 460, 503, 555, 629, 667, 792].
    * Iterates `blf.average_freqs`. If `abs(avg_freq - self.native_freq) < self.epsilon`, appends flow info to `self.activated_flows`[cite: 27, 76, 421, 461, 503, 555, 629, 667, 792].

3.  **Flow Combination (`AdaptiveCombinerNode.combine_flows`):**
    * Pops two flows (f1, f2) from `self.activated_flows` while `len >= 2`[cite: 30, 79, 423, 440, 462, 556, 630, 667, 793].
    * Forms a `combined_key` from sorted f1 and f2 hashes and frequencies[cite: 462, 557, 630].
    * Checks `self.combination_cache` for this key. If found, uses cached hash. Else, computes SHA256 hash and caches it[cite: 440, 462, 463, 556, 557, 630].
    * Computes `new_freq = (f1["freq"] + f2["freq"]) * CONFIG["FREQ_COMBINATION_FACTOR"]`[cite: 31, 79, 423, 440, 463, 556, 630, 668, 793].
    * Appends result to `self.combined_outputs`[cite: 31, 79, 423, 440, 463, 556, 631, 668, 793].

4.  **Fractal Memory Storage (`OptimizedFractalMemory.store`):**
    * Handles overlaps by incrementing `occurrences` if `delta_freq < 5` and `delta_time <= 2`[cite: 33, 82, 425, 441, 464, 506, 562, 633, 669, 796].
    * Handles conflicts by creating a `new_hash` (UUIDv5 from old hash + epoch) if thresholds for overlap are not met but the primary hash already exists[cite: 33, 82, 425, 441, 464, 506, 562, 633, 669, 796].
    * Evicts oldest entry if `max_size` is reached[cite: 450, 463, 505, 561, 633, 669, 812].

5.  **Parallel Processing (`parallel_node_processing`):**
    * Uses `concurrent.futures.ThreadPoolExecutor` to submit the `process_node` task (which calls `check_activation` and `combine_flows`) for each node in the network[cite: 466, 507, 563, 635, 638, 671, 673, 800, 814].
    * Collects results (e.g., activation counts) from the futures.

6.  **Node Chaining (`neighbor_chaining` and `AdaptiveCombinerNode.feed_new_outputs`):**
    * `neighbor_chaining`:
        * Collects all `combined_outputs` from all nodes in the network after their initial processing round[cite: 579, 637].
        * For each node, it iterates through its defined `neighbors` and passes their collected outputs to the current node's `feed_new_outputs` method[cite: 637].
    * `feed_new_outputs`:
        * Checks if the `combined_freq` of these received outputs resonates with its own `native_freq` using its current `epsilon`[cite: 559, 560, 632, 794].
        * If resonant, appends them to `self.activated_flows` as new input[cite: 560, 632, 794].
    * After feeding, `combine_flows` is called again on each node to process these newly acquired chained flows[cite: 573, 579, 637].

## Libraries Used

* **`time`:** For `TimeController` and generating unique elements for cache keys (though the latter was removed for consistency)[cite: 19, 66, 410, 450, 494, 542, 618, 783].
* **`uuid`:** For generating `BLF.identity` (UUIDv5) and for new hash generation in `FractalMemory` on conflict[cite: 19, 24, 66, 74, 410, 418, 425, 450, 464, 494, 499, 506, 542, 551, 562, 618, 633, 666, 669, 783, 790, 796].
* **`random`:** For `generate_binary_flows`, adaptive epsilon randomization, and random neighbor selection[cite: 19, 66, 410, 450, 494, 542, 618, 627, 634, 783, 798].
* **`hashlib`:** For generating SHA256 hashes in `AdaptiveCombinerNode.combine_flows`[cite: 403, 439, 450, 463, 494, 542, 556, 618, 630, 668, 783].
* **`numpy`:** For `np.random.choice` in flow generation, `np.mean` for average frequencies, and array operations in `AdaptiveFractalProcessor` and `mini_fractal_demo_test`[cite: 19, 22, 23, 66, 71, 410, 414, 415, 450, 453, 454, 494, 497, 498, 542, 549, 550, 618, 625, 652, 665, 674, 783, 785, 789, 790, 802].
* **`pandas`:** For formatting and displaying logs (`Logger.print_summary`) and the table in `mini_fractal_demo_test`[cite: 19, 39, 66, 87, 88, 410, 429, 450, 467, 494, 508, 509, 542, 566, 567, 618, 623, 624, 638, 639, 652, 654, 663, 664, 676, 677, 678, 783, 787, 805, 806].
* **`collections.deque`:** For `AdaptiveCombinerNode.activated_flows` to have efficient appends and poplefts, and a fixed maximum length[cite: 397, 404, 439, 450, 457, 494, 542, 551, 618, 626, 666, 783, 791].
* **`concurrent.futures.ThreadPoolExecutor`:** For parallel execution of node processing[cite: 448, 451, 466, 471, 494, 507, 542, 563, 618, 635, 671, 783, 814].
* **`matplotlib.pyplot`:** For plotting in `mini_fractal_demo_test`[cite: 17, 41, 89, 134, 444, 618, 652, 654, 676, 783, 805, 817].
* **`ace_tools` (optional):** A custom tool mentioned for displaying DataFrames, with a fallback to standard print if not available[cite: 655, 661, 676, 783, 805].

This setup provides a flexible and extensible framework for simulating and studying the behavior of the fractal processing system.

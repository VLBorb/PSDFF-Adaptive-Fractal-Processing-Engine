import time
import uuid
import random
import hashlib
import numpy as np
import pandas as pd
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# Optional: if you have a custom display package 'ace_tools'
# If not, we'll just use standard printing
try:
    import ace_tools as tools
except ImportError:
    tools = None

###############################################################################
# 1. CONFIGURATION AND GLOBAL DEFINITIONS
###############################################################################
CONFIG = {
    "NUM_NODES": 20,
    "DELTA_F": 13,  # Numeric increment per bit [cite: 784]
    "BASE_FREQ": 1000,  # Base numeric for flows [cite: 784]
    "INITIAL_EPSILON": 25,  # Starting resonance tolerance [cite: 784]
    "BITS_PER_FLOW": 32,  # Number of bits per flow [cite: 784]
    "BLF_SIZE": 300,  # Number of flows per Bulk Logic Frame [cite: 784]
    "FREQ_COMBINATION_FACTOR": 0.5,  # Factor for combining flows [cite: 784]
    "FRAMES_PER_SECOND": 30,  # "Fractal" epochs per second [cite: 784]
    "NUM_CYCLES": 10,  # 10 cycles for demonstration [cite: 784]
    "MAX_NODE_CACHE": 500,  # how many combos to cache per node [cite: 784]
    "MAX_FRACTAL_MEMORY": 2000,  # memory limit for fractal storage [cite: 784]
    "THREAD_WORKERS": 4,  # [cite: 784]
    "ADAPT_EPSILON_EACH_CYCLE": True,  # [cite: 784]
    "NODE_TO_NODE_ADJ_TYPE": "ring",  # ring or random adjacency [cite: 784, 785]
    "NEIGHBOR_COUNT": 2  # [cite: 785]
}

###############################################################################
# 2. ADAPTIVE FRACTAL PROCESSOR (Φ, Ψ, VFE)
###############################################################################
def sigmoid(x): # [cite: 785]
    return 1 / (1 + np.exp(-x)) # [cite: 785]

class AdaptiveFractalProcessor: # [cite: 785]
    """
    Implements the fractal priority Φ(t), adaptive response Ψ(t),
    and emergent fractal velocity VFE(t) from the FFPE theoretical model.
    """
    def __init__(self, num_pins, gamma1=0.1, gamma2=2.0, gamma3=5.0): # [cite: 785]
        self.num_pins = num_pins # [cite: 785]
        # Coefficients in Φ(t) = σ(γ1 * D_local - γ2 * C_global + γ3 * Π(t)) [cite: 785]
        self.gamma1 = gamma1 # [cite: 785]
        self.gamma2 = gamma2 # [cite: 785]
        self.gamma3 = gamma3 # [cite: 785]

    def compute_priority(self, d_local, c_global, impulses): # [cite: 785]
        """
        Fractal Priority Φ for each pin:
        Φ(t) = sigmoid( gamma1*D_local - gamma2*C_global + gamma3*Pi )
        Returns an array (same length as pins). [cite: 785]
        """
        x = self.gamma1 * d_local - self.gamma2 * c_global + self.gamma3 * impulses # [cite: 786]
        return sigmoid(x) # [cite: 786]

    def adaptive_response(self, phi, psi_local, psi_global): # [cite: 786]
        """  ψ(t) = Φ(t) * ψ_local + [1 - Φ(t)] * ψ_global """ # [cite: 786]
        return phi * psi_local + (1 - phi) * psi_global # [cite: 786]

    def compute_vfe(self, phi, c_global, freqs, impulses): # [cite: 786]
        """
        VFE(t) = (1 / T) * sum( Φ * C_global * f_i * Pi ) / N
        We'll assume T=1 for a single "time-slice" scenario. [cite: 786]
        """
        N = self.num_pins # we consider T = 1 for integration over one step [cite: 787]
        numerator = np.sum(phi * c_global * freqs * impulses) # [cite: 787]
        vfe_val = numerator / (N * 1.0) # [cite: 787]
        return vfe_val # [cite: 787]

###############################################################################
# 3. LOGGER FOR NODE ACTIVATION & FRACTAL MEMORY
###############################################################################
class Logger: # [cite: 787]
    def __init__(self): # [cite: 787]
        self.node_activation_log = []  # (cycle, node_id, activated_count) [cite: 787]
        self.memory_usage_log = []  # (cycle, memory_size) [cite: 787]

    def log_node_activation(self, cycle, node_id, activated_count): # [cite: 787]
        self.node_activation_log.append((cycle, node_id, activated_count)) # [cite: 787]

    def log_memory_size(self, cycle, size): # [cite: 787]
        self.memory_usage_log.append((cycle, size)) # [cite: 787]

    def print_summary(self): # [cite: 787]
        print("\n=== LOG SUMMARY ===") # [cite: 787]
        df_activation = pd.DataFrame(self.node_activation_log, columns=["Cycle", "NodeID", "ActivatedFlows"]) # [cite: 787]
        print("\nNode Activation Log (Sample):") # [cite: 787]
        print(df_activation.head(30).to_string(index=False)) # [cite: 787]
        df_memory_log = pd.DataFrame(self.memory_usage_log, columns=["Cycle", "MemorySize"]) # [cite: 787]
        print("\nFractal Memory Usage Log (Sample):") # [cite: 788]
        print(df_memory_log.head(30).to_string(index=False)) # [cite: 788]

###############################################################################
# 4. TIME CONTROLLER (FOR EPOCH MEASUREMENT)
###############################################################################
class TimeController: # [cite: 788]
    def __init__(self): # [cite: 788]
        self.start_time = time.time() # [cite: 788]
        self.tau = 1.0 / CONFIG["FRAMES_PER_SECOND"] # [cite: 788]

    def now(self): # [cite: 788]
        elapsed = time.time() - self.start_time # [cite: 788]
        epoch = int(elapsed // self.tau) # [cite: 788]
        fine = round(elapsed % self.tau, 6) # [cite: 788]
        return {"T_epoch": epoch, "T_fine": fine} # [cite: 788]

###############################################################################
# 5. BULK LOGIC FRAME (BLF) - FLOWS + FREQUENCY ENCODING
###############################################################################
def generate_binary_flows(num_flows: int, bits: int): # [cite: 788]
    return [np.random.choice([0, 1], bits).tolist() for _ in range(num_flows)] # [cite: 788]

class BLF: # [cite: 788]
    """
    Represents a Bulk Logic Frame that contains multiple flows. [cite: 788]
    Each flow is a sequence of bits mapped to frequencies. [cite: 789]
    """
    def __init__(self, time_data, flows): # [cite: 790]
        self.timestamp = time_data # [cite: 790]
        self.flows = flows # [cite: 790]
        self.frequencies = [self.encode_frequencies(flow) for flow in flows] # [cite: 790]
        self.average_freqs = [np.mean(freq_list) for freq_list in self.frequencies] # [cite: 790]
        self.identity = self.generate_spectral_hash() # [cite: 790]

    def encode_frequencies(self, flow_bits): # [cite: 790]
        encoded = [] # [cite: 790]
        for i, bit in enumerate(flow_bits): # [cite: 790]
            if bit == 1: # [cite: 790]
                freq = CONFIG["BASE_FREQ"] + CONFIG["DELTA_F"] * i # [cite: 790]
            else: # [cite: 790]
                freq = CONFIG["BASE_FREQ"] # [cite: 790]
            encoded.append(freq) # [cite: 790]
        return encoded # [cite: 790]

    def generate_spectral_hash(self): # [cite: 790]
        hash_input = ''.join(f"{round(f,2)}" for f in self.average_freqs) # [cite: 790]
        return uuid.uuid5(uuid.NAMESPACE_DNS, hash_input).hex # [cite: 790]

###############################################################################
# 6. ADAPTIVE COMBINER NODE (CHECK ACTIVATION, COMBINE FLOWS)
###############################################################################
class AdaptiveCombinerNode: # [cite: 790]
    """
    Each node has a 'native_freq', an epsilon (resonance tolerance),
    and a cache of previously combined outputs. [cite: 790]
    """
    def __init__(self, node_id): # [cite: 791]
        self.node_id = node_id # [cite: 791]
        self.native_freq = CONFIG["BASE_FREQ"] + node_id * CONFIG["DELTA_F"] # [cite: 791]
        self.epsilon = CONFIG["INITIAL_EPSILON"] # [cite: 791]
        self.activated_flows = deque(maxlen=CONFIG["MAX_NODE_CACHE"]) # [cite: 791]
        self.combined_outputs = [] # [cite: 791]
        self.active = False # [cite: 791]
        self.combination_cache = {} # [cite: 791]
        self.neighbors = [] # [cite: 791]

    def set_neighbors(self, neighbor_ids): # [cite: 791]
        self.neighbors = neighbor_ids # [cite: 791]

    def adapt_epsilon(self, cycle_idx=0): # [cite: 791]
        if CONFIG["ADAPT_EPSILON_EACH_CYCLE"]: # [cite: 791]
            factor = random.uniform(0.8, 1.2) # [cite: 791]
            self.epsilon = max(5, min(100, self.epsilon * factor)) # [cite: 791]
        else: # [cite: 791]
            size = len(self.activated_flows) # [cite: 791]
            if size > 50: # [cite: 791]
                self.epsilon = max(5, self.epsilon * 0.95) # [cite: 791]
            elif size < 10: # [cite: 791]
                self.epsilon = min(100, self.epsilon * 1.05) # [cite: 791]

    def check_activation(self, blf, current_time, cycle_idx=0): # [cite: 791]
        """
        Decides how many flows match the node's native frequency
        within a tolerance 'epsilon'. [cite: 791]
        If it's a different time epoch, node resets. [cite: 792]
        """
        if current_time["T_epoch"] != blf.timestamp["T_epoch"]: # [cite: 792]
            self.active = False # [cite: 792]
            self.activated_flows.clear() # [cite: 792]
            return 0 # [cite: 792]
        self.activated_flows.clear() # [cite: 792]
        self.adapt_epsilon(cycle_idx) # [cite: 792]
        count_activated = 0 # [cite: 792]
        for avg_freq in blf.average_freqs: # [cite: 792]
            if abs(avg_freq - self.native_freq) < self.epsilon: # [cite: 792]
                self.activated_flows.append({ # [cite: 792]
                    "freq": avg_freq, # [cite: 792]
                    "flow_hash": blf.identity, # [cite: 792]
                    "time": current_time # [cite: 792]
                })
                count_activated += 1 # [cite: 792]
        self.active = bool(self.activated_flows) # [cite: 792]
        return count_activated # [cite: 792]

    def combine_flows(self): # [cite: 792]
        """
        Combine pairs of flows from the queue, generate new combined
        outputs with an updated frequency = average * factor, plus a hashed ID. [cite: 792]
        """
        self.combined_outputs.clear() # [cite: 793]
        while len(self.activated_flows) >= 2: # [cite: 793]
            f1 = self.activated_flows.popleft() # [cite: 793]
            f2 = self.activated_flows.popleft() # [cite: 793]

            sorted_pair = sorted( # [cite: 793]
                [(f1["flow_hash"], f1["freq"]), (f2["flow_hash"], f2["freq"])], # [cite: 793]
                key=lambda x: (x[0], x[1]) # [cite: 793]
            )
            combined_key = f"{sorted_pair[0][0]}_{sorted_pair[0][1]:.2f}" \
                           f"_{sorted_pair[1][0]}_{sorted_pair[1][1]:.2f}" # [cite: 793]

            if combined_key in self.combination_cache: # [cite: 793]
                combined_hash = self.combination_cache[combined_key] # [cite: 793]
            else: # [cite: 793]
                combined_hash = hashlib.sha256(combined_key.encode()).hexdigest()[:12] # [cite: 793]
                if len(self.combination_cache) >= CONFIG["MAX_NODE_CACHE"]: # [cite: 793]
                    self.combination_cache.pop(next(iter(self.combination_cache))) # [cite: 793]
                self.combination_cache[combined_key] = combined_hash # [cite: 793]

            new_freq = (f1["freq"] + f2["freq"]) * CONFIG["FREQ_COMBINATION_FACTOR"] # [cite: 793]
            self.combined_outputs.append({ # [cite: 793]
                "node_id": self.node_id, # [cite: 793]
                "combined_freq": round(new_freq, 2), # [cite: 793]
                "source_hashes": (f1["flow_hash"][:8], f2["flow_hash"][:8]), # [cite: 793]
                "output_hash": combined_hash # [cite: 793]
            })
        self.activated_flows.clear() # [cite: 793]


    def feed_new_outputs(self, new_outputs): # [cite: 793]
        """
        Collect combined outputs from neighbors if they resonate
        with the node's native frequency. [cite: 793]
        """
        for output in new_outputs: # [cite: 794]
            freq = output["combined_freq"] # [cite: 794]
            if abs(freq - self.native_freq) < self.epsilon: # [cite: 794]
                self.activated_flows.append({ # [cite: 794]
                    "freq": freq, # [cite: 794]
                    "flow_hash": output["output_hash"], # [cite: 794]
                    "time": {"T_epoch": output.get("T_epoch", 0), "T_fine": 0.0} # [cite: 794]
                })
        self.active = bool(self.activated_flows) # [cite: 794]

###############################################################################
# 7. OPTIMIZED FRACTAL MEMORY (STORING COMBINED OUTPUTS) + MAX LIMIT
###############################################################################
class OptimizedFractalMemory: # [cite: 794]
    def __init__(self, max_size=1000): # [cite: 794]
        self.storage = {} # [cite: 794]
        self.max_size = max_size # [cite: 794]

    def store(self, flow, timestamp): # [cite: 794]
        """
        Store each combined output in fractal memory with a capacity limit. [cite: 794]
        If a near-duplicate (same hash & close freq/time) is found,
        we increment occurrences rather than storing anew. [cite: 795]
        """
        hash_key = flow["output_hash"] # [cite: 796]
        epoch = timestamp["T_epoch"] # [cite: 796]

        if hash_key in self.storage: # [cite: 796]
            existing = self.storage[hash_key] # [cite: 796]
            delta_freq = abs(existing["combined_freq"] - flow["combined_freq"]) # [cite: 796]
            delta_time = abs(existing["T_epoch"] - epoch) # [cite: 796]

            # If nearly identical, just increment occurrences [cite: 796]
            if delta_freq < 5 and delta_time <= 2: # [cite: 796]
                existing["occurrences"] += 1 # [cite: 796]
                existing["T_epoch"] = epoch # [cite: 796]
                return # [cite: 796]
            else: # Otherwise create new key [cite: 796]
                new_hash = uuid.uuid5(uuid.NAMESPACE_DNS, hash_key + str(epoch)).hex[:12] # [cite: 796]
                hash_key = new_hash # [cite: 796]

        if len(self.storage) >= self.max_size: # [cite: 796]
            oldest_key = min(self.storage, key=lambda k: self.storage[k]["T_epoch"]) # [cite: 796]
            del self.storage[oldest_key] # [cite: 796]

        self.storage[hash_key] = { # [cite: 796]
            "combined_freq": flow["combined_freq"], # [cite: 796]
            "source_hashes": flow["source_hashes"], # [cite: 796]
            "T_epoch": epoch, # [cite: 796]
            "occurrences": 1 # [cite: 796]
        }

###############################################################################
# 8. NETWORK + NEIGHBOR CHAINING + PARALLEL PROCESSING
###############################################################################
def create_adaptive_network(num_nodes: int): # [cite: 796]
    return [AdaptiveCombinerNode(i) for i in range(num_nodes)] # [cite: 797]

def define_adjacency(network): # [cite: 797]
    adj_type = CONFIG["NODE_TO_NODE_ADJ_TYPE"] # [cite: 797]
    neighbor_count = CONFIG["NEIGHBOR_COUNT"] # [cite: 797]
    total_nodes = len(network) # [cite: 797]

    if adj_type == "ring": # [cite: 797]
        for i, node in enumerate(network): # [cite: 797]
            neighbors = [] # [cite: 797]
            for n in range(1, neighbor_count + 1): # [cite: 797]
                forward = (i + n) % total_nodes # [cite: 797]
                neighbors.append(forward) # [cite: 797]
                backward = (i - n) % total_nodes # [cite: 797]
                neighbors.append(backward) # [cite: 797]
            node.set_neighbors(list(set(neighbors))) # [cite: 797]
    else:  # random adjacency [cite: 798]
        node_ids = [n.node_id for n in network] # [cite: 798]
        for node in network: # [cite: 798]
            possible = [nid for nid in node_ids if nid != node.node_id] # [cite: 798]
            chosen = random.sample(possible, min(neighbor_count, len(possible))) # [cite: 798]
            node.set_neighbors(chosen) # [cite: 798]

def process_node(node, blf, current_time, cycle_idx): # [cite: 798]
    activated_count = node.check_activation(blf, current_time, cycle_idx=cycle_idx) # [cite: 798]
    node.combine_flows() # [cite: 798]
    return activated_count # [cite: 798]

def parallel_node_processing(network, blf, current_time, cycle_idx): # [cite: 798]
    with ThreadPoolExecutor(max_workers=CONFIG["THREAD_WORKERS"]) as executor: # [cite: 798]
        futures = [executor.submit(process_node, node, blf, current_time, cycle_idx) for node in network] # [cite: 798]
        results = [f.result() for f in futures] # [cite: 798]
    return results # [cite: 798]

def neighbor_chaining(network): # [cite: 798]
    outputs_dict = {node.node_id: node.combined_outputs[:] for node in network} # [cite: 798]
    for node in network: # [cite: 798]
        combined_from_neighbors = [] # [cite: 798]
        for neighbor_id in node.neighbors: # [cite: 798]
            combined_from_neighbors.extend(outputs_dict[neighbor_id]) # [cite: 798]
        node.feed_new_outputs(combined_from_neighbors) # [cite: 798]
        node.combine_flows() # [cite: 798]

###############################################################################
# 9. ADAPTIVE FRACTAL RUN + INTEGRATED FFPE
###############################################################################
def main_run(): # [cite: 799]
    print("=== STARTING ENHANCED FRACTAL SYSTEM ===") # [cite: 799]
    logger = Logger() # [cite: 799]
    time_ctrl = TimeController() # [cite: 799]
    fractal_memory = OptimizedFractalMemory(max_size=CONFIG["MAX_FRACTAL_MEMORY"]) # [cite: 799]
    # Build the network [cite: 799]
    network = create_adaptive_network(CONFIG["NUM_NODES"]) # [cite: 799]
    define_adjacency(network) # [cite: 799]
    combined_history = [] # [cite: 799]

    # ------------------------------------------------------------- # [cite: 799]
    # We integrate the AdaptiveFractalProcessor for proof of concept: # [cite: 799]
    # We'll track Phi(t) and Psi(t) for demonstration # [cite: 799]
    # ------------------------------------------------------------- # [cite: 799]
    ffpe = AdaptiveFractalProcessor(num_pins=CONFIG["BLF_SIZE"])  # We create placeholders to log them across cycles [cite: 799]
    all_phi = [] # [cite: 799]
    all_psi = [] # [cite: 799]

    for cycle in range(CONFIG["NUM_CYCLES"]): # [cite: 799]
        # 1) Generate a Bulk Logic Frame [cite: 799]
        flows = generate_binary_flows(CONFIG["BLF_SIZE"], CONFIG["BITS_PER_FLOW"]) # [cite: 799]
        current_time = time_ctrl.now() # [cite: 799]
        blf = BLF(time_data=current_time, flows=flows) # [cite: 799]

        # 2) Node activation in parallel [cite: 800]
        activation_counts = parallel_node_processing(network, blf, current_time, cycle) # [cite: 800]
        for node_idx, activated_count in enumerate(activation_counts): # [cite: 800]
            logger.log_node_activation(cycle, node_idx, activated_count) # [cite: 800]

        # 3) Chain neighbors [cite: 800]
        neighbor_chaining(network) # [cite: 800]

        # 4) Store results in fractal memory [cite: 800]
        for node in network: # [cite: 800]
            if node.combined_outputs: # [cite: 800]
                for output in node.combined_outputs: # [cite: 800]
                    fractal_memory.store(output, current_time) # [cite: 800]
                    output["T_epoch"] = current_time["T_epoch"] # [cite: 800]
                    combined_history.append(output) # [cite: 800]
                node.combined_outputs.clear() # [cite: 800]
        logger.log_memory_size(cycle, len(fractal_memory.storage)) # [cite: 800]

        # --------------------------------------------------------- # [cite: 800]
        # Here, we do a quick demonstration of Φ(t) & Ψ(t) with random data # [cite: 800]
        # for the BLF flows. [cite: 801] This simulates local freq distance D_local, # [cite: 801]
        # global coherence C_global, and impulses Pi. [cite: 801]
        # --------------------------------------------------------- # [cite: 802]
        # For simplicity, let's define random or sinusoidal values: # [cite: 802]
        # f_i = frequencies of each pin (size=BLF_SIZE) [cite: 802]
        f_i = np.random.uniform(0, 100, CONFIG["BLF_SIZE"]) # [cite: 802]
        d_local = np.abs(f_i - np.mean(f_i))  # naive local difference # [cite: 802]
        # e.g. c_global as cos(...) of random phases [cite: 803]
        c_global = np.cos(2 * np.pi * np.random.random(CONFIG["BLF_SIZE"])) # [cite: 803]
        # impulses Pi as random [cite: 803]
        impulses = np.random.random(CONFIG["BLF_SIZE"]) * 0.3 # [cite: 803]

        # Compute fractal priority [cite: 803]
        phi = ffpe.compute_priority(d_local, c_global, impulses) # [cite: 803]
        # Just as an example, define Psi_local and Psi_global as: # [cite: 803]
        # Psi_local ~ 50 - 2*d_local # [cite: 803]
        # Psi_global ~ 50 + 2*c_global [cite: 803]
        psi_local = 50 - 2 * d_local # [cite: 803]
        psi_global = 50 + 2 * c_global # [cite: 803]
        psi = ffpe.adaptive_response(phi, psi_local, psi_global) # [cite: 803]

        # Optionally compute VFE [cite: 803]
        vfe_val = ffpe.compute_vfe(phi, c_global, f_i, impulses) # [cite: 803]

        # Save these arrays for demonstration [cite: 803]
        all_phi.append(phi) # [cite: 803]
        all_psi.append(psi) # [cite: 803]

        print(f"[CYCLE={cycle}] BLF Hash={blf.identity[:8]}, " # [cite: 803]
              f"Flows={len(flows)} | Memory Size={len(fractal_memory.storage)}") # [cite: 803]

    # Let's flatten them for final display [cite: 804]
    final_phi = np.concatenate(all_phi) # [cite: 804]
    final_psi = np.concatenate(all_psi) # [cite: 804]

    return network, fractal_memory, combined_history, logger, final_phi, final_psi # [cite: 804]

###############################################################################
# 10. MINI FRACTAL DEMO TEST (GRAPHICAL SMALL EXAMPLE)
###############################################################################
def mini_fractal_demo_test(): # [cite: 804]
    print("\n=== MINI FRACTAL DEMO TEST ===") # [cite: 804]
    num_flows = 10 # [cite: 804]
    num_nodes = 5 # [cite: 804]
    base_freq = 1000 # [cite: 804]
    delta_f = 20 # [cite: 804]

    def generate_data(num_flows, base_freq, delta_f): # [cite: 804]
        data = [] # [cite: 804]
        for _ in range(num_flows): # [cite: 804]
            flow = np.random.choice([0, 1], 16)  # 16-bit flow [cite: 804]
            frequencies = base_freq + delta_f * np.arange(16) * flow # [cite: 804]
            data.append(frequencies) # [cite: 804]
        return np.array(data) # [cite: 804]

    data = generate_data(num_flows, base_freq, delta_f) # [cite: 804]

    def process_data(data, num_nodes): # [cite: 804]
        results = np.zeros((data.shape[0], num_nodes)) # [cite: 804]
        for i in range(num_nodes): # [cite: 804]
            # each node adds i * 50 to the mean freq of that flow [cite: 805]
            results[:, i] = np.mean(data, axis=1) + (i * 50) # [cite: 805]
        return results # [cite: 805]

    processed_data = process_data(data, num_nodes) # [cite: 805]

    plt.figure(figsize=(10, 5)) # [cite: 805]
    for i in range(num_flows): # [cite: 805]
        plt.plot(np.arange(1, num_nodes + 1), processed_data[i, :], label=f"Flow {i+1}") # [cite: 805]
    plt.title("Fractal Processor: Frequency Processing of Data Flows (Mini Demo)") # [cite: 805]
    plt.xlabel("Nodes") # [cite: 805]
    plt.ylabel("Frequency (Hz)") # [cite: 805]
    plt.grid(True) # [cite: 805]
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) # [cite: 805]
    plt.tight_layout() # [cite: 805]
    plt.show() # [cite: 805]

    df_processed_data = pd.DataFrame(processed_data, columns=[f"Node {j+1}" for j in range(num_nodes)]) # [cite: 805]
    df_processed_data.insert(0, "Flow", [f"Flow {k+1}" for k in range(num_flows)]) # [cite: 805]
    print("\n--- MINI DEMO TABLE ---") # [cite: 805]
    if tools is not None: # [cite: 805]
        tools.display_dataframe_to_user( # [cite: 805]
            name="Processed Fractal Data (Mini Demo)", dataframe=df_processed_data # [cite: 805]
        )
    else: # [cite: 805]
        print(df_processed_data.to_string(index=False)) # [cite: 805]

###############################################################################
# 11. MAIN EXECUTION (PROOF OF CONCEPT)
###############################################################################
if __name__ == "__main__": # [cite: 806]
    network, fractal_memory, combined_history, logger, final_phi, final_psi = main_run() # [cite: 806]

    print("\n=== FINAL RESULTS ===") # [cite: 806]

    # Display combined history (sample) [cite: 806]
    df_history = pd.DataFrame([ # [cite: 806]
        { # [cite: 806]
            "Epoch": item.get("T_epoch", 0), # [cite: 806]
            "Node": item["node_id"], # [cite: 806]
            "Frequency (Hz)": item["combined_freq"], # [cite: 806]
            "Result Hash": item["output_hash"], # [cite: 806]
            "Sources": f"{item['source_hashes'][0]} + {item['source_hashes'][1]}" # [cite: 806]
        } for item in combined_history # [cite: 806]
    ])
    print("\n--- COMBINED FLOWS HISTORY (Sample) ---") # [cite: 806]
    print(df_history.head(30).to_string(index=False)) # [cite: 806]

    # Display fractal memory content (sample) [cite: 806]
    df_memory = pd.DataFrame([ # [cite: 806]
        { # [cite: 806]
            "Hash": key, # [cite: 806]
            "Frequency (Hz)": round(data["combined_freq"], 2), # [cite: 806]
            "Occurrences": data["occurrences"], # [cite: 806]
            "Epoch": data["T_epoch"], # [cite: 806]
            "Sources": f"{data['source_hashes'][0]} + {data['source_hashes'][1]}" # [cite: 806]
        } for key, data in fractal_memory.storage.items() # [cite: 806]
    ])
    print("\n--- ENHANCED FRACTAL MEMORY CONTENT (Sample) ---") # [cite: 806]
    print(df_memory.head(30).to_string(index=False)) # [cite: 806]

    logger.print_summary() # [cite: 806]
    print("\n=== END OF ENHANCED FRACTAL SYSTEM RUN ===") # [cite: 807]

    # Print the final arrays of Phi and Psi (just to show the user) [cite: 807]
    print("\nFractal Priority (Phi):", final_phi) # [cite: 807]
    print("Adaptive Response (Psi):", final_psi) # [cite: 807]

    # Run the smaller demonstration test [cite: 807]
    mini_fractal_demo_test() # [cite: 807]
    print("\n=== END OF MINI FRACTAL DEMO TEST ===") # [cite: 807]

# Theoretical Foundation of PSDFF

The PSDFF system, while implemented pragmatically, draws inspiration from concepts related to fractal dynamics, resonance, and emergent systems. The integration of the `AdaptiveFractalProcessor` class aims to formalize some of these ideas.

## Core Principles

1.  **Fractal Processing:**
    * The term "fractal" here relates to the self-similar nature of processing logic across different scales (individual nodes, network interactions) and the emergent complexity arising from simple, repeated rules (resonance, combination).
    * The "fractal chaining" where outputs of nodes become inputs for others can lead to complex, evolving processing pathways[cite: 759, 810].

2.  **Spectral Resonance:**
    * Data flows are converted into numeric "frequency" signatures[cite: 7, 153].
    * Nodes activate when the "distance" between a flow's average frequency and the node's native frequency is below a threshold (`epsilon`)[cite: 9, 153]. This is analogous to resonance in physical systems.
    * The `epsilon` can be adaptive, meaning the system can tune its sensitivity[cite: 393, 434, 614, 808].

3.  **Emergent Combination and Memory:**
    * New data flows are created by combining existing resonant flows[cite: 10, 11, 154].
    * Fractal Memory stores these emergent flows. The "overlap" mechanism, where similar new flows reinforce existing entries rather than creating duplicates, reflects a form of emergent pattern recognition or learning[cite: 12, 13, 159, 731].

## Mathematical Concepts (as discussed and partially implemented)

1.  **Frequency Combination:**
    * The combination of two frequencies <span class="math-inline">f\_1</span> and <span class="math-inline">f\_2</span> to produce a new frequency <span class="math-inline">f\_c</span> is given by:
        <span class="math-inline">f\_c \= \(f\_1 \+ f\_2\) \\times \\text\{FREQ\_COMBINATION\_FACTOR\}</span>[cite: 11, 31, 79, 423, 463, 556, 668, 793].
    * A simplified version discussed was <span class="math-inline">f\_c \= \\frac\{f\_1 \+ f\_2\}\{2\}</span> (when `FREQ_COMBINATION_FACTOR` is 0.5)[cite: 765]. This aims for a form of averaging or mean-seeking behavior.

2.  **Fractal Memory Emergence Condition:**
    * An entry <span class="math-inline">f\_\{new\}</span> overlaps with an existing entry <span class="math-inline">f\_\{existent\}</span> in memory if:
        <span class="math-inline">\|f\_\{new\}\[\\text\{combined\_freq\}\] \- f\_\{existent\}\[\\text\{combined\_freq\}\]\| < \\text\{threshold\}\_\{\\text\{freq\}\}</span>
        AND
        <span class="math-inline">\|f\_\{new\}\[\\text\{T\_epoch\}\] \- f\_\{existent\}\[\\text\{T\_epoch\}\]\| \\le \\text\{threshold\}\_\{\\text\{time\}\}</span>[cite: 13, 33, 82, 425, 441, 464, 506, 562, 633, 669, 796].
    * The discussion mentioned a "tolerance <span class="math-inline">\\delta</span> for emergence"[cite: 767, 769], which corresponds to these thresholds.

3.  **Adaptive Fractal Processor (FFPE) Model Equations**[cite: 785]:
    * **Fractal Priority Φ(t):**
        <span class="math-inline">\\Phi\(t\) \= \\sigma\(\\gamma\_1 \\cdot D\_\{\\text\{local\}\} \- \\gamma\_2 \\cdot C\_\{\\text\{global\}\} \+ \\gamma\_3 \\cdot \\Pi\(t\)\)</span> [cite: 785, 786]
        Where:
        * <span class="math-inline">\\sigma</span> is the sigmoid function: <span class="math-inline">\\sigma\(x\) \= \\frac\{1\}\{1 \+ e^\{\-x\}\}</span>[cite: 785].
        * <span class="math-inline">D\_\{\\text\{local\}\}</span>: Local data divergence/distance (e.g., deviation of a flow's frequency from a local mean).
        * <span class="math-inline">C\_\{\\text\{global\}\}</span>: Global coherence measure (e.g., how synchronized or similar flows are across the system).
        * <span class="math-inline">\\Pi\(t\)</span>: Incoming impulses or external stimuli.
        * <span class="math-inline">\\gamma\_1, \\gamma\_2, \\gamma\_3</span>: Weighting coefficients.
        This function determines the processing priority of a data unit (pin/flow).

    * **Adaptive Response Ψ(t):**
        <span class="math-inline">\\Psi\(t\) \= \\Phi\(t\) \\cdot \\psi\_\{\\text\{local\}\} \+ \(1 \- \\Phi\(t\)\) \\cdot \\psi\_\{\\text\{global\}\}</span> [cite: 785]
        Where:
        * <span class="math-inline">\\psi\_\{\\text\{local\}\}</span>: Local response state/behavior.
        * <span class="math-inline">\\psi\_\{\\text\{global\}\}</span>: Global response state/behavior.
        This modulates the actual response or processing applied, balancing local needs with global system state based on the calculated priority.

    * **Emergent Fractal Velocity VFE(t):**
        <span class="math-inline">VFE\(t\) \= \\frac\{1\}\{T\} \\sum\_\{i\=1\}^\{N\} \\frac\{\\Phi\_i\(t\) \\cdot C\_\{\\text\{global\},i\}\(t\) \\cdot f\_i \\cdot \\Pi\_i\(t\)\}\{N\}</span> (simplified from source, assuming T=1 for a single step)[cite: 785, 787].
        Where:
        * <span class="math-inline">N</span>: Number of pins/flows being considered.
        * <span class="math-inline">f\_i</span>: Frequency of the <span class="math-inline">i</span>-th pin/flow.
        This metric quantifies the overall rate or momentum of emergent information processing within the system.

These theoretical elements aim to provide a more principled way to guide the adaptive behavior of the fractal nodes and the overall system dynamics, moving beyond purely heuristic rules. The current implementation includes a demonstrative calculation of these values[cite: 801, 802, 803].

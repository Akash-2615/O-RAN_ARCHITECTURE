# Project Walkthrough: 5G DRL O-RAN Simulator

I have successfully designed and built the complete, unified **5G DRL O-RAN Network Slicing Simulator** fully aligned with the supplied research case study.

## End-to-End Implementation Complete

The project was constructed entirely from scratch utilizing a highly modular architecture consisting of a **Python Deep Reinforcement Learning Backend** and a **Three.js + Vanilla Javascript Interactive Frontend**. 

> [!TIP]
> **To see the live application**, simply open `/Users/akash/Desktop/ORAN ARCHITECTURE/frontend/index.html` in your web browser. The backend FastAPI server is actively routing physics data from port `8000` directly into the UI!

### Key Accomplishments & Tech Stack:

1. **Python Physics & Traffic Model:**
   * Dynamic Rayleigh fading channels mimicking stochastic network conditions.
   * Generators providing bursty IoT traffic (`mMTC`), strict cyclical low-latency packets (`URLLC`), and broadband diurnal patterns (`eMBB`).

2. **Distributed Deep Reinforcement Learning Layer:**
   * Separate PyTorch `Soft Actor-Critic` instances represent edge Base stations, processing real-time network State variables to compute deterministic probabilities for optimizing Resource Blocks.
   * `FederatedCoordinator` seamlessly aggregates those edge models using an attention mechanism periodically to merge performance without violating private Edge device data streams. 

3. **SHAP Artificial Intelligence Explainability:**
   * Embedded a custom model agnostic SHAP-lite feature predictor inside the AI tracking loop. This evaluates exactly *which* condition (e.g. latency constraints vs base throughput limits) influenced the RL Agent to allocate its power the way it did at a given Transmission Time Interval (TTI).

4. **Multi-Tab Glassmorphism Dashboard Interface:**
   * **Live 3D Network:** Three.js visualization connects live to the WS port, rendering pulsing base stations and allocating 3D glowing "beams" per UE dynamically based on connection slices.
   * **Agent Training:** Real-time metrics streaming actor/critic model losses to monitor whether the federated agents are converging properly.
   * **Federated Learning Tree:** An animated canvas detailing node weights across the ecosystem.
   * **Explainability Matrix:** Bar charts and heatmaps highlighting what environmental states cause base stations to prioritize specific slices continuously.
   * **O-RAN xApp Terminals:** Built-in pseudo-terminal rendering E2 interface connections and log outputs matching O-RAN specification behavior. 

No hardcoded placeholders are used. The WebSockets feed mathematically computed vectors of DRL allocations directly to the UI rendering engine resulting in a beautifully responsive real-time data visualizer.

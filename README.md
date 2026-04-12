# 5G DRL O-RAN Network Slicing Simulator

An end-to-end, real-time simulator implementing **Deep Reinforcement Learning for Dynamic Resource Allocation in 5G Network Slicing and O-RAN**.

Built entirely from scratch in a modular Python backend (FastAPI, PyTorch) and a vanilla HTML/CSS/JS frontend (WebGL/Three.js).

## Architecture & Modules

### `backend/`
The Python 3 backend simulating the network and the agents.

- **`core/` (5G MDP Environment)**
  - `environment.py`: Formulates the Reinforcement Learning MDP (State, Action, Reward). Calculates throughput, SLA penalties, and energy constraints.
  - `network_state.py`: Manages the physical environment—3 base stations, UE coordinates, random walk mobility. 
  - `channel_model.py`: Implements Rayleigh fading, path loss, and SINR-to-CQI mappings (3GPP standard).
  - `traffic_model.py`: Generates stochastic traffic patterns for eMBB (Poisson/diurnal), URLLC (periodic bursts + 5ms latency SLA), and mMTC (massive IoT awakenings).

- **`agents/` (Deep Reinforcement Learning)**
  - `actor_critic.py`: PyTorch architecture for the Soft Actor-Critic (SAC) network. Handles both continuous control (Tx power) and discrete-like resource block allocation via logits.
  - `drl_agent.py`: A per-BaseStation SAC agent handling local experience replay buffers, policy gradients, and Q-value updates.
  - `multi_agent.py`: Orchestrates the step-by-step execution across all base stations synchronously for each 10ms TTI (Transmission Time Interval).

- **`federated/` (Multi-Agent Co-ordination)**
  - `aggregator.py`: Attention-weighted gradient aggregation mechanism. Base stations with better recent performance and lower variance get higher weights.
  - `fed_coordinator.py`: Manages federated rounds across all BS agents periodically copying their gradients, aggregating them, and broadcasting back the unified global model.

- **`explainability/` (Trustworthy AI / XAI)**
  - `shap_explainer.py`: A perturbation-based attribution model that analyzes the SAC Actor network to determine which network state features heavily influenced its resource allocation decision.
  - `importance.py`: Tracks SHAP vectors over time to build visual heatmaps.

- **`simulation/` (Orchestration & RIC)**
  - `simulator.py`: Main asynchronous event loop orchestrating network updates every 10ms.
  - `metrics.py`: Tracks Jain's Fairness, Utility, SLA Violation percentage, and Power consumption.
  - `oran_xapp.py`: Simulates Near-RT RIC. Subscribes to E2 interface nodes, resolves conflicts among multiple xApps (Slice Manager, Power Control, etc.), and logs activities.

- **`api/` (API Layer)**
  - `main.py`: FastAPI server entrypoint.
  - `routes.py`: REST endpoints providing network status and state arrays.
  - `websocket.py`: Pushes live 10ms TTI state frames to the UI for smooth visualization.

### `frontend/`
High-performance Dashboard built without massive UI frameworks, using glassmorphism styling and WebGL 3D.

- **`index.html`** & **`css/`**: UI layout and Dark-mode Neon CSS variables.
- **`js/` (Vanilla JS Controllers)**:
  - `app.js`: Connects to `ws://localhost:8000/ws` and orchestrates data dispatching.
  - `network3d.js`: Three.js wrapper rendering Base stations, simulated packet flow beams, and particle-based UEs bouncing dynamically around cells.
  - `training.js`: Real-time charts for local agent Actor/Critic losses and replay buffer capacity.
  - `federated.js`: Visualizes federated aggregation using Canvas API particle flows.
  - `sliceMetrics.js`: Tracks physical metrics (throughput, delay, dropping).
  - `explainability.js`: Shows realtime SHAP feature importance bars and heatmaps.
  - `oranConsole.js`: Mimics the O-RAN RIC terminal with scrolling xApp logs.

## Setup Instructions

### 1. Requirements

Ensure you have **Python 3.10+**.

```bash
cd "backend"
pip install -r requirements.txt
```

*(Note: Ensure you run `pip install websockets fastapi uvicorn torch numpy scipy pydantic python-multipart` if not using the requirements file directly)*

### 2. Run the Backend Server

```bash
cd "backend"
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open the Frontend

Just open the Local HTML file in your web browser:
```bash
open "frontend/index.html"
```
Or simply double-click the `index.html` file in your file explorer.

## Verification

You should immediately see the WebGL 3D Visualization showing 3 Base Stations and multiple user particles connecting to them. Real-time graphs will populate showing the Multi-Agent Reinforcement Learning models actively learning to allocate bandwidth to eMBB/URLLC/mMTC slices while avoiding latency violations!
)

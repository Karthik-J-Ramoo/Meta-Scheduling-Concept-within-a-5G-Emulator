# RAN Meta-Scheduling Simulator

A sophisticated discrete-event simulation framework for exploring **Resource Allocation and Network (RAN) meta-scheduling strategies** in 5G/6G systems. This simulator models a **closed-loop control system** where network slices (URLLC, eMBB, IoT) compete for Physical Resource Blocks (PRBs) under Service-Level Agreement (SLA) constraints.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Core Concepts](#core-concepts)
- [Components](#components)
- [Execution Flow](#execution-flow)
- [Output & Results](#output--results)
- [Extensibility](#extensibility)
- [Research Applications](#research-applications)

---

## Overview

The **RAN Meta-Scheduling Simulator** models the hierarchical resource orchestration in modern cellular networks:

- **Network Slices**: Three slice types (URLLC, eMBB, IoT) with distinct traffic patterns and SLA requirements
- **Physical Resource Blocks (PRBs)**: Fixed-budget shared resource (~100 PRBs in the default configuration)
- **Closed-Loop Control**: SLA violations trigger dynamic PRB reallocation via the meta-scheduler
- **Stochastic Traffic**: Poisson-based arrival processes with burst overlays per slice
- **SLA Evaluation**: Latency, throughput, and relaxed SLA types with per-timestep violation detection

The simulator operates in a **discrete-event** manner with configurable timestep granularity, making it ideal for rapid prototyping and validation of allocation algorithms before real-world deployment.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Simulation Main Loop                      │
└─────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
   ┌──────────┐         ┌─────────┐         ┌──────────┐
   │ Traffic  │         │   DU    │         │  SLA     │
   │Generator │         │Scheduler│         │ Checker  │
   └──────────┘         └─────────┘         └──────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Performance     │
                    │ Monitor (CSV)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │Meta-Scheduler   │
                    │ (Decides PRBs)  │
                    └─────────────────┘
                             │
                   Feedback to next timestep
```

---

## Key Features

✅ **Hierarchical Scheduling**
- **Inter-slice**: Meta-scheduler allocates PRBs to slices based on SLA violations
- **Intra-slice**: DU scheduler drains queues using allocated PRBs

✅ **Network Slice Modeling**
- URLLC (Ultra-Reliable Low-Latency): Strict latency SLA (< 5ms)
- eMBB (Enhanced Mobile Broadband): Throughput-oriented SLA (> 30 packets/ts)
- IoT (Internet of Things): Relaxed SLA for low-priority traffic

✅ **Stochastic Traffic**
- Poisson base arrival process per slice
- Bursty overlay (configurable probability & multiplier)
- Per-slice randomization via shared numpy Generator (reproducible)

✅ **SLA-Driven Adaptation**
- Rule-based meta-scheduler with fairness decay
- Violation-driven resource reallocation (up to `reallocation_step` PRBs/slice)
- Minimum PRB floor to prevent slice starvation

✅ **Comprehensive Metrics**
- Per-timestep: arrivals, served packets, queue length, drops, latency, throughput
- Per-slice summaries: average latency, average throughput, SLA violation count
- Rolling statistics (configurable window size)

✅ **CSV Logging**
- Incremental metric export for easy post-simulation analysis
- 13 fields per row: timestep, slice metrics, SLA status
- Suitable for Python/Pandas analysis or external visualization

---

## Project Structure

```
ran_simulator/
├── main.py                          # Entry point; orchestrates simulation loop
├── config/
│   ├── __init__.py
│   └── simulation_config.py         # Central configuration registry
├── core/
│   ├── __init__.py
│   ├── slice.py                     # NetworkSlice & SliceMetrics dataclasses
│   ├── traffic_generator.py         # Poisson + burst traffic model
│   ├── du_scheduler.py              # Intra-slice capacity-limited scheduler
│   ├── meta_scheduler.py            # Inter-slice PRB allocator (rule-based)
│   ├── sla_checker.py               # SLA violation detection
│   └── performance_monitor.py       # Metrics aggregation & CSV export
├── utils/
│   ├── __init__.py
│   └── logger.py                    # Logging setup & CSV writer
└── results/
    ├── metrics.csv                  # Generated metrics (one row per slice/timestep)
    └── simulation.log               # Logging output

Learnings/                           # Reference materials
├── Actual_Emulator_Stack.txt
├── Paper_Logic.txt
├── Manuscript.txt
└── v1-Flow.txt
```

---

## Installation

### Prerequisites

- **Python 3.8+**
- **NumPy** (for random number generation and array operations)

### Setup

1. **Clone or navigate to the project**:
   ```bash
   cd Meta_Scheduling_Test
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy
   ```

3. **(Optional) Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   pip install numpy
   ```

---

## Quick Start

### Run with Default Configuration

```bash
cd ran_simulator
python main.py
```

**Output**:
- Console logs showing simulation progress, SLA violations, and meta-scheduler decisions
- `results/metrics.csv`: Per-slice KPIs for all timesteps
- `results/simulation.log`: Detailed debug logs

### Run with Custom Parameters

Override simulation parameters via CLI arguments:

```bash
# Run for 500 timesteps with custom seed
python main.py --steps 500 --seed 7

# Run for 1000 timesteps with default seed
python main.py --steps 1000
```

### Example Console Output

```
2026-03-30 12:34:56 INFO Simulation starts │ steps=200  seed=42
2026-03-30 12:34:56 INFO Slice URLLC created  │ type=urllc   │ init_prbs=40  │ capacity=200 │ sla=latency<5.0
2026-03-30 12:34:56 INFO Slice eMBB created   │ type=embb    │ init_prbs=40  │ capacity=500 │ sla=throughput<30.0
2026-03-30 12:34:56 INFO Slice IoT created    │ type=iot     │ init_prbs=20  │ capacity=300 │ sla=relaxed<0.0
2026-03-30 12:34:56 INFO Components ready │ slices=['URLLC', 'eMBB', 'IoT']
   [... simulation running ...]
2026-03-30 12:34:57 WARNING t=45 │ SLA VIOLATION │ URLLC │ type=latency  value=6.32  threshold=5.00
2026-03-30 12:34:57 INFO t=46 │ MetaSched │ VIOLATIONS ['URLLC'] │ boost=5/slice  donor_budget=60
   [... more timesteps ...]
2026-03-30 12:35:01 ────────────────────────────────────────────────────────
2026-03-30 12:35:01 Simulation complete │ 5.23 s wall-clock
2026-03-30 12:35:01 ────────────────────────────────────────────────────────
2026-03-30 12:35:01 URLLC  │ avg_lat=4.85  avg_tput=26.3  avg_q=15.2  drops=8   sla_violations=12/20
2026-03-30 12:35:01 eMBB   │ avg_lat=8.92  avg_tput=34.1  avg_q=42.1  drops=3   sla_violations=2/20
2026-03-30 12:35:01 IoT    │ avg_lat=12.15 avg_tput=9.2   avg_q=25.8  drops=5   sla_violations=0/20
2026-03-30 12:35:01 Results written to results/ (CSV + log)
```

---

## Configuration

All parameters are centralized in **`config/simulation_config.py`**:

### Global Simulation Parameters

```python
SIMULATION = {
    "total_prbs": 100,          # Total Physical Resource Blocks
    "num_timesteps": 200,       # Duration of one episode
    "random_seed": 42,          # For reproducibility (None → non-deterministic)
}
```

### Network Slice Definitions

Each slice is a dict with:
- **name**: Human-readable label (e.g., "URLLC")
- **slice_type**: One of `{"urllc", "embb", "iot"}`
- **initial_prbs**: PRB allocation at t=0
- **queue_capacity**: Max packets before overflow
- **service_rate**: Packets served per PRB per timestep
- **sla_type**: `"latency"` | `"throughput"` | `"relaxed"`
- **sla_threshold**: Numeric constraint for SLA metric
- **traffic_params**:
  - `mean_arrival`: Poisson λ (packets/timestep)
  - `burst_prob`: Probability of burst event
  - `burst_mult`: Multiplier on λ during burst

**Example URLLC Configuration**:
```python
{
    "name": "URLLC",
    "slice_type": "urllc",
    "initial_prbs": 40,
    "queue_capacity": 200,
    "service_rate": 1.5,       # Aggressive service per PRB
    "sla_type": "latency",
    "sla_threshold": 5.0,      # max tolerable latency
    "traffic_params": {
        "mean_arrival": 20,
        "burst_prob": 0.10,
        "burst_mult": 3.0,
    },
}
```

### Meta-Scheduler Configuration

```python
META_SCHEDULER = {
    "reallocation_step": 5,     # PRBs to shift per adjustment
    "fairness_decay": 0.02,     # Drift rate back to initial share
    "min_prbs_per_slice": 5,    # Minimum PRB floor
}
```

### Logging Configuration

```python
LOGGING = {
    "console_level": "INFO",    # DEBUG | INFO | WARNING | ERROR
    "file_level": "DEBUG",
    "log_dir": "results",       # Output directory
    "csv_filename": "metrics.csv",
    "log_filename": "simulation.log",
}
```

---

## Core Concepts

### Network Slice (NetworkSlice)

A logical isolated network partition with:
- **Queue**: Holds arriving packets (bounded capacity)
- **State**: Per-timestep counters (arrivals, served, dropped, latency, throughput)
- **SLA**: Service-level agreement (latency/throughput/relaxed)
- **Metrics**: Immutable snapshots (`SliceMetrics`) for analysis

### Traffic Generation (TrafficGenerator)

Per-slice stochastic traffic source:
- **Base model**: Poisson process with rate λ
- **Burst overlay**: Bernoulli trial each timestep; if true, λ → λ × burst_mult
- **Per-slice RNG**: Ensures reproducibility and independent randomness

### DU Scheduling (DUScheduler)

Intra-slice capacity-limited resource consumption:
```
max_serve = floor(allocated_prbs × service_rate)
served = min(queue_length, max_serve)
```
- Simple, deterministic policy
- Extensible: subclass to implement proportional-fair, round-robin, or ML-driven policies

### SLA Checking (SLAChecker)

Per-timestep SLA evaluation:
- **Latency SLA**: `latency ≤ threshold`
- **Throughput SLA**: `throughput ≥ threshold`
- **Relaxed SLA**: Always satisfied (for best-effort traffic)
- Returns: `{slice_name: violated}` dict for meta-scheduler feedback

### Meta-Scheduling (MetaScheduler)

Rule-based inter-slice PRB allocation:

1. **Detect violations** from SLA checker
2. **Boost violated slices**: +`reallocation_step` PRBs each (if donors available)
3. **Shrink donors**: Non-violated slices donate proportionally (floor: `min_prbs_per_slice`)
4. **Fairness decay**: When all SLAs satisfied, nudge allocations back toward initial shares
5. **Budget clamp**: Ensure total PRBs = `total_prbs` via greedy residual distribution

### Performance Monitoring (PerformanceMonitor)

Metrics aggregation and export:
- **Per-timestep**: Record `SliceMetrics` snapshots
- **Rolling statistics**: Per-slice moving averages (configurable window)
- **CSV export**: Incremental write of all metrics
- **Final summary**: Aggregated KPIs (avg latency, throughput, SLA violations, drops)

---

## Components

### `main.py` – Entry Point & Simulation Loop

**Responsibilities**:
- Parse CLI arguments (`--steps`, `--seed`)
- Initialize slices, traffic generators, schedulers, and monitors
- Run discrete-event loop for N timesteps
- Log summary statistics and flush results

**Key Loop Steps** (per timestep):
1. **Traffic generation**: Sample arrivals, enqueue packets
2. **DU scheduling**: Drain slices up to allocated PRB capacity
3. **Metric snapshots**: Create immutable `SliceMetrics` records
4. **SLA evaluation**: Stamp metrics with satisfaction status
5. **Performance recording**: Log metrics to CSV
6. **Meta-scheduling**: Adjust PRBs for next timestep based on violations

**Usage**:
```bash
python main.py [--steps N] [--seed S]
```

---

### `config/simulation_config.py` – Configuration Registry

**Design principle**: All knobs in one place for reproducibility and experiment diffing.

**Sections**:
- `SIMULATION`: Global parameters (total PRBs, timesteps, seed)
- `SLICES`: List of dicts defining each slice's behavior
- `META_SCHEDULER`: PRB allocation hyperparameters
- `LOGGING`: Output directories and verbosity levels

**Extensibility**: Slice dicts are JSON-serializable, suitable for future YAML/JSON loading.

---

### `core/slice.py` – Network Slice Entity

**Classes**:
- **`SliceMetrics`**: Immutable dataclass capturing per-timestep KPIs
  - Fields: timestep, slice_name, arrivals, served, queue_length, dropped, latency, throughput, sla_satisfied
  - Used by `PerformanceMonitor`, `SLAChecker`

- **`NetworkSlice`**: Mutable entity managing per-slice state
  - Attributes: queue, allocated_prbs, arrivals, served, dropped, latency, throughput
  - Methods: `enqueue(packets)`, `dequeue(max_packets)`, `compute_metrics()`, `snapshot()`
  - Does NOT contain scheduling logic (delegated to `DUScheduler` and `MetaScheduler`)

---

### `core/traffic_generator.py` – Stochastic Traffic Model

**Class**: `TrafficGenerator`

**Parameters**:
- `slice`: Reference to the `NetworkSlice`
- `rng`: Shared `numpy.random.Generator` for reproducibility

**Method**: `generate(timestep: int) -> int`
- Samples Poisson random variable with rate λ
- With probability `burst_prob`, scales λ by `burst_mult`
- Returns number of packet arrivals

**Extensibility**: Replace with trace-driven or GAN-driven traffic by implementing same interface.

---

### `core/du_scheduler.py` – Intra-Slice Scheduler

**Class**: `DUScheduler`

**Method**: `schedule(slices, timestep)`
- For each slice: compute `max_serve = floor(allocated_prbs × service_rate)`
- Dequeue `min(queue_length, max_serve)` packets
- Call `s.compute_metrics()` to update throughput/latency

**Extensibility**: Subclass to implement:
- Proportional-fair scheduling
- Round-robin within bursts
- ML-driven intra-slice policies

---

### `core/meta_scheduler.py` – Inter-Slice Allocator

**Class**: `MetaScheduler`

**Method**: `allocate(slices, violations, timestep) -> dict[str, int]`
- **Input**: Slices, violation flags, current timestep
- **Algorithm**:
  1. Boost violated slices by `reallocation_step` PRBs (if donors available)
  2. Shrink safe slices proportionally (respecting `min_prbs_per_slice` floor)
  3. Apply fairness decay toward initial shares (when no violations)
  4. Hard-clamp budget to `total_prbs` via greedy residual distribution
- **Output**: Dict mapping slice_name → allocated_prbs

**Extensibility**: Subclass to plug in:
- Active Inference (free-energy minimisation)
- Deep RL (PPO, SAC, A3C)
- Model-predictive control
- ChaosNet-driven allocation

---

### `core/sla_checker.py` – SLA Evaluation

**Class**: `SLAChecker`

**Method**: `check_all(slices, snapshots, timestep) -> (violations, updated_snapshots)`
- For each slice, check if SLA is satisfied:
  - **Latency**: `latency ≤ threshold`
  - **Throughput**: `throughput ≥ threshold`
  - **Relaxed**: Always pass
- Return violation dict and stamped metrics

**Extensibility**: Override `check_slice()` to implement:
- ChaosNet SLA-violation prediction (proactive)
- Composite SLA constraints (latency AND throughput)
- Probabilistic SLA (e.g., 99th percentile)

---

### `core/performance_monitor.py` – Metrics Aggregation

**Class**: `PerformanceMonitor`

**Constructor Parameters**:
- `csv_writer`: Optional `MetricCSVWriter` for incremental CSV export
- `window`: Size of rolling window for moving averages

**Key Methods**:
- `record(snapshots)`: Accumulate timestep metrics
- `flush()`: Finalize CSV writes
- `summary() -> dict`: Return per-slice aggregated stats

**Output Structure**:
```
{
    "URLLC": {
        "avg_latency": 4.85,
        "avg_throughput": 26.3,
        "avg_queue": 15.2,
        "total_dropped": 8,
        "sla_violations": 12,
        "window_size": 20,
    },
    ...
}
```

**Extensibility**: Subscribe to `record()` calls to add Prometheus/InfluxDB exporters or live dashboards.

---

### `utils/logger.py` – Logging & CSV Export

**Functions**:
- `setup_logger()`: Configure console and file logging
- `MetricCSVWriter`: Streaming CSV writer for metrics

**CSV Schema**:
```
timestep, slice_name, slice_type, allocated_prbs, arrivals, served,
queue_length, dropped, throughput, latency, sla_type, sla_threshold,
sla_satisfied
```

---

## Execution Flow

### One Complete Episode

```
┌─── Initialize ────────────────────────────────────────────┐
│ 1. Load config (slices, PRBs, timestamps)               │
│ 2. Create slices, traffic generators, schedulers        │
│ 3. Set up logging (console + file + CSV)                │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │ For t = 0 to num_steps  │
        └────────────┬────────────┘
                     │
      ┌──────────────▼──────────────┐ (step 1)
      │ Traffic Generation          │
      │ • Sample Poisson(λ)        │
      │ • Apply burst overlay      │
      │ → enqueue packets          │
      └──────────────┬──────────────┘
                     │
      ┌──────────────▼──────────────┐ (step 2)
      │ DU Scheduling              │
      │ • For each slice:          │
      │   - Compute max_serve      │
      │   - Dequeue packets        │
      │   - Update latency/tput    │
      └──────────────┬──────────────┘
                     │
      ┌──────────────▼──────────────┐ (step 3)
      │ Metric Snapshots           │
      │ • Create SliceMetrics      │
      │   immutable records        │
      └──────────────┬──────────────┘
                     │
      ┌──────────────▼──────────────┐ (step 4)
      │ SLA Checking               │
      │ • Evaluate constraints     │
      │ • Stamp violations         │
      │ → violation dict           │
      └──────────────┬──────────────┘
                     │
      ┌──────────────▼──────────────┐ (step 5)
      │ Performance Recording      │
      │ • Log to monitor           │
      │ • Write to CSV             │
      └──────────────┬──────────────┘
                     │
      ┌──────────────▼──────────────┐ (step 6)
      │ Meta-Scheduling            │
      │ • Adjust PRBs for t+1      │
      │ • Boost violated slices    │
      │ • Apply fairness decay     │
      └──────────────┬──────────────┘
                     │
        ┌────────────▼────────────┐
        │ num_steps done?        │
        └────┬──────────────┬────┘
             │              │
           NO              YES
             │              │
        Continue         ┌────────────────────┐
        next step        │ Finalization       │
                        │ • Flush CSV        │
                        │ • Print summary    │
                        │ • Log results dir  │
                        └────────────────────┘
```

---

## Output & Results

### Console Logs

**Format**: Timestamped, severity-leveled logs (DEBUG/INFO/WARNING/ERROR)

**Key Events**:
- Slice initialization
- Component readiness
- SLA violations (WARNING level)
- Meta-scheduler decisions (INFO level)
- Simulation completion summary

**Example**:
```
2026-03-30 12:34:56 INFO Slice URLLC created │ type=urllc │ init_prbs=40
...
2026-03-30 12:34:56 WARNING t=45 │ SLA VIOLATION │ URLLC │ type=latency value=6.32 threshold=5.00
2026-03-30 12:34:56 INFO t=46 │ MetaSched │ VIOLATIONS ['URLLC'] │ boost=5/slice donor_budget=60
...
```

### CSV File: `results/metrics.csv`

**13 columns**, one row per slice per timestep:

| timestep | slice_name | slice_type | allocated_prbs | arrivals | served | queue_length | dropped | throughput | latency | sla_type | sla_threshold | sla_satisfied |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | URLLC | urllc | 40 | 18 | 15 | 3 | 0 | 15.0 | 2.5 | latency | 5.0 | True |
| 0 | eMBB | embb | 40 | 35 | 28 | 7 | 0 | 28.0 | 3.2 | throughput | 30.0 | False |
| 0 | IoT | iot | 20 | 12 | 10 | 2 | 0 | 10.0 | 4.1 | relaxed | 0.0 | True |
| 1 | URLLC | urllc | 45 | 19 | 17 | 5 | 0 | 17.0 | 2.8 | latency | 5.0 | True |

**Use cases**:
- Load into Pandas for statistical analysis
- Plot latency/throughput trends over time
- Compare algorithms (different meta-scheduler implementations)
- Export to reporting tools

### Log File: `results/simulation.log`

**Full execution transcript** with DEBUG-level details:
- Initialization logs
- Per-timestep scheduling decisions
- Quality metrics
- Error traces (if any)

---

## Extensibility

The simulator is designed with **clean interfaces** for swapping core components:

### 1. Custom Meta-Scheduler (Allocation Algorithm)

**Subclass `MetaScheduler`** to implement alternative algorithms:

```python
from core.meta_scheduler import MetaScheduler

class MyMLScheduler(MetaScheduler):
    def allocate(self, slices, violations, timestep):
        # Your custom allocation logic
        # Return dict[str, int] mapping slice_name → allocated_prbs
        pass

# In main.py, replace:
# meta_scheduler = MetaScheduler()
# with:
meta_scheduler = MyMLScheduler()
```

**Examples**:
- **Active Inference**: Use free-energy minimisation to compute optimal allocation
- **Deep RL**: Train PPO/SAC agent to predict PRB adjustments from violations + queue depths
- **MPC**: Model-predictive control using linearized queue dynamics

---

### 2. Custom DU Scheduler (Intra-Slice Policy)

**Subclass `DUScheduler`** for slice-internal packet scheduling:

```python
class ProportionalFairScheduler(DUScheduler):
    def schedule_slice(self, s, timestep):
        # Implement weighted fair queueing, round-robin, etc.
        pass
```

---

### 3. Custom Traffic Model (Arrival Process)

**Implement `generate(timestep) → int` interface**:

```python
class TraceFileTraffic:
    def __init__(self, slice, trace_file, rng):
        self.slice = slice
        self.rng = rng
        self.trace = load_trace(trace_file)

    def generate(self, timestep):
        return self.trace[timestep % len(self.trace)]
```

---

### 4. Custom SLA Checker (Constraint Evaluation)

**Subclass `SLAChecker`** for predictive or composite SLAs:

```python
class PredictiveSLAChecker(SLAChecker):
    def check_slice(self, s, snap):
        # ChaosNet-based violation prediction
        # Or: composite SLA (latency AND throughput)
        pass
```

---

### 5. Monitoring & Export

**Subscribe to `PerformanceMonitor.record()` calls** for real-time dashboards or external systems:

```python
class PrometheusExporter:
    def export_metrics(self, snapshots):
        for snap in snapshots:
            # Push to Prometheus pushgateway
            pass

monitor.on_record(exporter.export_metrics)
```

---

## Research Applications

### 1. **SLA-Aware Resource Orchestration**

Validate allocation strategies for heterogeneous SLA types:
- Compare rule-based, RL-based, and MPC approaches
- Study fairness trade-offs (URLLC vs eMBB vs IoT)
- Optimize for network-wide utility or Pareto efficiency

### 2. **Traffic Burstiness & Queue Stability**

Explore impact of burst events on resource allocation:
- Vary `burst_prob` and `burst_mult` per slice
- Study queue oscillations and stability
- Tune meta-scheduler `fairness_decay` for responsiveness

### 3. **Active Inference & Free-Energy Minimisation**

Test Active Inference agents as meta-schedulers:
- Model slices as generative processes with preferred queue depths
- Minimize surprise (SLA violations as prediction error)
- Study convergence and energy landscape

### 4. **Multi-Agent RL**

Train multi-agent PPO where each slice negotiates PRB allocation:
- Slices as agents; PRBs as shared resource
- Reward: SLA satisfaction + fairness bonus
- Analyze emergent cooperation and fairness

### 5. **Predictive SLA Violation Detection**

Train neural network to forecast violations from queue state:
- Input: Historical queue depths, latencies, arrivals
- Output: SLA violation probability next step
- Use predictions in proactive meta-scheduler

### 6. **Slice Admission Control**

Study trade-offs of accepting new slices:
- Simulate arrival of new slice requests
- Evaluate admission policies (greedy, Lyapunov, RL-based)
- Measure impact on existing slice SLAs

### 7. **ChaosNet-Driven Adaptation**

Integrate ChaosNet for SLA-violation prediction:
- Operate simulator in lookahead mode (predict violations)
- Use chaotic attractors to model slice demand patterns
- Proactively rebalance PRBs before violations occur

---

## Performance & Scalability

- **Typical runtime**: 200 timesteps × 3 slices ≈ 5 seconds (wall-clock)
- **Memory footprint**: ~10 MB (100 PRBs, 3 slices, 200 timesteps)
- **Scalability**: Linear in number of slices and timesteps

---

## Contributing & Future Work

**Potential enhancements**:
- [ ] Multi-cell / inter-cell interference modeling
- [ ] User mobility patterns (handover events)
- [ ] QoE-aware SLA metrics (video streaming, VoIP)
- [ ] Energy-efficiency constraints
- [ ] Real-time parameter tuning via API
- [ ] Interactive web dashboard

---

## References & Learning Materials

See `Learnings/` directory for:
- `Actual_Emulator_Stack.txt`: O-RAN architecture overview
- `Paper_Logic.txt`: Theoretical foundations
- `Manuscript.txt`: Academic write-up
- `v1-Flow.txt`: Initial design flow

---

## License

[Specify your license here, e.g., MIT, Apache 2.0]

## Contact & Support

For questions or issues:
- Check `results/simulation.log` for error traces
- Review `config/simulation_config.py` for tuneable parameters
- Subclass core components to experiment with custom algorithms

---

## Summary

The **RAN Meta-Scheduling Simulator** is a powerful, extensible platform for exploring resource allocation strategies in 5G/6G networks. With clean component interfaces and comprehensive logging, it enables rapid prototyping of novel scheduling algorithms, SLA-aware orchestration, and Active Inference / RL-based control policies.

**Start simulating**:
```bash
cd ran_simulator
python main.py
```

Happy researching! 🚀

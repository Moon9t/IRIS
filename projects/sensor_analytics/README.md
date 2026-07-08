# Sensor Analytics — Real-Time Sensor Data Pipeline with ML

An IRIS multifile project demonstrating a comprehensive real-time sensor data
analytics pipeline with machine learning capabilities.

## What It Demonstrates

| Feature | Where |
|---|---|
| **Records & Constants** | `sensor.iris` — `SensorReading`, `NUM_SENSORS`, `READINGS_PER_SENSOR` |
| **Lists & Maps** | Throughout — sensor batches, value extraction, per-sensor grouping |
| **Math Functions** | `sensor.iris` — `sin`, `cos`, `random` for signal generation |
| **Statistics** | `stats.iris` — mean, variance, std_dev, z-scores, moving average |
| **Autodiff (tape/backward/grad)** | `model.iris` — reverse-mode AD for linear regression training |
| **`result<T,E>` & Error Handling** | `stats.iris` — safe division, edge-case handling |
| **Closures & Higher-Order Functions** | `main.iris` — lambda-based data transforms |
| **`par for` Parallel Loops** | `main.iris` — parallel sensor data generation |
| **SVG Visualization** | `viz.iris` — line charts and bar charts |
| **Tensors** | `model.iris` — tensor operations for batch processing |
| **Time Measurement** | `main.iris` — `now_ms()` pipeline timing |

## Project Structure

```
sensor_analytics/
├── README.md          # This file
├── sensor.iris        # Sensor data generation with periodic signals
├── stats.iris         # Statistical analysis (mean, std_dev, z-scores, anomalies)
├── model.iris         # ML linear model with autodiff training
├── viz.iris           # SVG visualization (line charts, bar charts)
└── main.iris          # Full pipeline orchestration
```

## How to Run

```bash
cargo run -- run projects/sensor_analytics/main.iris
```

## Pipeline Flow

1. **Data Generation** — Generates synthetic sensor readings using sinusoidal
   signals with random noise for temperature, humidity, and pressure.
2. **Statistical Analysis** — Computes per-sensor summaries (mean, std_dev,
   min, max) and detects anomalies via z-score thresholding.
3. **ML Training** — Trains a linear regression model using reverse-mode
   autodiff to predict temperature from time step.
4. **Visualization** — Produces SVG line charts of temperature data and
   bar charts of anomaly counts.
5. **Reporting** — Prints a formatted report with timing information.

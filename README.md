
# THERMO-X

## Cryo-Inspired Adaptive Thermal Management for Water-Sustainable AI Data Centers

> **Project status:** Software prototype / simulation-first validation platform  
> **Current hardware status:** No ESP32 or physical cooling hardware is connected yet  
> **Primary interface:** Streamlit + Plotly  
> **Primary control concept:** Workload-aware adaptive cooling with safety override  
> **ML concept:** Four-model ensemble using Random Forest, XGBoost, Deep MLP, and LSTM

---

## 1. Overview

THERMO-X is a climate-tech engineering project focused on adaptive thermal management for high-density AI and data-center infrastructure.

The system is inspired by principles associated with advanced thermal and cryogenic engineering, but it does **not** attempt to operate ordinary AI processors at cryogenic temperatures.

The phrase **cryo-inspired** refers to the engineering ideas used in the architecture:

- controlled heat paths
- localized thermal management
- precise sensing
- staged thermal management
- thermal isolation
- dynamic control of heat-transfer resources

The practical prototype direction is conventional electronics-temperature-range thermal management.

The current software system provides a complete simulation-first development environment.

It models:

- computational heat generation
- thermal mass
- cold-plate-style heat transfer
- closed-loop coolant flow
- radiator heat rejection
- fan and pump control
- branch valve allocation
- sensor noise
- thermal safety logic
- adaptive cooling
- experiment logging
- fixed-vs-adaptive validation
- machine-learning prediction
- four-model ML fusion
- an interactive engineering dashboard

---

## 2. Core Problem

High-density computing infrastructure generates substantial thermal loads.

That heat has to be removed continuously.

Cooling infrastructure itself consumes resources.

The THERMO-X project therefore investigates a system-level architecture in which cooling intensity can respond to actual thermal demand rather than remaining unnecessarily aggressive at every moment.

The central control principle is:

> **Cooling intensity should follow actual thermal demand while every heat source remains within its defined safe operating envelope.**

THERMO-X therefore focuses on the interaction between:

- workload
- thermal state
- coolant conditions
- cooling demand
- control response
- energy use
- water-conscious heat rejection

The project is not claiming to have invented liquid cooling, dry cooling, thermal storage, heat recovery, artificial intelligence for cooling, or cryogenic computing.

The proposed contribution is the **integration of these engineering ideas into an adaptive thermal-management architecture**.

---

## 3. Important Scientific Scope

THERMO-X is intentionally designed around realistic prototype temperatures.

The software does not model ordinary processors operating at:

- 0 K
- 4 K
- millikelvin temperatures
- liquid-helium temperatures

Cryogenic engineering is treated as an inspiration for the system architecture, not as a direct operating temperature target for the student prototype.

The physical prototype should therefore use a controlled electrical heat source rather than exposing expensive computing hardware to a first-generation liquid loop.

This makes the laboratory system:

- safer
- cheaper
- controllable
- repeatable
- easier to instrument
- easier to validate

---

## 4. Project Philosophy

The project follows a simulation-to-hardware progression.

The current software system is the engineering sandbox.

The future physical prototype is intended to use the same high-level interfaces.

The development path is:

```text
Problem Definition
        |
        v
Thermal Model
        |
        v
Virtual Plant
        |
        v
Safety Controller
        |
        v
Adaptive Controller
        |
        v
Controlled Experiments
        |
        v
Analytics
        |
        v
ML Prediction
        |
        v
Multi-Model Ensemble
        |
        v
Predictive Advisory
        |
        v
ESP32 Integration
        |
        v
Physical Prototype
```

The system is intentionally developed incrementally.

---

## 5. Current MVP

The current software MVP contains:

- two thermal zones
- two independent simulated heat loads
- simulated coolant loop
- simulated pump
- simulated fan
- independently controlled branch valves
- virtual temperature sensing
- virtual flow sensing
- virtual pressure sensing
- ambient temperature
- safety controller
- adaptive thermal-demand control
- CSV data logging
- fixed cooling baseline
- adaptive cooling baseline
- engineering metrics
- Streamlit dashboard
- predictive ML pipeline

The intended visible behaviour is:

```text
LOAD CHANGE
      |
      v
THERMAL CHANGE
      |
      v
CONTROLLER RESPONSE
      |
      v
COOLING CHANGE
      |
      v
THERMAL RESPONSE
```

---

## 6. High-Level Architecture

```text
                           THERMO-X
                               |
             +-----------------+-----------------+
             |                                   |
             v                                   v
        Workload Model                     Environment
             |                        Ambient Temperature
             |                                   |
             +-----------------+-----------------+
                               |
                               v
                      Thermal Sensor State
                               |
                  +------------+------------+
                  |                         |
                  v                         v
             Safety Layer              ML Prediction
                  |                         |
                  |                         v
                  |                  Future Thermal State
                  |                         |
                  +------------+------------+
                               |
                               v
                     Adaptive Controller
                               |
                  +------------+------------+
                  |            |            |
                  v            v            v
                Pump         Fan         Valves
                  |            |            |
                  +------------+------------+
                               |
                               v
                         Thermal Plant
                               |
                               +---- feedback ----+
```

---

## 7. Thermal Path

The conceptual thermal path is:

```text
HEAT SOURCE
     |
     v
THERMAL INTERFACE
     |
     v
COLD PLATE
     |
     v
CLOSED COOLANT LOOP
     |
     v
HEAT EXCHANGER / RADIATOR
     |
     v
AMBIENT AIR
     |
     v
COOLANT RETURN
```

The physical prototype should emphasize non-evaporative or low-water heat rejection.

The software simulator models the same functional sequence.

---

## 8. Control Path

The control path is:

```text
WORKLOAD
+
HEATER POWER
+
TEMPERATURE
+
FLOW
+
PRESSURE
+
AMBIENT
+
THERMAL HISTORY
        |
        v
THERMAL INTELLIGENCE
        |
        v
SAFETY CHECK
        |
        v
ADAPTIVE CONTROL
        |
        v
PUMP / FAN / VALVE COMMANDS
```

The safety layer is always authoritative.

Optimization does not override safety.

---

## 9. Safety Hierarchy

The project uses the following conceptual hierarchy:

```text
1. SAFETY
2. RELIABILITY
3. RESOURCE OPTIMIZATION
```

A safety condition can override an optimization command.

Examples include:

- excessive temperature
- loss of coolant flow
- excessive pressure
- leak detection
- invalid sensor data

The current safety limits are simulation parameters.

They are not hardware ratings.

Physical limits must later be derived from the selected component datasheets and laboratory testing.

---

## 10. Thermal Model

The current model is a simplified lumped thermal model.

The core energy balance is:

\[
C_{th}\frac{dT}{dt}=P_{heat}-Q_{cooling}
\]

where:

- \(C_{th}\) is effective thermal capacitance
- \(T\) is thermal-source temperature
- \(P_{heat}\) is the simulated heat input
- \(Q_{cooling}\) is estimated heat removal

The coolant heat-transfer calculation follows:

\[
\dot Q=\dot m C_p \Delta T
\]

where:

- \(\dot Q\) is heat-transfer rate
- \(\dot m\) is coolant mass flow rate
- \(C_p\) is coolant specific heat
- \(\Delta T\) is coolant temperature rise

The equations are deliberately simplified so that parameters can later be calibrated against a physical prototype.

---

## 11. Thermal Model Scope

The current software model is not CFD.

It does not claim to reproduce:

- detailed fluid turbulence
- 3D temperature fields
- microscopic cold-plate geometry
- pump-specific pressure curves
- exact radiator performance
- real component manufacturing tolerances

Instead, it provides a transparent system-level model that is:

- deterministic
- tunable
- testable
- computationally inexpensive
- suitable for repeated experiments
- suitable for controller development

This is appropriate for the current prototype-development stage.

---

## 12. Virtual Thermal Zones

THERMO-X currently uses:

```text
SERVER_A
SERVER_B
```

Each zone has its own:

- workload
- heat input
- temperature
- coolant branch flow
- valve allocation

## Project Quick Start

1. Use Windows PowerShell from the project root.
2. Activate the project virtual environment.
3. Verify Python is 3.11.
4. Run the test suite.
5. Run the syntax compilation check.
6. Run the baseline validation experiment.
7. Generate the multi-scenario ML dataset.
8. Train the four-model ensemble.
9. Launch the dashboard.
10. Use the Command Centre for live simulation.
11. Use the Validation Lab for fixed-versus-adaptive testing.
12. Use Predictive Control for interactive AI forecasting.
13. Use System Architecture to explain the system to reviewers.

## Core Files and Responsibilities

1. `core/models.py` — shared telemetry and actuator data models.
2. `core/interfaces.py` — hardware-independent protocols.
3. `core/thermal_calculations.py` — engineering calculations.
4. `core/safety.py` — hard safety logic.
5. `core/controller.py` — adaptive thermal-demand controller.
6. `thermal_model/parameters.py` — simulation parameters.
7. `thermal_model/model.py` — lumped thermal plant.
8. `thermal_model/calibration.py` — calibration utilities.
9. `simulator/virtual_sensors.py` — simulated measurements.
10. `simulator/virtual_actuators.py` — simulated actuator state.
11. `simulator/workload_generator.py` — controlled workload schedules.
12. `simulator/thermal_system.py` — simulation integration.
13. `adapters/simulator_adapter.py` — simulation interface.
14. `adapters/serial_adapter.py` — future USB/serial interface.
15. `adapters/mqtt_adapter.py` — future MQTT interface.
16. `analytics/metrics.py` — engineering metrics.
17. `analytics/comparison.py` — fixed/adaptive comparison.
18. `analytics/plots.py` — experiment plots.
19. `experiments/experiment_runner.py` — controlled validation.
20. `experiments/scenario_generator.py` — diverse thermal scenarios.
21. `dashboard/dashboard.py` — Streamlit application entry point.
22. `dashboard/components.py` — reusable dashboard UI.
23. `ml/features.py` — physical and temporal features.
24. `ml/neural_models.py` — MLP and LSTM definitions.
25. `ml/ensemble.py` — four-model fusion and evaluation.
26. `ml/train_ensemble.py` — ensemble training CLI.
27. `ml/predict_ensemble.py` — ensemble inference.
28. `ml/run_ai_pipeline.py` — one-command AI workflow.
29. `tests/` — automated regression tests.
30. `run_dashboard.py` — automatic Streamlit/Chrome launcher.

## Experiment Commands

1. Activate: `.\.venv\Scripts\Activate.ps1`
2. Tests: `pytest -q`
3. Compile: `python -m compileall core thermal_model simulator adapters analytics experiments app dashboard ml tests`
4. Basic simulator: `python -m simulator.simulator_runner`
5. Validation: `python -m experiments.experiment_runner`
6. Scenario dataset: `python -m experiments.scenario_generator --scenarios 80 --duration 240`
7. Ensemble training: `python -m ml.train_ensemble --input data/processed/thermo_x_training_dataset.csv --horizon 30`
8. Full AI pipeline: `python -m ml.run_ai_pipeline --scenarios 80 --duration 240 --horizon 30`
9. Direct dashboard: `python -m streamlit run dashboard/dashboard.py`
10. Preferred dashboard launcher: `python run_dashboard.py`

## ML Interpretation Guide

1. MAE reports average absolute prediction error in degrees Celsius.
2. RMSE emphasizes larger prediction errors.
3. R² compares model performance with a constant-target reference.
4. Use MAE, RMSE and R² together rather than relying on one metric.
5. A persistence baseline is included to test whether ML adds value beyond current-state persistence.
6. Training, validation and testing are separated by scenario ID.
7. The final test scenarios are not used to optimize ensemble weights.
8. The ensemble combines Random Forest, XGBoost, Deep MLP and LSTM.
9. The current target is future temperature change.
10. Future temperature is reconstructed as current temperature plus predicted change.
11. The LSTM receives a short thermal history.
12. The MLP receives standardized tabular features.
13. Tree models operate on engineered thermal telemetry.
14. Model weights are learned from validation behaviour.
15. The ensemble is not manually assigned a preferred model.
16. Model disagreement is retained as a diagnostic signal.
17. AI output is advisory in the software prototype.
18. AI output must not bypass physical safety in a future hardware system.

## Dashboard Demonstration

1. Open Command Centre.
2. Confirm two thermal zones are visible.
3. Run Adaptive Simulation.
4. Observe workload changes.
5. Observe temperature response.
6. Observe pump command response.
7. Observe fan command response.
8. Observe branch valve allocation.
9. Inspect coolant inlet and outlet temperatures.
10. Inspect flow and pressure.
11. Open Validation Lab.
12. Run Fixed vs Adaptive Validation.
13. Inspect the comparison curve.
14. Inspect engineering metrics.
15. Open Predictive Control.
16. Select a Spike scenario.
17. Increase ambient temperature.
18. Increase peak workload.
19. Increase zone imbalance.
20. Run AI Forecast.
21. Inspect thermal outlook.
22. Inspect AI action.
23. Inspect model consensus.
24. Expand Four-model fusion details.
25. Expand ML validation diagnostics.
26. Open System Architecture.

## Physical Prototype Roadmap

1. Start with an electrical heater as the controlled heat source.
2. Attach the heater to a safe thermal block.
3. Interface the thermal block with the selected cold plate.
4. Build a closed coolant loop.
5. Add temperature measurement at important points.
6. Add coolant flow measurement.
7. Add pressure measurement.
8. Add leak detection.
9. Add pump control.
10. Add fan control.
11. Add branch valve control.
12. Add local hardware protection.
13. Connect the ESP32.
14. Integrate the serial or MQTT adapter.
15. Calibrate the thermal model.
16. Compare physical and simulated response.
17. Collect physical datasets.
18. Re-train or recalibrate the ML system.
19. Validate predictive control only after safety validation.

## Evidence Rules

1. A simulation result is not a physical measurement.
2. A simulation parameter is not a component datasheet value.
3. A manufacturer specification is external evidence.
4. A model prediction is not a measurement.
5. A dashboard screenshot is not proof of energy savings.
6. Water savings must be measured or explicitly modeled.
7. Cooling-energy savings must be measured or explicitly modeled.
8. Industrial-scale claims require scale-up analysis.
9. Physical claims require physical experiments.
10. Every experiment should retain its configuration.
11. Every ML result should identify its dataset and split.
12. Every reported metric should include units.
13. Every safety limit used in hardware should be justified.

## Git Workflow

1. Check repository state with `git status`.
2. Review staged changes before committing.
3. Do not commit `.venv`.
4. Do not commit secrets or access tokens.
5. Do not commit local caches.
6. Do not commit large generated datasets unless intentionally published.
7. Do not commit generated model artifacts unless intentionally versioned.
8. Use meaningful commit messages.
9. Push the `main` branch after the first commit.
10. Use normal fast-forward pushes.
11. Avoid force-push unless intentionally rewriting history.

### Testing Checklist — Revision 1

- Run `pytest -q`.
- Confirm all tests pass.
- Run the compile check.
- Verify imports.
- Verify thermal values remain finite.
- Verify safety states trigger correctly.
- Verify actuator values stay within range.
- Verify scenario splits do not overlap.
- Verify ensemble weights sum to one.
- Verify generated predictions contain expected fields.

### Testing Checklist — Revision 2

- Run `pytest -q`.
- Confirm all tests pass.
- Run the compile check.
- Verify imports.
- Verify thermal values remain finite.
- Verify safety states trigger correctly.
- Verify actuator values stay within range.
- Verify scenario splits do not overlap.
- Verify ensemble weights sum to one.
- Verify generated predictions contain expected fields.

### Testing Checklist — Revision 3

- Run `pytest -q`.
- Confirm all tests pass.
- Run the compile check.
- Verify imports.
- Verify thermal values remain finite.
- Verify safety states trigger correctly.
- Verify actuator values stay within range.
- Verify scenario splits do not overlap.
- Verify ensemble weights sum to one.
- Verify generated predictions contain expected fields.

### Testing Checklist — Revision 4

- Run `pytest -q`.
- Confirm all tests pass.
- Run the compile check.
- Verify imports.
- Verify thermal values remain finite.
- Verify safety states trigger correctly.
- Verify actuator values stay within range.
- Verify scenario splits do not overlap.
- Verify ensemble weights sum to one.
- Verify generated predictions contain expected fields.

### Testing Checklist — Revision 5

- Run `pytest -q`.
- Confirm all tests pass.
- Run the compile check.
- Verify imports.
- Verify thermal values remain finite.
- Verify safety states trigger correctly.
- Verify actuator values stay within range.
- Verify scenario splits do not overlap.
- Verify ensemble weights sum to one.
- Verify generated predictions contain expected fields.

### Final Demonstration Checklist — Revision 1

- Regenerate validation data.
- Regenerate AI dataset.
- Train the ensemble.
- Inspect the test metrics.
- Confirm the ensemble model exists.
- Launch the dashboard.
- Confirm Chrome opens.
- Confirm the header renders correctly.
- Confirm no raw HTML is visible.
- Run Command Centre simulation.
- Run Validation Lab.
- Run Predictive Control scenario.
- Inspect model weights.
- Inspect System Architecture.

### Final Demonstration Checklist — Revision 2

- Regenerate validation data.
- Regenerate AI dataset.
- Train the ensemble.
- Inspect the test metrics.
- Confirm the ensemble model exists.
- Launch the dashboard.
- Confirm Chrome opens.
- Confirm the header renders correctly.
- Confirm no raw HTML is visible.
- Run Command Centre simulation.
- Run Validation Lab.
- Run Predictive Control scenario.
- Inspect model weights.
- Inspect System Architecture.

### Final Demonstration Checklist — Revision 3

- Regenerate validation data.
- Regenerate AI dataset.
- Train the ensemble.
- Inspect the test metrics.
- Confirm the ensemble model exists.
- Launch the dashboard.
- Confirm Chrome opens.
- Confirm the header renders correctly.
- Confirm no raw HTML is visible.
- Run Command Centre simulation.
- Run Validation Lab.
- Run Predictive Control scenario.
- Inspect model weights.
- Inspect System Architecture.

### Final Demonstration Checklist — Revision 4

- Regenerate validation data.
- Regenerate AI dataset.
- Train the ensemble.
- Inspect the test metrics.
- Confirm the ensemble model exists.
- Launch the dashboard.
- Confirm Chrome opens.
- Confirm the header renders correctly.
- Confirm no raw HTML is visible.
- Run Command Centre simulation.
- Run Validation Lab.
- Run Predictive Control scenario.
- Inspect model weights.
- Inspect System Architecture.

### Final Demonstration Checklist — Revision 5

- Regenerate validation data.
- Regenerate AI dataset.
- Train the ensemble.
- Inspect the test metrics.
- Confirm the ensemble model exists.
- Launch the dashboard.
- Confirm Chrome opens.
- Confirm the header renders correctly.
- Confirm no raw HTML is visible.
- Run Command Centre simulation.
- Run Validation Lab.
- Run Predictive Control scenario.
- Inspect model weights.
- Inspect System Architecture.

### Physical Validation Checklist — Revision 1

- Verify all hoses and fittings.
- Verify electrical isolation.
- Verify fusing/protection.
- Verify leak containment.
- Verify sensor wiring.
- Verify pump driver.
- Verify fan driver.
- Verify valve driver.
- Verify ESP32 local safety.
- Verify emergency shutdown.
- Verify sensor calibration.
- Record baseline ambient.
- Record heater power.
- Record coolant temperatures.
- Record flow.
- Record pressure.
- Record electrical power.
- Repeat each important condition.
- Archive raw physical data.

### Physical Validation Checklist — Revision 2

- Verify all hoses and fittings.
- Verify electrical isolation.
- Verify fusing/protection.
- Verify leak containment.
- Verify sensor wiring.
- Verify pump driver.
- Verify fan driver.
- Verify valve driver.
- Verify ESP32 local safety.
- Verify emergency shutdown.
- Verify sensor calibration.
- Record baseline ambient.
- Record heater power.
- Record coolant temperatures.
- Record flow.
- Record pressure.
- Record electrical power.
- Repeat each important condition.
- Archive raw physical data.

### Physical Validation Checklist — Revision 3

- Verify all hoses and fittings.
- Verify electrical isolation.
- Verify fusing/protection.
- Verify leak containment.
- Verify sensor wiring.
- Verify pump driver.
- Verify fan driver.
- Verify valve driver.
- Verify ESP32 local safety.
- Verify emergency shutdown.
- Verify sensor calibration.
- Record baseline ambient.
- Record heater power.
- Record coolant temperatures.
- Record flow.
- Record pressure.
- Record electrical power.
- Repeat each important condition.
- Archive raw physical data.

### Physical Validation Checklist — Revision 4

- Verify all hoses and fittings.
- Verify electrical isolation.
- Verify fusing/protection.
- Verify leak containment.
- Verify sensor wiring.
- Verify pump driver.
- Verify fan driver.
- Verify valve driver.
- Verify ESP32 local safety.
- Verify emergency shutdown.
- Verify sensor calibration.
- Record baseline ambient.
- Record heater power.
- Record coolant temperatures.
- Record flow.
- Record pressure.
- Record electrical power.
- Repeat each important condition.
- Archive raw physical data.

### Physical Validation Checklist — Revision 5

- Verify all hoses and fittings.
- Verify electrical isolation.
- Verify fusing/protection.
- Verify leak containment.
- Verify sensor wiring.
- Verify pump driver.
- Verify fan driver.
- Verify valve driver.
- Verify ESP32 local safety.
- Verify emergency shutdown.
- Verify sensor calibration.
- Record baseline ambient.
- Record heater power.
- Record coolant temperatures.
- Record flow.
- Record pressure.
- Record electrical power.
- Repeat each important condition.
- Archive raw physical data.

## Glossary

**Adaptive cooling:** Cooling command varies with measured or simulated thermal demand.
**Thermal zone:** An independently monitored and controlled heat-producing region.
**TDI:** Thermal Demand Index used by the adaptive controller.
**ΔT:** Temperature difference, such as coolant outlet minus inlet temperature.
**Thermal mass:** Effective heat capacity represented in the lumped thermal model.
**UA:** Effective heat-transfer conductance used by the simplified model.
**Heat removed:** Estimated thermal power transferred from the heat source to the coolant.
**Cooling energy:** Integrated pump/fan electrical-energy estimate in the current software model.
**Persistence baseline:** Prediction that assumes future temperature remains at the current temperature.
**MAE:** Mean absolute error.
**RMSE:** Root mean squared error.
**R²:** Coefficient of determination.
**Ensemble:** Combination of multiple model predictions.
**LSTM:** Long Short-Term Memory recurrent neural-network architecture.
**MLP:** Multi-layer perceptron feed-forward neural network.
**Scenario:** One independent synthetic thermal experiment.
**Hardware-in-the-loop:** Real controller interacting with a simulated plant.
**Physical prototype:** Instrumented laboratory implementation of the thermal path.

## Common Mistakes to Avoid

- Do not describe simulated temperature as measured temperature.
- Do not claim industrial-scale water savings from the current software model.
- Do not say the AI is physically controlling a pump before hardware integration exists.
- Do not train and test on overlapping scenario IDs.
- Do not select a model before examining held-out results.
- Do not bypass the safety layer with ML output.
- Do not treat model disagreement as a formal confidence interval.
- Do not change experiment conditions halfway through a comparison without documenting it.
- Do not delete failing tests to make the test suite pass.
- Do not use force-push casually.
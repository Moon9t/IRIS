# IRIS Examples

This directory contains examples showcasing the IRIS language capabilities, ranging from basic syntax to advanced systems and machine learning implementations.

## Directory Structure

The examples are categorized progressively:

### `01_basics/`
Core language features and syntax.
- **Variables & Constants:** `constants.iris`
- **Strings:** `strings.iris`, `string_processing.iris`, `fstrings.iris`
- **Control Flow:** `control_flow.iris`, `loops.iris`, `block_expressions.iris`
- **Data Collections:** `arrays.iris`, `test_list.iris`
- **Other:** `hello.iris`, `fizzbuzz.iris`, `error_handling.iris`

### `02_functions/`
Functional programming paradigms.
- **First-class functions:** `closures.iris`, `closures_functional.iris`, `lambdas.iris`
- **Recursion:** `factorial.iris`, `fibonacci.iris`, `recursion.iris`

### `03_data_structures_algorithms/`
Common computer science primitives.
- `data_structures.iris`, `maps.iris`
- `algorithms.iris`, `sorting.iris`, `math_utils.iris`

### `04_types_and_traits/`
The IRIS static type system.
- **Algebraic Data Types:** `enums.iris`, `structs.iris`, `tuples.iris`
- **Interfaces & Patterns:** `traits.iris`, `generics.iris`, `pattern_matching.iris`, `options_results.iris`
- **System Overview:** `type_system.iris`

### `05_systems/`
Low-level systems programming and FFI.
- **Concurrency & Networking:** `concurrency.iris`, `networking.iris`, `http_server_smoke.iris`
- **FFI & State:** `ffi.iris`, `backend.iris`
- **Persistence:** `database.iris`, `sql_params.iris`

### `06_machine_learning/`
Production-grade ML integrations (ONNX, LibTorch, TensorFlow) and Automatic Differentiation.
- **Tensors & Math:** `matrix_ops.iris`, `ml_tensor_operations.iris`, `statistics.iris`
- **Neural Networks:** `neural_net.iris`, `ml_neural_computation.iris`, `autodiff.iris`
- **Pipelines:** `ml_data_preprocessing.iris`, `ml_full_pipeline.iris`, `ml_model_loading.iris`, `ml_showcase.iris`

### `07_applications/`
End-to-end applications built in IRIS.
- `ais_agent.iris`, `calculator.iris`, `comprehensive.iris`, `game_of_life.iris`, `runtime_soak.iris`

---

## Running Examples

```bash
# Run a specific example
iris 01_basics/hello.iris

# Run a machine learning showcase
iris 06_machine_learning/ml_showcase.iris
```

See the `docs/` folder for more detailed documentation regarding compiler usage and ML bridge setup.

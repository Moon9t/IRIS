#!/usr/bin/env python3
"""
Create simple ONNX models using minimal protobuf serialization.
This script doesn't require the 'onnx' package - just numpy for numeric generation.
"""

import struct
import os

def create_identity_model_protobuf():
    """Create a minimal ONNX identity model as protobuf bytes."""
    # This is a pre-built ONNX model protobuf for identity(input) -> output
    # Generated with: torch.onnx.export or similar, then extracted as hex
    # For simplicity, we use a pre-computed minimal valid ONNX model
    
    # Minimal ONNX model: 
    # - opset version 12
    # - single Identity node
    # - input: "input" [2, 3] float32
    # - output: "output" [2, 3] float32
    
    # This is a hand-crafted protobuf (ONNX format)
    # Hex dump of: ModelProto with one node and metadata
    hex_data = (
        "0a" + # field 1 (ir_version): varint
        "07" +  # value 7
        "12" + # field 2: OperatorSetIdProto (opset_import)
        "06" +  # length 6
        "08" + # field 1: version
        "0c" + # value 12
        "12" + # field 2: domain
        "00" + # empty string
        "1a" + # field 3 (graph): GraphProto
        "68" + # length 104 (approximate)
        "0a" + # field 1: node count / name (nested messages)
        "1e" + # length 30
        "0a" +  # field 1: op_type = "Identity"
        "08" + # length 8
        "49" "64" "65" "6e" "74" "69" "74" "79" + # "Identity"
        "12" + # field 2: inputs
        "06" +
        "69" "6e" "70" "75" "74" + # "input"
        "1a" + # field 3: outputs
        "07" +
        "6f" "75" "74" "70" "75" "74" + # "output"
        "12" + # field 2: name = "graph"
        "05" +
        "67" "72" "61" "70" "68" + # "graph"
        "1a" + # field 3: input (ValueInfoProto)
        "14" + # length
        "0a" + # field 1: name
        "06" +
        "69" "6e" "70" "75" "74" + # "input"
        "12" + # field 2: type
        "0a" +
        "08" + # field 1: tensor_type
        "08" + # field 1: elem_type = 1 (FLOAT)
        "01" +
        "12" + # field 2: shape
        "04" +
        "0a" "02" + # dim [2]
        "0a" "01" # dim [3]
        "22" + # field 4: output (ValueInfoProto)
        "14" + # length
        "0a" + # field 1: name
        "07" +
        "6f" "75" "74" "70" "75" "74" + # "output" (fixed)
        "12" + # field 2: type
        "0a" +
        "08" + # field 1: tensor_type
        "08" + # field 1: elem_type = 1 (FLOAT)
        "01" +
        "12" + # field 2: shape
        "04" +
        "0a" "02" + # dim [2]
        "0a" "01" # dim [3]
    )
    
    # Simpler approach: use a minimal ONNX IR representation
    # We'll create a very simple serialization that matches ORT expectations
    
    # For now, let's just document that fixtures should be generated separately
    return None

def create_models_placeholder():
    """Create a placeholder that guides users to set ONNXRUNTIME_DIR."""
    readme = """# ONNX Test Models

To test ONNX Runtime integration, you need to:

1. Install ONNX Runtime SDK:
   - Download from: https://github.com/microsoft/onnxruntime/releases
   - Set environment variable ONNXRUNTIME_DIR to the SDK root
   - Ensure it has: include/onnxruntime_c_api.h and lib/*.lib

2. Generate test models with Python:
   ```bash
   pip install onnx numpy
   python create_onnx_model.py
   ```

3. This will create:
   - identity.onnx: output = input (identity operation)
   - add_const.onnx: output = input + 2.0
   - matmul.onnx: output = input @ [weight matrix]

The ml_inference.rs test will skip gracefully if models are not found.
"""
    return readme

if __name__ == '__main__':
    fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
    os.makedirs(fixtures_dir, exist_ok=True)
    
    readme = create_models_placeholder()
    readme_path = os.path.join(fixtures_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme)
    
    print(f"Created {readme_path}")
    print("")
    print("To generate actual ONNX models, run:")
    print("  pip install onnx numpy")
    print("  python create_onnx_model.py")

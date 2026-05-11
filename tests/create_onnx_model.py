#!/usr/bin/env python3
"""
Create simple ONNX models for testing ML integration.
Requires: onnx, numpy
"""

import numpy as np
import onnx
from onnx import helper, TensorProto

def create_identity_model():
    """Create a simple identity model: output = input"""
    # Input
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    # Output
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    
    # Identity node
    node = helper.make_node('Identity', inputs=['input'], outputs=['output'])
    
    # Create graph
    graph = helper.make_graph([node], 'identity_graph', [X], [Y])
    
    # Create model
    model = helper.make_model(graph, producer_name='iris-test')
    model.opset_import[0].version = 12
    
    return model

def create_matmul_model():
    """Create a simple matrix multiplication model: output = input @ weight"""
    # Input: [batch=2, features=3]
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    # Weight: [features=3, output=4]
    W_data = np.arange(12, dtype=np.float32).reshape(3, 4)
    W = helper.make_tensor('weight', TensorProto.FLOAT, [3, 4], W_data.tobytes(), raw=True)
    # Output: [batch=2, output=4]
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 4])
    
    # MatMul node
    node = helper.make_node('MatMul', inputs=['input', 'weight'], outputs=['output'])
    
    # Create graph
    graph = helper.make_graph([node], 'matmul_graph', [X], [Y], [W])
    
    # Create model
    model = helper.make_model(graph, producer_name='iris-test')
    model.opset_import[0].version = 12
    
    return model

def create_add_const_model():
    """Create a simple addition model: output = input + 2.0"""
    # Input: [2, 3]
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])
    # Const: [2, 3] filled with 2.0
    const_data = np.full((2, 3), 2.0, dtype=np.float32)
    const = helper.make_tensor('const_2', TensorProto.FLOAT, [2, 3], const_data.tobytes(), raw=True)
    # Output: [2, 3]
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    
    # Add node
    node = helper.make_node('Add', inputs=['input', 'const_2'], outputs=['output'])
    
    # Create graph
    graph = helper.make_graph([node], 'add_graph', [X], [Y], [const])
    
    # Create model
    model = helper.make_model(graph, producer_name='iris-test')
    model.opset_import[0].version = 12
    
    return model

if __name__ == '__main__':
    import os
    
    # Create fixtures directory
    fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
    os.makedirs(fixtures_dir, exist_ok=True)
    
    # Save models
    identity = create_identity_model()
    onnx.save(identity, os.path.join(fixtures_dir, 'identity.onnx'))
    print("Created fixtures/identity.onnx")
    
    matmul = create_matmul_model()
    onnx.save(matmul, os.path.join(fixtures_dir, 'matmul.onnx'))
    print("Created fixtures/matmul.onnx")
    
    add_const = create_add_const_model()
    onnx.save(add_const, os.path.join(fixtures_dir, 'add_const.onnx'))
    print("Created fixtures/add_const.onnx")

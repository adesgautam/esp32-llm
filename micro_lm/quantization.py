import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# Post-Training Quantization (PTQ)
# ---------------------------------------------------------

def quantize_tensor_int8(tensor):
    """Simple symmetric INT8 Post-Training Quantization"""
    max_val = tensor.abs().max()
    if max_val == 0:
        return tensor
    scale = 127.0 / max_val
    q_tensor = torch.round(tensor * scale)
    return q_tensor / scale # Dequantize immediately for simulated evaluation

def quantize_tensor_int4(tensor):
    """Simple symmetric INT4 Post-Training Quantization"""
    max_val = tensor.abs().max()
    if max_val == 0:
        return tensor
    scale = 7.0 / max_val
    q_tensor = torch.round(tensor * scale)
    return q_tensor / scale

def quantize_tensor_ternary(tensor):
    """Ternary (-1, 0, 1) Post-Training Quantization"""
    max_val = tensor.abs().max()
    if max_val == 0:
        return tensor
    # Threshold heuristic (e.g. 0.7 of mean absolute value)
    threshold = 0.7 * tensor.abs().mean()
    
    q_tensor = torch.zeros_like(tensor)
    q_tensor[tensor > threshold] = 1.0
    q_tensor[tensor < -threshold] = -1.0
    
    # Scaling factor alpha
    alpha = tensor.abs().mean()
    return q_tensor * alpha

def apply_ptq(model, bits="int8"):
    """Applies PTQ to all Linear layers in the model in-place (simulated)"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                if bits == "int8":
                    module.weight.data = quantize_tensor_int8(module.weight.data)
                elif bits == "int4":
                    module.weight.data = quantize_tensor_int4(module.weight.data)
                elif bits == "ternary":
                    module.weight.data = quantize_tensor_ternary(module.weight.data)
    return model

# ---------------------------------------------------------
# Quantization-Aware Training (QAT) Modules
# ---------------------------------------------------------

class TernaryWeightFunction(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for Ternary Weights.
    Simulates BitNet 1.58b weight quantization during the forward pass,
    but lets gradients pass through unmodified during the backward pass.
    """
    @staticmethod
    def forward(ctx, weight):
        # 1. Calculate scaling factor (mean absolute value)
        scale = weight.abs().mean().clamp(min=1e-8)
        
        # 2. Normalize and round to {-1, 0, 1}
        weight_norm = weight / scale
        weight_q = torch.round(weight_norm).clamp(-1, 1)
        
        # 3. Scale back (dequantize) for the forward pass math
        return weight_q * scale

    @staticmethod
    def backward(ctx, grad_output):
        # STE: Gradients pass straight through the non-differentiable rounding step
        return grad_output

class TernaryLinear(nn.Linear):
    """A Linear layer that quantizes its weights to Ternary (-1, 0, 1) during training."""
    def forward(self, input):
        # Quantize the weights on the fly
        quantized_weight = TernaryWeightFunction.apply(self.weight)
        return F.linear(input, quantized_weight, self.bias)

def replace_linear_with_ternary(model):
    """
    Recursively replaces all nn.Linear layers in the model with TernaryLinear layers.
    This prepares the model for Quantization-Aware Training (QAT).
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name != "lm_head":
            # Create a new TernaryLinear layer with the same params
            ternary_layer = TernaryLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None
            )
            # Copy weights and biases if they exist
            ternary_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                ternary_layer.bias.data.copy_(module.bias.data)
            
            # Replace the module
            setattr(model, name, ternary_layer)
        else:
            # Recurse into child modules
            replace_linear_with_ternary(module)
    return model

import mlx.core as mx
import mlx.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        # Simulating the structure in lora.py
        self.weight = mx.random.normal((output_dims, input_dims))
        self.lora_a = mx.random.normal((4, input_dims))
        self.lora_b = mx.zeros((output_dims, 4))
        
    def trainable_parameters(self):
        # Allow default behavior to be checked first
        return super().trainable_parameters()

class SelfFreezingLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.zeros((10, 10))
        self.lora_a = mx.ones((2, 10))
        # Freeze 'weight' immediately
        self.freeze(keys=['weight'])

def test_internal_freeze():
    print("--- Test Internal Freeze ---")
    lora = SelfFreezingLoRA()
    print(f"Self-frozen trainable: {list(lora.trainable_parameters().keys())}")
    
    # Test overwriting the parameter
    lora.weight = mx.ones((10, 10))
    # Does it stay frozen?
    print(f"After re-assigning weight: {list(lora.trainable_parameters().keys())}")

if __name__ == "__main__":
    test_internal_freeze()

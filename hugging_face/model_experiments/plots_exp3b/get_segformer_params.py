import torch
from transformers import SegformerForSemanticSegmentation
import json

def count_parameters(model_name):
    """Count the number of parameters in a SegFormer model."""
    try:
        print(f"Loading {model_name}...")
        model = SegformerForSemanticSegmentation.from_pretrained(model_name, use_safetensors=True)
        
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model: {model_name}")
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        print("-" * 50)
        
        return {
            'model_name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_params_m': total_params / 1e6,
            'trainable_params_m': trainable_params / 1e6
        }
        
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

def main():
    # SegFormer model names
    model_names = [
        "nvidia/segformer-b0-finetuned-ade-512-512",
        "nvidia/segformer-b1-finetuned-ade-512-512",
        "nvidia/segformer-b2-finetuned-ade-512-512", 
        "nvidia/segformer-b3-finetuned-ade-512-512",
        "nvidia/segformer-b4-finetuned-ade-512-512",
        "nvidia/segformer-b5-finetuned-ade-640-640"
    ]
    
    results = []
    
    for model_name in model_names:
        result = count_parameters(model_name)
        if result:
            results.append(result)
    
    # Create a summary table
    print("\n" + "="*80)
    print("SEGFORMER PARAMETER COUNT SUMMARY")
    print("="*80)
    print(f"{'Model':<35} {'Total (M)':<12} {'Trainable (M)':<15}")
    print("-" * 80)
    
    for result in results:
        model_short = result['model_name'].split('/')[-1]
        print(f"{model_short:<35} {result['total_params_m']:<12.1f} {result['trainable_params_m']:<15.1f}")
    
    # Save results to JSON
    with open('segformer_parameter_counts.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to 'segformer_parameter_counts.json'")
    
    # Create a dictionary for use in the correlation script
    print("\nDictionary for correlation script:")
    print("segformer_params = {")
    for result in results:
        variant = result['model_name'].split('-')[1]  # Extract b0, b1, etc.
        print(f"    '{variant}': {result['total_params']},  # {result['total_params_m']:.1f}M parameters")
    print("}")

if __name__ == "__main__":
    main() 
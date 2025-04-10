from transformers import ChameleonForConditionalGeneration, Trainer, TrainingArguments
import torch
import os
import json
from math import ceil, sqrt
from PIL import Image
import argparse
import torch.nn.functional as F

class TaskVector():
    def __init__(self, pretrained_model=None, finetuned_model=None, vector=None, sequential=False):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        
        Args:
            sequential: If True, loads models sequentially to save memory
        """
        if vector is not None:
            self.vector = vector
        else:
            # Modified assertion to account for sequential loading
            assert pretrained_model is not None
            assert finetuned_model is not None or sequential
            with torch.no_grad():
                if sequential:
                    # Create task vector by loading each model's parameters sequentially
                    self.vector = {}
                    pretrained_state_dict = pretrained_model.state_dict()
                    for key, param in pretrained_state_dict.items():
                        if param.dtype in [torch.int64, torch.uint8]:
                            continue
                        # Store the pretrained parameter value
                        self.vector[key] = -param.clone()  # Store negative of pretrained
                else:
                    # Original implementation - both models in memory
                    pretrained_state_dict = pretrained_model.state_dict()
                    finetuned_state_dict = finetuned_model.state_dict()
                    self.vector = {}
                    for key in pretrained_state_dict:
                        if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                            continue
                        self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def update_with_finetuned(self, finetuned_model):
        """Update task vector with finetuned model parameters (second phase of sequential loading)"""
        with torch.no_grad():
            finetuned_state_dict = finetuned_model.state_dict()
            for key in list(self.vector.keys()):
                if key in finetuned_state_dict:
                    # Add finetuned params to the negative pretrained params
                    self.vector[key].add_(finetuned_state_dict[key])
                else:
                    # If key doesn't exist in finetuned model, remove it
                    del self.vector[key]
        return self

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_model, scaling_coef=1.0, inplace=False):
        """Apply a task vector to a pretrained model.
        
        Args:
            pretrained_model: Model to apply the task vector to
            scaling_coef: Scaling coefficient for the task vector
            inplace: If True, modifies the model in-place to save memory
        """
        with torch.no_grad():
            pretrained_state_dict = pretrained_model.state_dict()
            if inplace:
                # Apply changes directly to model parameters without creating a new state dict
                for key in pretrained_state_dict:
                    if key not in self.vector:
                        continue
                    param = pretrained_model.get_parameter(key)
                    param.add_(scaling_coef * self.vector[key])
            else:
                # Original implementation
                new_state_dict = {}
                for key in pretrained_state_dict:
                    if key not in self.vector:
                        print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                        continue
                    new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
                pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model
    

#add parser function
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pretrained', type=str, default="stabilityai/stable-diffusion-2", help='pretrained model')
    parser.add_argument('--model_finetuned', type=str, default="", help='finetuned model')
    parser.add_argument('--output_dir', type=str, default="/scratch/mp5847/diffusers_ckpt/output", help='output directory')
    parser.add_argument('--tv_edit_alpha', type=float, default=0.5, help='amount of edit to task vector layer')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load pretrained model first
    model_pretrained = ChameleonForConditionalGeneration.from_pretrained(args.model_pretrained, torch_dtype=torch.float16).to(device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create task vector with sequential loading (first phase - just stores negative of pretrained)
    task_vector = TaskVector(pretrained_model=model_pretrained, finetuned_model=None, sequential=True)
    
    # Load finetuned model only after creating initial task vector
    model_finetuned = ChameleonForConditionalGeneration.from_pretrained(args.model_finetuned, torch_dtype=torch.float16).to(device)
    
    # Complete task vector with finetuned model
    task_vector.update_with_finetuned(model_finetuned)
    
    # Free memory from finetuned model
    del model_finetuned
    torch.cuda.empty_cache()
    
    # Apply task vector in-place to save memory
    model_edited = task_vector.apply_to(model_pretrained, scaling_coef=-args.tv_edit_alpha, inplace=True)
    
    # Save the edited model
    model_edited.save_pretrained(args.output_dir)
    model_edited_bin_path = os.path.join(args.output_dir, "pytorch_model.bin")
    torch.save(model_edited.state_dict(), model_edited_bin_path)
    

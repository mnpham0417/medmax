import os
import json
import torch
import wandb
import deepspeed
import argparse
from accelerate.utils import DummyOptim, DummyScheduler

from transformers import ChameleonForConditionalGeneration, Trainer, TrainingArguments, get_scheduler
from peft import LoraConfig, get_peft_model
from .data import TokenizedDataset, collate_fn, create_new_tokens
from .parser import parse_args

#custom trainer that only updates certain weights in lm_head
# Custom Trainer class to update only specific embeddings
class LMHEADTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Get the actual model, handling DDP wrapping
        self.actual_model = self.model.module if hasattr(self.model, "module") else self.model
        
        # Store original embeddings that should not be updated
        # Make sure to store on the same device as the model
        self.original_lm_head = self.actual_model.lm_head.weight.data.clone().to(self.model.device)
        
        # Convert to bf16 if bf16 training is enabled
        if self.args.bf16:
            self.original_lm_head = self.original_lm_head.to(torch.bfloat16)
        
        # Create a mask for tokens that should not be updated
        self.mask_no_updates = torch.ones((len(self.original_lm_head),), dtype=torch.bool, device=self.model.device)
        self.mask_no_updates[3:8199] = False #only update image tokens

    # Update the method signature to match the parent class
    def training_step(self, model, inputs, num_items_in_batch=None):
        # Run normal training step
        loss = super().training_step(model, inputs)
        
        # After the optimizer step, restore embeddings for tokens that should not be updated
        with torch.no_grad():
            # Get the actual model, handling DDP wrapping
            actual_model = model.module if hasattr(model, "module") else model
            current_lm_head = actual_model.lm_head.weight.data
            current_lm_head[self.mask_no_updates] = self.original_lm_head[self.mask_no_updates]
 
        return loss

def main():
    args = parse_args()
    
    # Configure WandB settings if enabled
    if args.wandb:
        import wandb
        import os
        
        # Handle SSL verification issues
        if args.wandb_disable_ssl:
            os.environ['WANDB_INSECURE_DISABLE_CERT_VERIFICATION'] = 'true'
        
        # Set offline mode if requested
        if args.wandb_offline:
            os.environ['WANDB_MODE'] = 'offline'
            print("Running WandB in offline mode. Logs will be saved locally.")
        
        try:
            wandb.init(
                project="Medmax",
                entity=args.wandb_entity,
                name=args.name,
                config=vars(args)
            )
        except Exception as e:
            print(f"Failed to initialize WandB: {e}")
            print("Continuing without WandB logging...")
            args.wandb = False
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.wandb:
        if local_rank == 0:
            wandb.init(
                entity=args.wandb_entity,
                project="Medmax",
                config=args
            )
            if args.name != "":
                wandb.run.name = args.name
    else:
        os.environ['WANDB_DISABLED'] = 'true'

    os.makedirs(args.output_dir, exist_ok = True)
    
    # Check CUDA availability and set device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU support.")
    
    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChameleonForConditionalGeneration.from_pretrained(args.ckpt, device_map="auto")
    if args.bf16:
        model.to(torch.bfloat16)
    model.mode = 'train-all'
    
    # print(model)
    # assert 0
    if args.extend_vocab:
        model.resize_token_embeddings(len(tokenizer))
    
    TRAINER = Trainer
    if args.lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj","gate_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
    else: #only train transformer block
        
        #set all parameters to requires_grad
        for param in model.parameters():
            param.requires_grad = False
        if args.trainable_layers == "self_attn":
            #set layers with model.layers in name to trainable
            for name, param in model.named_parameters():
                if "model.layers" in name and "self_attn" in name:
                    param.requires_grad = True
        elif args.trainable_layers == "mlp":
            for name, param in model.named_parameters():
                if "model.layers" in name and "mlp" in name:
                    param.requires_grad = True
        elif args.trainable_layers == "lm_head":
            TRAINER = LMHEADTrainer
            for name, param in model.named_parameters():
                if "lm_head" in name:
                    param.requires_grad = True
        elif args.trainable_layers == "whole_model":
            # First, we disable gradient for all parameters
            model.requires_grad_(False)
            
            # Then enable gradient only for the inner transformer model
            model.model.requires_grad_(True)
            
            # Make sure embeddings are trainable
            model.model.embed_tokens.weight.requires_grad_(False)
            
            # Make sure vqmodel is frozen - this is critical since we don't want to change the tokenizer
            model.model.vqmodel.requires_grad_(False)
            
        #print all parameters that are trainable
        print("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
                        
    # Initialize the dataset
    train_dataset = TokenizedDataset(args.train_data)
    val_dataset = TokenizedDataset(args.val_data) if args.val_data != "" else None

   # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epoch,
        max_steps=args.steps,
        gradient_accumulation_steps=args.grad_acc,
        per_device_train_batch_size=args.bs,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=10,  # Keep last 3 checkpoints
        bf16=args.bf16,
        fp16=args.fp16,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps" if args.val_data != "" else "no",
        eval_steps=args.eval_steps,
        deepspeed=args.ds,
        report_to="wandb" if args.wandb else "none",
    )

    # Initialize the Trainer with custom collate_fn
    trainer = TRAINER(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn
    )

    # Train the model
    trainer.train(resume_from_checkpoint=args.resume)
        
if __name__ == "__main__":
    main()
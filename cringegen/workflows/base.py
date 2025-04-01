"""
Base classes for ComfyUI workflows
"""

import json
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable


class WorkflowError(Exception):
    """Exception raised for errors in a workflow."""

    pass


class NodeOutput:
    """Represents an output from a node in a workflow"""

    def __init__(self, node_id: int, output_index: int = 0):
        """Initialize a node output

        Args:
            node_id: The ID of the node
            output_index: The index of the output
        """
        self.node_id = node_id
        self.output_index = output_index

    def to_json(self) -> List[Any]:
        """Convert to the format used by ComfyUI

        Returns:
            A list in the format [node_id, output_index]
        """
        return [str(self.node_id), self.output_index]


class ComfyWorkflow:
    """Builder for workflows which can be sent to the ComfyUI prompt API"""

    def __init__(self):
        """Initialize a ComfyUI workflow"""
        self.nodes = {}
        self.node_counter = 0

    def add_node(self, class_type: str, inputs: Dict[str, Any]) -> int:
        """Add a node to the workflow

        Args:
            class_type: The type of node to add
            inputs: The inputs for the node

        Returns:
            The ID of the added node
        """
        self.node_counter += 1
        node_id = self.node_counter

        # Convert NodeOutput objects to the format expected by ComfyUI
        processed_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, NodeOutput):
                processed_inputs[key] = value.to_json()
            else:
                processed_inputs[key] = value

        self.nodes[str(node_id)] = {"class_type": class_type, "inputs": processed_inputs}

        return node_id

    def get_output(self, node_id: int, output_index: int = 0) -> NodeOutput:
        """Get an output from a node

        Args:
            node_id: The ID of the node
            output_index: The index of the output

        Returns:
            A NodeOutput object
        """
        return NodeOutput(node_id, output_index)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert the workflow to a dictionary

        Returns:
            A dictionary representation of the workflow
        """
        return self.nodes

    def save(self, filepath: Union[str, Path]) -> None:
        """Save the workflow to a file

        Args:
            filepath: Path to save the workflow
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.nodes, f, indent=2)

    def load(self, filepath: Union[str, Path]) -> None:
        """Load a workflow from a file

        Args:
            filepath: Path to the workflow file
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise WorkflowError(f"Workflow file not found: {filepath}")

        with open(filepath, "r") as f:
            self.nodes = json.load(f)

        # Set node_counter to the highest node ID
        max_id = 0
        for node_id in self.nodes.keys():
            try:
                max_id = max(max_id, int(node_id))
            except ValueError:
                pass

        self.node_counter = max_id

    # Common node builders for Stable Diffusion workflows

    def load_checkpoint(self, checkpoint: str) -> NodeOutput:
        """Add a node to load a checkpoint model

        Args:
            checkpoint: Name of the checkpoint file

        Returns:
            A NodeOutput for the loaded model
        """
        node_id = self.add_node("CheckpointLoaderSimple", {"ckpt_name": checkpoint})
        return self.get_output(node_id)

    def load_lora(
        self, model: NodeOutput, lora: str, strength_model: float = 1.0, strength_clip: float = 1.0
    ) -> Tuple[NodeOutput, NodeOutput]:
        """Add a node to load a LoRA model

        Args:
            model: The base model to apply the LoRA to
            lora: Name of the LoRA file
            strength_model: Strength of the LoRA for the model
            strength_clip: Strength of the LoRA for the CLIP

        Returns:
            A tuple of NodeOutputs (model, clip)
        """
        node_id = self.add_node(
            "LoraLoader",
            {
                "model": model,
                "lora_name": lora,
                "strength_model": strength_model,
                "strength_clip": strength_clip,
            },
        )
        return self.get_output(node_id, 0), self.get_output(node_id, 1)

    def load_vae(self, vae_name: str) -> NodeOutput:
        """Add a node to load a VAE model

        Args:
            vae_name: Name of the VAE file

        Returns:
            A NodeOutput for the loaded VAE
        """
        node_id = self.add_node("VAELoader", {"vae_name": vae_name})
        return self.get_output(node_id)

    def clip_text_encode(self, clip: NodeOutput, text: str) -> NodeOutput:
        """Add a node to encode text with CLIP

        Args:
            clip: The CLIP model to use
            text: The text to encode

        Returns:
            A NodeOutput for the encoded text
        """
        node_id = self.add_node("CLIPTextEncode", {"clip": clip, "text": text})
        return self.get_output(node_id)

    def ksampler(
        self,
        model: NodeOutput,
        positive: NodeOutput,
        negative: NodeOutput,
        latent: NodeOutput,
        seed: int = 0,
        steps: int = 40,
        cfg: float = 3.5,
        sampler_name: str = "euler",
        scheduler: str = "normal",
        denoise: float = 1.0,
    ) -> NodeOutput:
        """Add a KSampler node

        Args:
            model: The model to use
            positive: Positive conditioning
            negative: Negative conditioning
            latent: Latent image
            seed: Random seed
            steps: Number of steps
            cfg: CFG scale
            sampler_name: Name of the sampler to use
            scheduler: Name of the scheduler to use
            denoise: Denoising strength

        Returns:
            A NodeOutput for the sampled latent
        """
        node_id = self.add_node(
            "KSampler",
            {
                "model": model,
                "positive": positive,
                "negative": negative,
                "latent_image": latent,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
            },
        )
        return self.get_output(node_id)

    def empty_latent(
        self, width: int = 1024, height: int = 1024, batch_size: int = 1
    ) -> NodeOutput:
        """Add a node to create an empty latent image

        Args:
            width: Width of the latent image
            height: Height of the latent image
            batch_size: Batch size

        Returns:
            A NodeOutput for the empty latent
        """
        node_id = self.add_node(
            "EmptyLatentImage", {"width": width, "height": height, "batch_size": batch_size}
        )
        return self.get_output(node_id)

    def vae_decode(self, vae: NodeOutput, latent: NodeOutput) -> NodeOutput:
        """Add a node to decode a latent image

        Args:
            vae: The VAE to use
            latent: The latent image to decode

        Returns:
            A NodeOutput for the decoded image
        """
        node_id = self.add_node("VAEDecode", {"vae": vae, "samples": latent})
        return self.get_output(node_id)

    def save_image(self, image: NodeOutput, filename_prefix: str = "cringegen") -> NodeOutput:
        """Add a node to save an image

        Args:
            image: The image to save
            filename_prefix: Prefix for the filename

        Returns:
            A NodeOutput for the saved image
        """
        node_id = self.add_node("SaveImage", {"images": image, "filename_prefix": filename_prefix})
        return self.get_output(node_id)


def get_workflow_template(workflow_type: str) -> Optional[Callable]:
    """Get a function that creates a workflow template based on the workflow type

    Args:
        workflow_type: Type of workflow to create (e.g., "furry", "nsfw", "character")

    Returns:
        A function that creates a workflow template, or None if the workflow type is unknown
    """
    if workflow_type == "furry":
        from .furry import create_basic_furry_workflow

        def furry_workflow_wrapper(args):
            """Wrapper to convert args to proper parameters for create_basic_furry_workflow"""
            import random

            checkpoint = (
                args.checkpoint
                if hasattr(args, "checkpoint") and args.checkpoint
                else "noobaiXLVpredv10.safetensors"
            )
            lora = args.lora if hasattr(args, "lora") and args.lora else None
            lora_strength = args.lora_strength if hasattr(args, "lora_strength") else 0.35
            prompt = (
                args.prompt if hasattr(args, "prompt") and args.prompt else "a cute furry character"
            )
            negative_prompt = (
                args.negative_prompt
                if hasattr(args, "negative_prompt")
                else "worst quality, low quality"
            )

            # Generate a seed if not provided or -1
            if hasattr(args, "seed") and args.seed != -1:
                seed = args.seed
            else:
                seed = random.randint(0, 2**32 - 1)

            # Other parameters
            steps = args.steps if hasattr(args, "steps") else 20
            cfg = args.cfg if hasattr(args, "cfg") else None
            width = args.width if hasattr(args, "width") else 1024
            height = args.height if hasattr(args, "height") else 1024
            sampler = args.sampler if hasattr(args, "sampler") else None
            scheduler = args.scheduler if hasattr(args, "scheduler") else None

            # PAG options
            use_pag = args.pag if hasattr(args, "pag") else False
            pag_scale = args.pag_scale if hasattr(args, "pag_scale") else 3.0
            pag_sigma_start = args.pag_sigma_start if hasattr(args, "pag_sigma_start") else -1.0
            pag_sigma_end = args.pag_sigma_end if hasattr(args, "pag_sigma_end") else -1.0

            # DeepShrink options
            use_deepshrink = args.deepshrink if hasattr(args, "deepshrink") else False
            deepshrink_factor = (
                args.deepshrink_factor if hasattr(args, "deepshrink_factor") else 2.0
            )
            deepshrink_start = args.deepshrink_start if hasattr(args, "deepshrink_start") else 0.0
            deepshrink_end = args.deepshrink_end if hasattr(args, "deepshrink_end") else 0.35
            deepshrink_gradual = (
                args.deepshrink_gradual if hasattr(args, "deepshrink_gradual") else 0.6
            )
            
            # Split-sigma options
            split_sigmas = args.split_sigmas if hasattr(args, "split_sigmas") else None
            split_first_cfg = args.split_first_cfg if hasattr(args, "split_first_cfg") else None
            split_second_cfg = args.split_second_cfg if hasattr(args, "split_second_cfg") else None
            split_first_sampler = args.split_first_sampler if hasattr(args, "split_first_sampler") else None
            split_second_sampler = args.split_second_sampler if hasattr(args, "split_second_sampler") else None
            split_first_scheduler = args.split_first_scheduler if hasattr(args, "split_first_scheduler") else None
            split_second_scheduler = args.split_second_scheduler if hasattr(args, "split_second_scheduler") else None
            
            # Detail daemon options
            use_detail_daemon = args.detail_daemon if hasattr(args, "detail_daemon") else False
            detail_amount = args.detail_amount if hasattr(args, "detail_amount") else 0.1
            detail_start = args.detail_start if hasattr(args, "detail_start") else 0.5
            detail_end = args.detail_end if hasattr(args, "detail_end") else 0.8

            return create_basic_furry_workflow(
                checkpoint=checkpoint,
                lora=lora,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                seed=seed,
                steps=steps,
                cfg=cfg,
                lora_strength=lora_strength,
                sampler=sampler,
                scheduler=scheduler,
                use_pag=use_pag,
                pag_scale=pag_scale,
                pag_sigma_start=pag_sigma_start,
                pag_sigma_end=pag_sigma_end,
                use_deepshrink=use_deepshrink,
                deepshrink_factor=deepshrink_factor,
                deepshrink_start=deepshrink_start,
                deepshrink_end=deepshrink_end,
                deepshrink_gradual=deepshrink_gradual,
                split_sigmas=split_sigmas,
                split_first_cfg=split_first_cfg,
                split_second_cfg=split_second_cfg,
                split_first_sampler=split_first_sampler, 
                split_second_sampler=split_second_sampler,
                split_first_scheduler=split_first_scheduler,
                split_second_scheduler=split_second_scheduler,
                use_detail_daemon=use_detail_daemon,
                detail_amount=detail_amount,
                detail_start=detail_start,
                detail_end=detail_end,
            )

        return furry_workflow_wrapper
    elif workflow_type == "nsfw":
        from ..commands.nsfw import create_nsfw_workflow

        return create_nsfw_workflow
    elif workflow_type == "character":
        from ..commands.character import create_character_workflow

        return create_character_workflow
    else:
        return None

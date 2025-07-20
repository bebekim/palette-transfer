# ABOUTME: This file provides the main CLI interface for palette transfer algorithms
# ABOUTME: Imports modular components and coordinates image processing workflows for hair clinics

import os
from PIL import Image
import numpy as np
from helpers import build_argument_parser, get_image
from kmeans_palette import KMeansReducedPalette, UniqueKMeansReducedPalette
from reinhard_transfer import ReinhardColorTransfer
from targeted_transfer import TargetedReinhardTransfer
from entire_palette import EntirePalette


def main():
    args = build_argument_parser()
    method = args["method"]
    k_colors = args["color"]
    color_space = args["color_space"]
    random_walk = args.get("random_walk", False)
    walk_steps = args.get("walk_steps", 5)
    
    # Targeted transfer parameters
    skin_blend = args.get("skin_blend", 0.9)
    hair_blend = args.get("hair_blend", 0.5)
    bg_blend = args.get("bg_blend", 0.3)
    
    # Load source and target images
    src = get_image(args["source"], color_space="RGB")
    tgt = get_image(args["target"], color_space="RGB")

    # Create output directory if not specified
    output_dir = args["output"]
    if output_dir is None:
        # Create an output directory in the same folder as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # Print information
    print(f"Source image: {args['source']} ({src.shape})")
    print(f"Target image: {args['target']} ({tgt.shape})")
    print(f"Output directory: {output_dir}")
    print(f"Color transfer method: {method}")
    
    # Save original images to output directory for comparison
    src_path = os.path.join(output_dir, f"source_{os.path.basename(args['source'])}")
    tgt_path = os.path.join(output_dir, f"target_{os.path.basename(args['target'])}")
    Image.fromarray(src).save(src_path)
    Image.fromarray(tgt).save(tgt_path)
    
    # [Keep existing code for other methods...]
    
    # Apply Targeted Reinhard color transfer
    if method in ["targeted", "all"]:
        print("Applying Targeted Reinhard color transfer...")
        print(f"Skin blend: {skin_blend}, Hair blend: {hair_blend}, Background blend: {bg_blend}")
        
        targeted = TargetedReinhardTransfer(
            skin_blend_factor=skin_blend,
            hair_region_blend_factor=hair_blend,
            background_blend_factor=bg_blend
        )
        targeted.fit(src)
        tgt_targeted = targeted.recolor(tgt)
        
        # Save result
        tgt_targeted_path = os.path.join(output_dir, 
            f"targeted_skin{skin_blend}_hair{hair_blend}_bg{bg_blend}_{os.path.basename(args['target'])}")
        Image.fromarray(tgt_targeted).save(tgt_targeted_path)
        print(f"Targeted recolored image saved to {tgt_targeted_path}")
        
        # Also save visualization of the masks
        if targeted.skin_mask is not None:
            # Create visualization of skin mask
            skin_vis = (targeted.skin_mask * 255).astype(np.uint8)
            skin_mask_path = os.path.join(output_dir, f"skin_mask_{os.path.basename(args['target'])}")
            Image.fromarray(skin_vis).save(skin_mask_path)
            
            # Create visualization of hair region mask
            if targeted.hair_region_mask is not None:
                hair_vis = (targeted.hair_region_mask * 255).astype(np.uint8)
                hair_mask_path = os.path.join(output_dir, f"hair_region_mask_{os.path.basename(args['target'])}")
                Image.fromarray(hair_vis).save(hair_mask_path)
            
            print(f"Mask visualizations saved to {output_dir}")
    
    print(f"All output saved to {output_dir}")


if __name__=="__main__":
    main()

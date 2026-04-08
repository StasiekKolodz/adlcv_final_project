import os
from datasets import load_dataset

def main():
    save_path = "./local_fineweb"
    
    print("Downloading FineWeb-1B from Hugging Face... (This might take a few minutes)")
    # Notice we removed streaming=True so it actually downloads everything
    dataset = load_dataset("VisionTheta/fineweb-1B", split="train")
    
    print(f"Download complete! Saving to {save_path}...")
    dataset.save_to_disk(save_path)
    
    print("Successfully saved locally!")

if __name__ == "__main__":
    main()
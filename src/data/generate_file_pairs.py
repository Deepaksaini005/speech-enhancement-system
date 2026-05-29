import csv
from pathlib import Path

def generate_file_pairs():
    # Root of Voice-Enhancement-AI project is three levels up from this script,
    # and the dataset directories (clean, 0dB, 5dB, 10dB) are siblings of Voice-Enhancement-AI.
    project_root = Path(__file__).resolve().parents[2]
    voice_root = project_root.parent
    
    clean_dir = voice_root / "clean"
    noise_dirs = [voice_root / "0dB", voice_root / "5dB", voice_root / "10dB"]
    
    metadata_dir = project_root / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metadata_dir / "file_pairs.csv"
    
    pairs = []
    
    if not clean_dir.exists():
        # Fallback to absolute paths directly on D: drive if structure is slightly different
        clean_dir = Path("D:/Voice/clean")
        noise_dirs = [Path("D:/Voice/0dB"), Path("D:/Voice/5dB"), Path("D:/Voice/10dB")]
        
    print(f"Searching for audio files in: {clean_dir}")
    
    if clean_dir.exists():
        for clean_file in sorted(clean_dir.glob("*.wav")):
            stem = clean_file.stem  # e.g. "sp01"
            
            # Look for corresponding noisy files
            for nd in noise_dirs:
                if not nd.exists():
                    continue
                # For 0dB, it's spXX_street_sn0.wav
                # For 5dB, it's spXX_street_sn5.wav
                # For 10dB, it's spXX_street_sn10.wav
                db_suffix = nd.name.replace("dB", "")
                noisy_filename = f"{stem}_street_sn{db_suffix}.wav"
                noisy_file = nd / noisy_filename
                
                if noisy_file.exists():
                    pairs.append({
                        "clean_path": str(clean_file.resolve()),
                        "noisy_path": str(noisy_file.resolve())
                    })
                    
    print(f"Found {len(pairs)} clean-noisy pairs.")
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clean_path", "noisy_path"])
        writer.writeheader()
        writer.writerows(pairs)
        
    print(f"Successfully generated: {csv_path}")

if __name__ == "__main__":
    generate_file_pairs()

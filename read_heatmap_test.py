#!/usr/bin/env python3
"""
Simple script to read and explore the new position matrices.

The position files are located at data/position/step_X.npz and contain:
- positions: array with shape (n, 2) representing x,y coordinates
- entity_ids: list[str] with entity identifiers in same order as positions
"""

import numpy as np
import os


def read_position_data(step: int):
    """
    Read position data for a given step.

    Parameters
    ----------
    step : int
        The step number to load

    Returns
    -------
    tuple
        (positions, entity_ids) where positions is (n, 2) array and entity_ids is list[str]
    """
    filename = f"step_{step}.npz"
    filepath = os.path.join("./data/position", filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Position file '{filename}' not found in './data/position'.")

    with np.load(filepath, allow_pickle=True) as data:
        positions = data["positions"]
        entity_ids = data["entity_ids"]

    return positions, entity_ids


def explore_position_data(step: int):
    """
    Load and explore position data for a given step.
    """
    print(f"\n=== Exploring Position Data for Step {step} ===")

    try:
        positions, entity_ids = read_position_data(step)

        print(f"\nData loaded successfully!")
        print(f"Positions shape: {positions.shape}")
        print(f"Entity IDs count: {len(entity_ids)}")
        print(f"Data types: positions={positions.dtype}, entity_ids={type(entity_ids)}")

        # Show basic statistics
        print(f"\nPosition Statistics:")
        print(f"  X coordinates - min: {positions[:, 0].min():.2f}, max: {positions[:, 0].max():.2f}")
        print(f"  Y coordinates - min: {positions[:, 1].min():.2f}, max: {positions[:, 1].max():.2f}")
        print(f"  X range: {positions[:, 0].max() - positions[:, 0].min():.2f}")
        print(f"  Y range: {positions[:, 1].max() - positions[:, 1].min():.2f}")

        # Show first few entries
        print(f"\nFirst 5 entities:")
        for i in range(min(5, len(entity_ids))):
            x, y = positions[i]
            print(f"  {i}: ID='{entity_ids[i]}' at ({x:.2f}, {y:.2f})")

        # Show last few entries if there are more than 5
        if len(entity_ids) > 5:
            print(f"\nLast 3 entities:")
            for i in range(max(len(entity_ids) - 3, 5), len(entity_ids)):
                x, y = positions[i]
                print(f"  {i}: ID='{entity_ids[i]}' at ({x:.2f}, {y:.2f})")

        # Check for any duplicate IDs
        unique_ids = set(entity_ids)
        if len(unique_ids) != len(entity_ids):
            print(f"\nWarning: Found duplicate entity IDs! Unique: {len(unique_ids)}, Total: {len(entity_ids)}")
        else:
            print(f"\nAll entity IDs are unique ✓")

        # Sample a few random entities
        if len(entity_ids) > 10:
            print(f"\n3 Random samples:")
            random_indices = np.random.choice(len(entity_ids), size=3, replace=False)
            for i in random_indices:
                x, y = positions[i]
                print(f"  {i}: ID='{entity_ids[i]}' at ({x:.2f}, {y:.2f})")

        return positions, entity_ids

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None


def main():
    """
    Main function to explore position data.
    """
    # Check what position files are available
    position_dir = "./data/position"
    if os.path.exists(position_dir):
        files = [f for f in os.listdir(position_dir) if f.startswith("step_") and f.endswith(".npz")]
        print(f"Found {len(files)} position files in {position_dir}")
        if files:
            # Sort files by step number
            step_numbers = []
            for f in files:
                try:
                    step_num = int(f.replace("step_", "").replace(".npz", ""))
                    step_numbers.append(step_num)
                except ValueError:
                    continue

            if step_numbers:
                step_numbers.sort()
                print(f"Step range: {min(step_numbers)} to {max(step_numbers)}")
                print(f"First few steps available: {step_numbers[:10]}")

                # Explore the first available step
                first_step = step_numbers[0]
                positions, entity_ids = explore_position_data(first_step)

                # If successful, try a few more steps
                if positions is not None and len(step_numbers) > 1:
                    print(f"\n" + "=" * 60)
                    # Try a middle step
                    mid_step = step_numbers[len(step_numbers) // 2]
                    explore_position_data(mid_step)

                    # Check if sizes change between steps
                    positions2, entity_ids2 = read_position_data(mid_step)
                    if positions2 is not None:
                        if positions.shape != positions2.shape:
                            print(f"\nNote: Position array size changed from {positions.shape} to {positions2.shape}")
                        if len(entity_ids) != len(entity_ids2):
                            print(f"Note: Entity count changed from {len(entity_ids)} to {len(entity_ids2)}")
            else:
                print("No valid step files found")
        else:
            print("No position files found")
    else:
        print(f"Position directory {position_dir} does not exist")
        print("Please make sure the data/position directory exists with step_X.npz files")

if __name__ == "__main__":
    # Set random seed for reproducible random samples
    np.random.seed(42)
    main()

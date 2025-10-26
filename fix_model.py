#!/usr/bin/env python3
"""
Script to fix the pickled model compatibility issue by reloading and resaving it.
"""
import pickle
import numpy as np
import sys

# Monkey-patch to allow loading old numpy random states
class LegacyMT19937:
    """Dummy class to handle old MT19937 states"""
    pass

# Register the legacy bit generator
if not hasattr(np.random, 'MT19937'):
    np.random.MT19937 = LegacyMT19937

# Custom unpickler to handle MT19937
class MT19937Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect old MT19937 references
        if 'MT19937' in name or 'mt19937' in module:
            return np.random.RandomState
        return super().find_class(module, name)

def fix_model(input_path, output_path):
    """Load model with old numpy version and save with new version"""
    print(f"Attempting to load model from {input_path}...")

    # Try multiple loading strategies
    model = None

    # Strategy 1: Direct pickle load with latin1 encoding
    try:
        print("Strategy 1: Loading with latin1 encoding...")
        with open(input_path, 'rb') as f:
            # Skip the numpy random state by using a custom unpickler
            model = pickle.load(f, encoding='latin1')
        print("✓ Strategy 1 succeeded!")
    except Exception as e:
        print(f"✗ Strategy 1 failed: {e}")

    # Strategy 2: Use custom unpickler
    if model is None:
        try:
            print("Strategy 2: Using custom unpickler...")
            with open(input_path, 'rb') as f:
                unpickler = MT19937Unpickler(f, encoding='latin1')
                model = unpickler.load()
            print("✓ Strategy 2 succeeded!")
        except Exception as e:
            print(f"✗ Strategy 2 failed: {e}")

    # Strategy 3: Load with numpy pickle protocol
    if model is None:
        try:
            print("Strategy 3: Loading with numpy load...")
            model = np.load(input_path, allow_pickle=True)
            print("✓ Strategy 3 succeeded!")
        except Exception as e:
            print(f"✗ Strategy 3 failed: {e}")

    if model is None:
        print("❌ All strategies failed. Cannot load the model.")
        sys.exit(1)

    # If we get here, we have a model - now re-save it
    print(f"\n✓ Model loaded successfully!")
    print(f"Model type: {type(model)}")

    # Save with current numpy/pickle version
    print(f"\nSaving fixed model to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✓ Model saved successfully!")
    print(f"\nYou can now use '{output_path}' in your app.")

if __name__ == '__main__':
    input_model = 'best_ml_model'
    output_model = 'best_ml_model_fixed'

    fix_model(input_model, output_model)

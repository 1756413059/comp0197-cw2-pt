import os
import subprocess

def run_script(script_name):
    script_path = os.path.join('scripts', script_name)
    print(f"\n Running: {script_path}")
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors/Warnings:\n", result.stderr)

def main():
    # === Step 1: Generate GT masks
    run_script('generate_gt_masks.py')

    # === Step 2: Train the supervised U-Net model
    run_script('train_supervised.py')

    # === Step 3: Predict and save visualizations
    run_script('predict_and_visualize.py')

    print("\n Pipeline completed successfully.")

if __name__ == '__main__':
    main()
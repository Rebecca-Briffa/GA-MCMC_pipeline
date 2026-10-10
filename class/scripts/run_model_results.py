# import subprocess
# import os
# import shutil
# import re
#
# # Base directory paths
# BASE_CLASS_DIR = "/home/rebecca/GA_Project-main/class"  # Original CLASS source directory path
# TXT_FILE = os.path.join(BASE_CLASS_DIR, "user_eq.txt")  # Path to input equations file
# HEADER_SUBPATH = os.path.join("include", "user_eq.h")  # Relative path to user_eq.h inside CLASS
# PARENT_DIR = "/home/rebecca/GA_Project-main"  # Parent directory for model folders
#
#
# from class_modifiers import (  # Import helper functions from another module
#     extract_new_parameters,
#     add_param_to_struct_h,
#     insert_init_in_background_c,
#     insert_parser_in_input_c,
#     insert_param_in_cclassy_pxd
# )
#
# def update_class_for_new_params(class_dir, expr):
#     """
#     For all new parameters extracted from expr,
#     automatically update necessary CLASS files.
#     """
#     new_params = extract_new_parameters(expr)  # Extract new param names from the fluid equation expression
#     if not new_params:
#         print("No new parameters to insert.")  # Nothing to do if no new parameters
#         return
#
#     print(f"Detected new parameters: {new_params}")
#
#     # Paths to key CLASS source files to modify
#     header_h = os.path.join(class_dir, "include", "background.h")  # Header file where struct lives
#     background_c = os.path.join(class_dir, "source", "background.c")  # Source file to initialize params
#     input_c = os.path.join(class_dir, "source", "input.c")  # Source file to add parser code
#     cclassy_pxd = os.path.join(class_dir, "python", "cclassy.pxd")  # Cython header for Python bindings
#
#     # Loop over all new parameters and update all relevant files accordingly
#     for param in new_params:
#         print(f"Updating files for parameter: {param}")
#         add_param_to_struct_h(header_h, param)  # Add param to C struct in background.h
#         insert_init_in_background_c(background_c, param)  # Initialize param in background.c
#         insert_parser_in_input_c(input_c, param)  # Add code for param in input.c
#         insert_param_in_cclassy_pxd(cclassy_pxd, param)  # Add param to Python binding header
#
#     # After all params are declared, update the FLUID_W_FLD macro call with new params
#     update_fluid_w_call(background_c, new_params)
#
#
# def read_equations():
#     """
#     Reads multiple model definitions from user_eq.txt.
#     Returns a list of dictionaries with MODEL and EXPR keys.
#     """
#     models = []
#     current = {}
#     with open(TXT_FILE, "r") as f:  # Open the user_eq.txt file for reading
#         for line in f:
#             if "=" in line:  # Look for lines containing key=value
#                 key, val = line.strip().split("=")  # Split on '=' and strip whitespace
#                 current[key.strip()] = val.strip()  # Store in current dict
#             elif line.strip() == "" and current:  # Blank line indicates end of one model entry
#                 models.append(current)  # Add collected model data to list
#                 current = {}  # Reset for next model
#         if current:  # If last model doesn't end with a blank line, add it anyway
#             models.append(current)
#     return models  # Return list of dicts, each with MODEL and EXPR
#
#
# def prepare_model_directory(model_name):
#     """
#     Prepares a model-specific directory that contains a fresh copy of the CLASS source
#     located in BASE_CLASS_DIR. Returns the path to the 'class' subdirectory inside
#     the model-specific folder (e.g., /home/rebecca/CLASS/w0wa/class).
#     """
#     model_root = os.path.join(PARENT_DIR, model_name)  # Model root folder path
#     model_class_dir = os.path.join(model_root, "class")  # CLASS source inside model folder
#
#     # Remove existing directory if present to start fresh
#     if os.path.exists(model_root):
#         try:
#             shutil.rmtree(model_root)  # Delete whole folder recursively
#             print(f"Removed existing directory: {model_root}")
#         except Exception as e:
#             print(f"Error removing {model_root}: {e}")
#             raise  # Propagate error if unable to remove
#
#     # Copy the base CLASS source into new model-specific folder
#     try:
#         shutil.copytree(BASE_CLASS_DIR, model_class_dir)
#         print(f"CLASS source copied to: {model_class_dir}")
#     except Exception as e:
#         print(f"Error copying CLASS source to {model_class_dir}: {e}")
#         raise  # Propagate error if copy fails
#
#     return model_class_dir  # Return path where make will be run
#
#
# def write_header(expr, class_dir):
#     """
#     Dynamically writes the fluid equation macro with all parameters used in the expression.
#     """
#     header_file = os.path.join(class_dir, HEADER_SUBPATH)  # Full path to user_eq.h
#     params = extract_new_parameters(expr)  # Extract any new parameters from expr
#     full_param_list = ["a", "w0", "wa"] + sorted(params)  # Standard params + new ones
#     param_string = ", ".join(full_param_list)  # Create macro argument string
#     macro = f"#define FLUID_W_FLD({param_string}) ({expr})\n"  # Create macro definition line
#     with open(header_file, "w") as f:  # Write macro line to user_eq.h, overwriting
#         f.write(macro)
#     print(f"Header written with macro: FLUID_W_FLD({param_string})")  # Confirmation print
#
# def update_fluid_w_call(background_c_path, new_params):
#     """
#     Updates the call to FLUID_W_FLD(...) in background.c to include any new parameters.
#     """
#     with open(background_c_path, "r") as f:
#         content = f.read()  # Read entire background.c content
#
#     # Regex pattern to find the line assigning *w_fld with FLUID_W_FLD call
#     pattern = r"\*w_fld\s*=\s*FLUID_W_FLD\([^)]*\);"
#
#     # Compose the new argument list to FLUID_W_FLD including all new params with pba-> prefix
#     full_args = ["a", "pba->w0_fld", "pba->wa_fld"] + [f"pba->{p}" for p in sorted(new_params)]
#     new_call = f"*w_fld = FLUID_W_FLD({', '.join(full_args)});"  # New line replacing old call
#
#     content_new = re.sub(pattern, new_call, content)  # Substitute the old call with new one
#
#     with open(background_c_path, "w") as f:
#         f.write(content_new)  # Write updated content back to background.c
#
#     print(f"Updated FLUID_W_FLD call in {background_c_path}")  # Confirmation
#
#
# def compile_class(class_dir):
#     """
#     Compiles CLASS using 'make clean' and 'make all' in the given directory.
#     """
#     print(f"Running make in: {class_dir}")  # Inform user where compilation is happening
#     subprocess.run(["make", "clean"], cwd=class_dir, check=True)  # Clean old build files
#     subprocess.run(["make", "all", "-f", "Makefile"], cwd=class_dir, check=True)  # Build CLASS
#     print("CLASS compiled")  # Confirmation
#
#
# def main():
#     models = read_equations()  # Read model definitions from user_eq.txt
#     compiled_models = []  # Track models that compiled successfully
#     skipped_models = []   # Track models that failed or skipped
#
#     for model_data in models:
#         expr = model_data.get("EXPR")  # Fluid equation expression
#         model_name = model_data.get("MODEL", "unnamed")  # Model name or fallback
#
#         if expr is None:  # Skip if expression missing
#             print(f"Skipping model '{model_name}' (missing EXPR)")
#             skipped_models.append(model_name)
#             continue
#
#         print(f"\nCompiling model '{model_name}'...")
#
#         try:
#             class_dir = prepare_model_directory(model_name)  # Prepare fresh CLASS source for model
#             write_header(expr, class_dir)  # Write the fluid macro with correct parameters
#
#             # Add new parameters to all CLASS source files as needed
#             update_class_for_new_params(class_dir, expr)
#
#             compile_class(class_dir)  # Compile CLASS for this model
#             compiled_models.append(model_name)  # Track success
#             print(f"Model '{model_name}' compiled successfully.\n")
#         except Exception as e:
#             print(f"Compilation failed for model '{model_name}': {e}")  # Error handling
#             skipped_models.append(model_name)
#
#     # Print summary of compilation results
#     print("\nCompilation Summary")
#     print("----------------------------")
#     if compiled_models:
#         print(" Successfully compiled:")
#         for name in compiled_models:
#             print(f"  - {name}")
#     if skipped_models:
#         print("\n Skipped or failed:")
#         for name in skipped_models:
#             print(f"  - {name}")
#
#
# if __name__ == "__main__":
#     main()  # Run main function if script executed as main program


import glob
import os
import re
import shutil
import sys
import subprocess
from cobaya_runner import *

# Base directory paths (override in Docker/cluster via env vars)
THIS_FILE = os.path.abspath(__file__)
SCRIPTS_DIR = os.path.dirname(THIS_FILE)
_DEFAULT_CLASS_DIR = os.path.dirname(SCRIPTS_DIR)
BASE_CLASS_DIR = os.path.abspath(
    os.environ.get("GA_CLASS_DIR", _DEFAULT_CLASS_DIR)
)
TXT_FILE = os.path.join(BASE_CLASS_DIR, "user_eq.txt")
HEADER_SUBPATH = os.path.join("include", "user_eq.h")
_DEFAULT_PROJECT_ROOT = os.path.dirname(_DEFAULT_CLASS_DIR)
PROJECT_ROOT = os.path.abspath(
    os.environ.get("GA_PROJECT_ROOT", _DEFAULT_PROJECT_ROOT)
)
# Where gen_* model directories are written (default: same as project root)
PARENT_DIR = os.path.abspath(
    os.environ.get("GA_OUTPUT_DIR", os.environ.get("GA_PROJECT_ROOT", PROJECT_ROOT))
)

from class_modifiers import (  # Import helper functions from another module
    extract_new_parameters,
    add_param_to_struct_h,
    insert_init_in_background_c,
    insert_parser_in_input_c,
    insert_extra_params_in_cclassy_pxd,
    patch_background_clp_macros,
)

def update_class_for_new_params(class_dir, expr):
    new_params = extract_new_parameters(expr)
    if not new_params:
        print("No new parameters to insert.")
        return

    print(f"Detected new parameters: {new_params}")

    header_h = os.path.join(class_dir, "include", "background.h")
    background_c = os.path.join(class_dir, "source", "background.c")
    input_c = os.path.join(class_dir, "source", "input.c")
    cclassy_pxd = os.path.join(class_dir, "python", "cclassy.pxd")

    for param in new_params:
        print(f"Updating files for parameter: {param}")
        add_param_to_struct_h(header_h, param)
        insert_init_in_background_c(background_c, param)
        insert_parser_in_input_c(input_c, param)

    insert_extra_params_in_cclassy_pxd(cclassy_pxd, new_params)

    update_fluid_w_call(background_c, new_params)
    patch_background_clp_macros(background_c, new_params)


def read_equations():
    models = []
    current = {}
    with open(TXT_FILE, "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=")
                current[key.strip()] = val.strip()
            elif line.strip() == "" and current:
                models.append(current)
                current = {}
        if current:
            models.append(current)
    return models


def prepare_model_directory(model_name):
    model_root = os.path.join(PARENT_DIR, model_name)
    model_class_dir = os.path.join(model_root, "class")

    if os.path.exists(model_root):
        try:
            shutil.rmtree(model_root)
            print(f"Removed existing directory: {model_root}")
        except Exception as e:
            print(f"Error removing {model_root}: {e}")
            raise

    try:
        shutil.copytree(BASE_CLASS_DIR, model_class_dir)
        print(f"CLASS source copied to: {model_class_dir}")
    except Exception as e:
        print(f"Error copying CLASS source to {model_class_dir}: {e}")
        raise

    return model_class_dir


def write_header(expr, class_dir):
    header_file = os.path.join(class_dir, HEADER_SUBPATH)
    params = extract_new_parameters(expr)
    full_param_list = ["a", "w0", "wa"] + sorted(params)
    macro = f"#define FLUID_W_FLD({', '.join(full_param_list)}) ({expr})\n"

    os.makedirs(os.path.dirname(header_file), exist_ok=True)

    # write macro
    with open(header_file, "w") as f:
        f.write("// Auto-generated from user_eq.txt\n")
        f.write(macro)

    print(f"✅ Wrote FLUID_W_FLD in {header_file}:\n{macro}")


def update_fluid_w_call(background_c_path, new_params):
    with open(background_c_path, "r") as f:
        content = f.read()

    pattern = r"\*w_fld\s*=\s*FLUID_W_FLD\([^)]*\);"
    full_args = ["a", "pba->w0_fld", "pba->wa_fld"] + [f"pba->{p}" for p in sorted(new_params)]
    new_call = f"*w_fld = FLUID_W_FLD({', '.join(full_args)});"

    content_new = re.sub(pattern, new_call, content)

    with open(background_c_path, "w") as f:
        f.write(content_new)

    print(f"Updated FLUID_W_FLD call in {background_c_path}, {new_call}")


def compile_class(class_dir, model_name, master_classy_dir=None):
    """
    Compile the C side of CLASS only (libclass.a). Python wrapper is built
    separately in build_model_classy() so cclassy.pxd edits are not wiped.
    """
    print(f"\n--- Compiling CLASS for model '{model_name}' ---")
    print(f"CLASS directory: {class_dir}")

    subprocess.run(["make", "clean"], cwd=class_dir, check=True)
    subprocess.run(["make", "class", "libclass.a"], cwd=class_dir, check=True)

    libclass_path = os.path.join(class_dir, "libclass.a")
    if not os.path.exists(libclass_path):
        raise FileNotFoundError(f"libclass.a not found in {class_dir}")

    build_lib_dir = os.path.join(class_dir, "build", "lib")
    os.makedirs(build_lib_dir, exist_ok=True)
    subprocess.run(["cp", libclass_path, build_lib_dir], check=True)

    print(f"CLASS C library compiled for model '{model_name}'.")


def build_master_class(master_class_dir=None):
    if master_class_dir is None:
        master_class_dir = BASE_CLASS_DIR
    python_dir = os.path.join(master_class_dir, "python")

    # Use the current Python executable
    python_exe = sys.executable

    print(f"\n--- Building MASTER CLASS at: {master_class_dir} ---")
    subprocess.run(["make", "clean"], cwd=master_class_dir, check=True)
    subprocess.run(["make", "all"], cwd=master_class_dir, check=True)
    print("Master CLASS compiled successfully.")

    # Install CLASSy using the same Python as this script
    subprocess.run(
        [python_exe, "-m", "pip", "install", "--no-build-isolation", "--user", "."],
        cwd=python_dir,
        check=True
    )
    print(f"Master CLASSy installed at: {python_dir}")


def build_model_classy(class_dir, model_name):
    """
    Build the model-specific classy.so under python/build/lib.*.

    Cobaya's get_compiled_import_path() only looks in build/lib.<platform>-cpython-XY/,
    not at a top-level --inplace .so. Must run AFTER update_class_for_new_params() so
    cclassy.pxd includes w3, w4, etc.
    """
    model_python = os.path.join(class_dir, "python")
    if not os.path.isdir(model_python):
        raise FileNotFoundError(
            f"Missing {model_python}. Model CLASS copy should include python/."
        )

    for stale in glob.glob(os.path.join(model_python, "build", "**", "classy*.so"), recursive=True):
        os.remove(stale)
    for stale in glob.glob(os.path.join(model_python, "classy*.so")):
        os.remove(stale)

    print(f"Building model-specific CLASSy in {model_python} ...")
    subprocess.run(
        [sys.executable, "setup.py", "build"],
        cwd=model_python,
        check=True,
    )

    build_dir = os.path.join(model_python, "build")
    lib_dirs = [
        d for d in os.listdir(build_dir)
        if d.startswith("lib.") and os.path.isdir(os.path.join(build_dir, d))
    ] if os.path.isdir(build_dir) else []
    if not lib_dirs:
        raise FileNotFoundError(
            f"No build/lib.* folder under {model_python} after CLASSy build. "
            "Cobaya cannot load this model's classy module."
        )
    print(f"Model-specific CLASSy ready at: {os.path.join(model_python, 'build', lib_dirs[0])}")

from user_generator_GA import UserEquationGenerator


def main():
    # Read model equations
    models = read_equations()
    compiled_models = []
    skipped_models = []

    master_classy_dir = os.path.join(BASE_CLASS_DIR, "python")  # master CLASSy path

    for model_data in models:

        expr = model_data.get("EXPR")
        print(expr)
        model_name = model_data.get("MODEL", "unnamed")

        if expr is None:
            print(f"Skipping model '{model_name}' (missing EXPR)")
            skipped_models.append(model_name)
            continue

        print(f"\n--- Preparing model '{model_name}' ---")
        try:
            # --- Dynamic paths ---
            model_root = os.path.join(PARENT_DIR, model_name)          # model folder
            class_dir = os.path.join(model_root, "class")              # model-specific CLASS
            chains_dir = os.path.join(model_root, "class", "scripts", "chains")  # Cobaya chains

            # Remove old model folder if exists
            if os.path.exists(model_root):
                shutil.rmtree(model_root)
                print(f"Removed existing model directory: {model_root}")

            # Copy master CLASS source to model folder
            shutil.copytree(BASE_CLASS_DIR, class_dir)
            print(f"Copied CLASS source to: {class_dir}")

            # --- Update CLASS for new parameters ---
            generator = UserEquationGenerator(class_dir, model_name)
            generator.run()

            update_class_for_new_params(class_dir, expr)

            # --- Compile CLASS for this model (no CLASSy reinstall) ---
            compile_class(class_dir, model_name)
            build_model_classy(class_dir, model_name)

            # --- Run Cobaya ---
            #run_cobaya(expr, class_dir, model_name, master_classy_dir=master_classy_dir)

            compiled_models.append(model_name)
            print(f"Model '{model_name}' compiled and ran successfully.\n")

        except Exception as e:
            print(f"Compilation or Cobaya run failed for model '{model_name}': {e}")
            skipped_models.append(model_name)

    # --- Summary ---
    print("\nCompilation Summary")
    print("----------------------------")
    if compiled_models:
        print(" Successfully compiled and run:")
        for name in compiled_models:
            print(f"  - {name}")
    if skipped_models:
        print("\n Skipped or failed:")
        for name in skipped_models:
            print(f"  - {name}")




if __name__ == "__main__":
    main()
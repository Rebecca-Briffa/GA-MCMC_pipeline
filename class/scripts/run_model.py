import subprocess
import os

CLASS_DIR = "/home/rebecca/CLASS/class"
TXT_FILE = os.path.join(CLASS_DIR, "user_eq.txt")
HEADER_FILE = os.path.join(CLASS_DIR, "include/user_eq.h")


def read_equation():
    with open(TXT_FILE, "r") as f:
        lines = f.readlines()
    model_info = {}
    for line in lines:
        if "=" in line:
            key, val = line.strip().split("=")
            model_info[key.strip()] = val.strip()
    return model_info


def write_header(expr):
    with open(HEADER_FILE, "w") as f:
        f.write(f"#define FLUID_W_FLD(a, w0, wa) ({expr})\n")
    print(f"✅ Header written with expression: {expr}")


def compile_class():
    subprocess.run(["make", "clean"], cwd=CLASS_DIR, check=True)
    subprocess.run(["make", "all"], cwd=CLASS_DIR, check=True)
    print("✅ CLASS compiled")


def main():
    model_data = read_equation()
    expr = model_data.get("EXPR", None)
    model = model_data.get("MODEL", "default")

    if expr is None:
        raise ValueError("'EXPR' not found in fluid_equation.txt")

    write_header(expr)
    compile_class()

    print(f"✅ Model '{model}' compiled with fluid equation: {expr}")


if __name__ == "__main__":
    main()

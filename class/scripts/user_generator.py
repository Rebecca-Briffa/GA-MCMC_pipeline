import os
import re
from sympy import symbols, sympify, diff, integrate, simplify

from class_modifiers import extract_new_parameters, expr_to_c

class UserEquationGenerator:
    """
    Reads user_eq.txt, generates w(a), dw/da, and integral macros,
    and writes them to include/user_eq.h for CLASS.
    """

    def __init__(self, base_class_dir, model_name):
        self.base_class_dir = base_class_dir
        self.model_name = model_name
        self.txt_file = os.path.join(base_class_dir, "user_eq.txt")
        self.header_file = os.path.join(base_class_dir, "include", "user_eq.h")
        self.a = symbols("a")

    # ------------------------------
    # Extract extra parameters from expression
    # ------------------------------
    def extract_extra_parameters(self, expr):
        """
        Return list of parameters in expr beyond a, w0, wa
        """
        all_symbols = expr.free_symbols
        return sorted([str(s) for s in all_symbols if str(s) not in ("a", "w0", "wa")])

    # ------------------------------
    # Parse user_eq.txt for specific model
    # ------------------------------
    def parse_expr_for_model(self):
        if not os.path.exists(self.txt_file):
            raise FileNotFoundError(f"Missing file: {self.txt_file}")

        with open(self.txt_file, "r") as f:
            lines = f.readlines()

        current_model = None
        for line in lines:
            line = line.strip()
            if line.startswith("MODEL="):
                current_model = line.split("=", 1)[1]
                continue
            if line.startswith("EXPR=") and current_model == self.model_name:
                return line.split("=", 1)[1]

        raise ValueError(f"MODEL='{self.model_name}' not found in user_eq.txt")

    # ------------------------------
    # Main header generation
    # ------------------------------
    def ensure_user_eq_included(class_dir):
        """Automatically add #include "user_eq.h" in background.c if missing"""
        file_path = os.path.join(class_dir, "source", "background.c")
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Only add if not already present
        if not any('user_eq.h' in l for l in lines):
            for i, line in enumerate(lines):
                # Insert after other #includes
                if line.strip().startswith("#include") and not line.strip().startswith("#include \"user_eq.h\""):
                    lines.insert(i + 1, '#include "user_eq.h"\n')
                    break
            with open(file_path, "w") as f:
                f.writelines(lines)
            print("✅ Added #include \"user_eq.h\" to background.c")
        else:
            print("ℹ️ user_eq.h already included in background.c")

    def generate_header(self):
        # Parse expression
        expr_str = self.parse_expr_for_model()
        w_expr = sympify(expr_str)

        # Derivative
        dw_da_expr = simplify(diff(w_expr, self.a))

        # Definite integral from a to 1 (CLASS convention; see user_generator_GA.py)
        extra_params = extract_new_parameters(expr_str)

        integrand = 3 * (1 + w_expr) / self.a
        try:
            integral_expr = simplify(integrate(integrand, (self.a, self.a, 1)))
            integral_c_code = expr_to_c(integral_expr, extra_params)
        except Exception as e:
            integral_c_code = "0.0"
            print("⚠️ Analytic integral failed, using 0.0:", e)

        w_c_code = expr_to_c(w_expr, extra_params)
        dw_da_c_code = expr_to_c(dw_da_expr, extra_params)

        # Prepare macros (keep ... for extra args)
        macros = [
            "// Auto-generated from user_eq.txt\n",
            f"#define FLUID_W_FLD(a, w0, wa, ...) ({w_c_code})\n",
            f"#define DFLUID_W_FLD_DA(a, w0, wa, ...) ({dw_da_c_code})\n",
            f"#define INTEGRAL_FLD(a, w0, wa, ...) ({integral_c_code})\n",
        ]

        # Ensure include dir exists
        os.makedirs(os.path.dirname(self.header_file), exist_ok=True)

        # Write header file
        with open(self.header_file, "w") as f:
            f.writelines(macros)

        print(f"✅ Generated {self.header_file}")
        for m in macros:
            print(m.strip())

    # ------------------------------
    # Convenience runner
    # ------------------------------
    def run(self):
        self.generate_header()


if __name__ == "__main__":
    # Example usage
    base_dir = "../"  # Replace with your CLASS base directory
    model = "MY_MODEL"  # Replace with your model name in user_eq.txt
    gen = UserEquationGenerator(base_dir, model)
    gen.run()

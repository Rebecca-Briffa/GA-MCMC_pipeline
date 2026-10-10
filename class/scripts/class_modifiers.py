import re
import os

from sympy import Symbol, ccode


# Identifiers that appear in w(a) strings but are functions/constants, not CLASS fields
# (e.g. w4*log(a) must not create a parameter named "log").
_RESERVED_EXPR_IDENTIFIERS = frozenset(
    (
        "log", "log10", "ln", "exp", "sqrt", "pow",
        "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh",
        "fabs", "abs", "min", "max",
        "pi", "M_PI",
    )
)


def pba_subs_map(extra_params):
    """Map w3, w4, ... to C struct fields for sympy → C conversion."""
    return {Symbol(p): Symbol(f"pba_{p}") for p in extra_params}


def expr_to_c(expr, extra_params):
    """Convert a sympy expression to C, with extra params as pba->field."""
    if not extra_params:
        return ccode(expr)
    code = ccode(expr.subs(pba_subs_map(extra_params)))
    for p in extra_params:
        code = code.replace(f"pba_{p}", f"pba->{p}")
    return code


def extract_new_parameters(expr):
    # Coefficients w0, wa and scale factor a are macro arguments, not new struct fields
    known_params = {"w0", "wa", "a"}

    all_params = set(re.findall(r"\b[a-zA-Z_]\w*\b", expr))

    return sorted(all_params - known_params - _RESERVED_EXPR_IDENTIFIERS)


def add_param_to_struct_h(file_path, param):
    # Read the entire header file
    with open(file_path, "r") as f:
        lines = f.readlines()

    inside_struct = False
    new_lines = []

    for line in lines:
        # Detect start of the background struct declaration
        if "struct background" in line:
            inside_struct = True

        # When reaching the end of the struct, add new parameter before closing brace
        if inside_struct and line.strip() == "};":
            new_lines.append(f" double {param};\n")
            inside_struct = False

        new_lines.append(line)

    # Write back updated file
    with open(file_path, "w") as f:
        f.writelines(new_lines)


def insert_init_in_background_c(file_path, param):
    with open(file_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "background_init(struct background *pba" in line:
            for j in range(i, len(lines)):
                if "{" in lines[j]:
                    lines.insert(j + 1, f" pba->{param} = 0.0;\n")
                    lines.insert(j + 2, " pba->fluid_equation_of_state = CLP;\n")
                    break
            break

    with open(file_path, "w") as f:
        f.writelines(lines)


def insert_parser_in_input_c(file_path, param):
    """
    Register reading `param` in input.c so CLASS marks it consumed.
    """

    with open(file_path, "r") as f:
        lines = f.readlines()

    new_line = f' class_read_double("{param}",pba->{param});\n'

    # avoid duplicate insertion
    if any(f'"{param}"' in ln and "class_read_double" in ln for ln in lines):
        return

    anchor = "/* ** END OF BUDGET EQUATION ** */"
    inserted = False

    for i, line in enumerate(lines):
        if anchor in line:
            lines.insert(i + 1, "\n" + new_line)
            inserted = True
            break

    if not inserted:
        raise RuntimeError(
            f"Could not find budget anchor {anchor!r} in {file_path}; "
            "cannot add class_read_double for new fluid parameter."
        )

    with open(file_path, "w") as f:
        f.writelines(lines)


def patch_background_clp_macros(background_c_path, new_params=None):
    """
    Route CLP dw/da and integral through user_eq.h macros.

    Without this, only w(a) uses the custom equation while CLASS still applies
    CPL formulas for the derivative and integral, which can segfault at some
    parameter points.
    """
    macro_args = ", ".join(["a", "pba->w0_fld", "pba->wa_fld"] + [
        f"pba->{p}" for p in sorted(new_params or [])
    ])

    with open(background_c_path, "r") as f:
        src = f.read()

    if "DFLUID_W_FLD_DA" in src and "INTEGRAL_FLD" in src:
        return

    old_dw = "  case CLP:\n    *dw_over_da_fld = - pba->wa_fld;\n    break;"
    new_dw = (
        "  case CLP:\n"
        f"    *dw_over_da_fld = DFLUID_W_FLD_DA({macro_args});\n"
        "    break;"
    )
    old_int = (
        "  case CLP:\n"
        "    *integral_fld = 3.*((1.+pba->w0_fld+pba->wa_fld)*log(1./a) + pba->wa_fld*(a-1.));\n"
        "    break;"
    )
    new_int = (
        "  case CLP:\n"
        f"    *integral_fld = INTEGRAL_FLD({macro_args});\n"
        "    break;"
    )

    if old_dw not in src or old_int not in src:
        raise RuntimeError(
            f"Could not find CPL hardcoded dw/integral blocks in {background_c_path}"
        )

    src = src.replace(old_dw, new_dw).replace(old_int, new_int)

    with open(background_c_path, "w") as f:
        f.write(src)

    print(f"Patched CLP dw/da and integral macros in {background_c_path}")


_CCLASSY_BACKGROUND_BLOCK = re.compile(
    r"(cdef struct background:\s.*?"
    r"        double wa_fld\n"
    r")((?:        double \w+\n)*?)"
    r"(        double cs2_fld\n)",
    re.DOTALL,
)


def insert_extra_params_in_cclassy_pxd(file_path, params):
    """
    Insert extra fluid coefficients (w2, w3, w4, ...) into the background
    struct in cclassy.pxd, immediately before cs2_fld.

    Safe to call with multiple parameters; existing fields are left unchanged.
    """
    extra = sorted(set(params))
    if not extra:
        return

    with open(file_path, "r", encoding="utf-8") as f:
        src = f.read()

    match = _CCLASSY_BACKGROUND_BLOCK.search(src)
    if not match:
        raise RuntimeError(
            f"Could not find background struct (wa_fld / cs2_fld) in {file_path}; "
            f"cannot add {extra} to cclassy.pxd."
        )

    prefix, middle, suffix = match.groups()
    existing = set(re.findall(r"double (\w+)", middle))
    to_add = [p for p in extra if p not in existing]
    if not to_add:
        return

    new_middle = middle + "".join(f"        double {p}\n" for p in to_add)
    new_block = prefix + new_middle + suffix
    src = src[: match.start()] + new_block + src[match.end() :]

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(src)

    print(f"Added to cclassy.pxd background struct: {to_add}")


def insert_param_in_cclassy_pxd(file_path, param):
    """Insert one extra parameter; prefer insert_extra_params_in_cclassy_pxd for batches."""
    insert_extra_params_in_cclassy_pxd(file_path, [param])
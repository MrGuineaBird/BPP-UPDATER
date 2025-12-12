import shutil
import requests
import tempfile
import subprocess
import sys
import ast
import os
import time

# =============================
# Config / Packaging helpers
# =============================
# Toggle to disallow Python imports from within .bpm modules (optional sandbox)
SAFE_MODE = False

# Locate base directory (works for running .py and PyInstaller/auto-py-to-exe builds)
# Determine base directory for modules
if getattr(sys, 'frozen', False):
    # Running as a PyInstaller EXE
    BASE_DIR = os.path.dirname(sys.executable)
else:
    # Running as a normal Python script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_module_file(name):
    if not name.endswith(".bpm"):
        name += ".bpm"

    module_path = os.path.join(BASE_DIR, "modules", name)

    if not os.path.exists(module_path):
        raise FileNotFoundError(
            f"Module file '{name}' not found in modules directory at: {os.path.join(BASE_DIR, 'modules')}"
        )

    try:
        with open(module_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(module_path, "r", encoding="utf-16") as f:
            return f.read()


CURRENT_VERSION = "4.0"

LATEST_VERSION_URL = "https://raw.githubusercontent.com/<username>/<repo>/main/latest.txt"
LATEST_EXE_URL     = "https://raw.githubusercontent.com/<username>/<repo>/main/bpp.exe"


def check_for_update():
    try:
        version = requests.get(LATEST_VERSION_URL, timeout=5).text.strip()
        if version != CURRENT_VERSION:
            return version
    except Exception:
        return False
    return False


def download_update():
    temp_path = os.path.join(tempfile.gettempdir(), "bpp_new.exe")
    with requests.get(LATEST_EXE_URL, stream=True) as r:
        r.raise_for_status()
        with open(temp_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
    return temp_path


def apply_update(new_exe):
    old_exe = sys.executable
    backup = old_exe + ".old"

    # Delete old backup if exists
    if os.path.exists(backup):
        os.remove(backup)

    # Backup existing exe
    shutil.move(old_exe, backup)
    # Replace with new
    shutil.move(new_exe, old_exe)

    # Relaunch new version
    subprocess.Popen([old_exe])
    sys.exit(0)


def auto_update():
    new_ver = check_for_update()
    if not new_ver:
        return

    print(f"New version available: {new_ver}")
    print("Updating B++ automatically...")

    new_exe = download_update()
    apply_update(new_exe)

# =============================
# Tokenizer
# =============================
def tokenize(src):
    tokens = []
    for lineno, line in enumerate(src.splitlines(), start=1):
        # Handle inline comments outside quotes, support escaped quotes
        in_quotes = False
        result = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == '"':
                # count backslashes immediately before quote to detect escaping
                j = i - 1
                backslashes = 0
                while j >= 0 and line[j] == '\\':
                    backslashes += 1
                    j -= 1
                # if backslashes is even, the quote is not escaped
                if backslashes % 2 == 0:
                    in_quotes = not in_quotes
                result.append(ch)
                i += 1
                continue
            if not in_quotes and ch == '#':
                break
            result.append(ch)
            i += 1
        stripped = ''.join(result).strip()
        if not stripped:
            continue
        tokens.append((lineno, stripped))
    return tokens

# =============================
# Environment
# =============================
class Env:
    def __init__(self, parent=None, is_module=False):
        self.vars = {}
        self.parent = parent
        # marker to indicate this env is a function call's local env (affects Return handling)
        self.is_function = False
        # module env flag (for SAFE_MODE checks)
        self.__is_module_load__ = is_module

    def get(self, name):
        if name in self.vars:
            return self.vars[name]
        if self.parent:
            return self.parent.get(name)
        raise NameError(f"Variable '{name}' not defined")

    def set(self, name, value):
        # If variable exists in any parent, update where it exists (nonlocal semantics)
        env = self
        while env:
            if name in env.vars:
                env.vars[name] = value
                return
            env = env.parent
        # otherwise set in current
        self.vars[name] = value

    def define_local(self, name, value):
        # force creation in current local scope (e.g., parameters)
        self.vars[name] = value

# Custom exception used to implement `return` inside functions
class ReturnException(Exception):
    def __init__(self, value):
        self.value = value

# =============================
# Expression Utilities
# =============================
def _split_outside_quotes(s, sep):
    parts, cur, i = [], [], 0
    while i < len(s):
        ch = s[i]
        if ch == '"':
            # append the opening quote
            cur.append(ch)
            i += 1
            # copy until the matching (non-escaped) closing quote
            while i < len(s):
                cur.append(s[i])
                if s[i] == '"':
                    # count backslashes just before this quote to detect escaping
                    j = i - 1
                    backslashes = 0
                    while j >= 0 and s[j] == '\\':
                        backslashes += 1
                        j -= 1
                    if backslashes % 2 == 0:
                        i += 1
                        break
                i += 1
            continue
        if s.startswith(sep, i):
            parts.append(''.join(cur).strip())
            cur = []
            i += len(sep)
            continue
        cur.append(ch)
        i += 1
    parts.append(''.join(cur).strip())
    return parts

def _split_args(arg_str):
    # Split args by commas but not inside quotes or nested structures (type-aware)
    args = []
    cur = []
    stack = []
    i = 0
    while i < len(arg_str):
        ch = arg_str[i]
        if ch == '"':
            cur.append(ch)
            i += 1
            while i < len(arg_str):
                cur.append(arg_str[i])
                if arg_str[i] == '"':
                    # count backslashes
                    j = i - 1
                    backslashes = 0
                    while j >= 0 and arg_str[j] == '\\':
                        backslashes += 1
                        j -= 1
                    if backslashes % 2 == 0:
                        i += 1
                        break
                i += 1
            continue
        if ch in "([{":
            stack.append(ch)
            cur.append(ch)
            i += 1
            continue
        if ch in ")]}":
            if stack:
                opener = stack[-1]
                if (opener == '(' and ch == ')') or (opener == '[' and ch == ']') or (opener == '{' and ch == '}'):
                    stack.pop()
            cur.append(ch)
            i += 1
            continue
        if ch == ',' and not stack:
            args.append(''.join(cur).strip())
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    last = ''.join(cur).strip()
    if last != "":
        args.append(last)
    return args

# Helper to split by top-level plus signs respecting quotes and brackets
def _split_plus(expr):
    parts = []
    cur = []
    stack = []
    i = 0
    in_quotes = False
    while i < len(expr):
        ch = expr[i]
        if ch == '"':
            cur.append(ch)
            # handle escaping
            j = i - 1
            backslashes = 0
            while j >= 0 and expr[j] == '\\':
                backslashes += 1
                j -= 1
            if backslashes % 2 == 0:
                in_quotes = not in_quotes
            i += 1
            continue
        if in_quotes:
            cur.append(ch)
            i += 1
            continue
        if ch in "([{":
            stack.append(ch)
            cur.append(ch)
            i += 1
            continue
        if ch in ")]}":
            if stack:
                opener = stack[-1]
                if (opener == '(' and ch == ')') or (opener == '[' and ch == ']') or (opener == '{' and ch == '}'):
                    stack.pop()
            cur.append(ch)
            i += 1
            continue
        if ch == '+' and not stack and not in_quotes:
            parts.append(''.join(cur).strip())
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    parts.append(''.join(cur).strip())
    return parts

# =============================
# Evaluators
# =============================
def eval_expr(expr, env):
    expr = expr.strip()
    if not expr:
        raise SyntaxError("Empty expression")

    # Try to parse literals safely first (strings, numbers, lists, tuples, booleans)
    try:
        val = ast.literal_eval(expr)
        return val
    except (ValueError, SyntaxError):
        pass

    # Handle string concatenation and plus with proper nesting handling
    if '+' in expr:
        parts = _split_plus(expr)
        if len(parts) > 1:
            values = [eval_expr(p, env) for p in parts]
            if any(isinstance(v, str) for v in values):
                return ''.join(str(v) for v in values)
            total = values[0]
            for v in values[1:]:
                total += v
            return total

    # Function call detection: something(...) possibly dotted like a.b.c(...)
    # We detect by finding the first '(' that is not nested inside brackets/quotes and ends with ')'
    def _find_call(expr):
        stack = []
        in_quotes = False
        for i, ch in enumerate(expr):
            if ch == '"':
                # count backslashes
                j = i - 1
                backslashes = 0
                while j >= 0 and expr[j] == '\\':
                    backslashes += 1
                    j -= 1
                if backslashes % 2 == 0:
                    in_quotes = not in_quotes
            if in_quotes:
                continue
            if ch in "([{":
                stack.append(ch)
            elif ch in ")]}":
                if stack:
                    stack.pop()
            elif ch == '(' and not stack:
                return i
        return -1

    call_pos = _find_call(expr)
    if call_pos != -1 and expr.endswith(")"):
        func_part = expr[:call_pos].strip()
        arg_str = expr[call_pos+1:-1]
        args = [eval_expr(a, env) for a in _split_args(arg_str)] if arg_str.strip() != "" else []

        # Resolve function object by dotted lookup
        parts = func_part.split(".")
        try:
            obj = env.get(parts[0])
        except NameError:
            raise NameError(f"Function or module '{parts[0]}' not defined")
        for part in parts[1:]:
            # If obj is Env, use get
            if isinstance(obj, Env):
                obj = obj.get(part)
            elif isinstance(obj, dict):
                if part in obj:
                    obj = obj[part]
                else:
                    raise NameError(f"Name '{part}' not found in module")
            else:
                # try Python attribute access via getattr
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                elif isinstance(obj, dict) and part in obj:
                    obj = obj[part]
                else:
                    raise SyntaxError(f"Cannot access attribute '{part}' of {type(obj).__name__}")

        # Now obj should be a B++ function object (our marker), or a Python callable
        if isinstance(obj, dict) and obj.get("__bpp_fn__"):
            fn = obj
            # create call_env with parent = function's defining env
            call_env = Env(parent=fn["env"])
            call_env.is_function = True
            # bind parameters as locals
            for p, a in zip(fn["params"], args):
                call_env.define_local(p, a)
            # missing args become undefined
            try:
                run_lines(fn["block"], call_env)
            except ReturnException as re:
                return re.value
            # if function sets __return__ explicitly
            try:
                return call_env.get("__return__")
            except NameError:
                return None
        else:
            # fallback: allow Python callables if stored (e.g. importing Python module function)
            if callable(obj):
                try:
                    return obj(*args)
                except Exception as e:
                    raise RuntimeError(f"Error calling Python callable: {e}")
            raise SyntaxError(f"Not a function: {func_part}")

    # Dotted attribute access (module.var or module.sub.var)
    if "." in expr:
        parts = expr.split(".")
        try:
            val = env.get(parts[0])
        except NameError:
            raise NameError(f"Name '{parts[0]}' not defined")
        for part in parts[1:]:
            if isinstance(val, Env):
                val = val.get(part)
            elif isinstance(val, dict):
                if part in val:
                    val = val[part]
                else:
                    raise NameError(f"Name '{part}' not found in module")
            else:
                # allow Python attribute access
                if hasattr(val, part):
                    val = getattr(val, part)
                elif isinstance(val, dict) and part in val:
                    val = val[part]
                else:
                    raise SyntaxError(f"Cannot access attribute '{part}' of {type(val).__name__}")
        return val

    # Variable lookup
    if expr in env.vars or (env.parent and _exists_in_parents(env, expr)):
        return env.get(expr)

    # As a last resort, use a safe AST evaluator (no calls, no attribute access, no __class__ tricks)
    try:
        node = ast.parse(expr, mode='eval')
    except SyntaxError:
        raise SyntaxError(f"Cannot evaluate expression: {expr}")

    # Allowed node types and operators
    ALLOWED_BINOP = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    ALLOWED_UNARYOP = (ast.UAdd, ast.USub, ast.Not)
    ALLOWED_BOOL_OP = (ast.And, ast.Or)
    ALLOWED_CMP_OPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Str):
            return node.s
        if isinstance(node, ast.Name):
            if node.id in {"True", "False", "None"}:
                return {"True": True, "False": False, "None": None}[node.id]
            return env.get(node.id)
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, ALLOWED_BINOP):
                raise SyntaxError(f"Operator {type(node.op).__name__} not allowed")
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                if isinstance(left, str) or isinstance(right, str):
                    return str(left) + str(right)
            return {
                ast.Add: lambda a, b: a + b,
                ast.Sub: lambda a, b: a - b,
                ast.Mult: lambda a, b: a * b,
                ast.Div: lambda a, b: a / b,
                ast.FloorDiv: lambda a, b: a // b,
                ast.Mod: lambda a, b: a % b,
                ast.Pow: lambda a, b: a ** b,
            }[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, ALLOWED_UNARYOP):
                raise SyntaxError(f"Unary operator {type(node.op).__name__} not allowed")
            operand = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.Not):
                return not operand
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, ALLOWED_BOOL_OP):
                raise SyntaxError(f"Boolean operator {type(node.op).__name__} not allowed")
            if isinstance(node.op, ast.And):
                for value in node.values:
                    if not _eval(value):
                        return False
                return True
            else:
                for value in node.values:
                    if _eval(value):
                        return True
                return False
        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                if not isinstance(op, ALLOWED_CMP_OPS):
                    raise SyntaxError(f"Comparison {type(op).__name__} not allowed")
                right = _eval(comparator)
                if isinstance(op, ast.Eq) and not (left == right):
                    return False
                if isinstance(op, ast.NotEq) and not (left != right):
                    return False
                if isinstance(op, ast.Lt) and not (left < right):
                    return False
                if isinstance(op, ast.LtE) and not (left <= right):
                    return False
                if isinstance(op, ast.Gt) and not (left > right):
                    return False
                if isinstance(op, ast.GtE) and not (left >= right):
                    return False
                left = right
            return True
        if isinstance(node, (ast.List, ast.Tuple)):
            if isinstance(node, ast.List):
                return [_eval(elt) for elt in node.elts]
            else:
                return tuple(_eval(elt) for elt in node.elts)
        if isinstance(node, ast.Dict):
            return {_eval(k): _eval(v) for k, v in zip(node.keys, node.values)}
        raise SyntaxError(f"Disallowed expression: {ast.dump(node)}")

    try:
        return _eval(node)
    except Exception as e:
        if isinstance(e, NameError):
            raise
        raise SyntaxError(f"Cannot evaluate expression: {expr} ({e})")

def _exists_in_parents(env, name):
    e = env.parent
    while e:
        if name in e.vars:
            return True
        e = e.parent
    return False

def eval_condition(expr, env):
    expr = expr.strip()
    if " is not " in expr:
        left, _, right = expr.partition(" is not ")
        return eval_expr(left, env) != eval_expr(right, env)
    if " is " in expr:
        left, _, right = expr.partition(" is ")
        return eval_expr(left, env) == eval_expr(right, env)
    for op in ["<=", ">=", "==", "!=", "<", ">"]:
        if op in expr:
            left, _, right = expr.partition(op)
            lval = eval_expr(left, env)
            rval = eval_expr(right, env)
            if op == "<=":
                return lval <= rval
            if op == ">=":
                return lval >= rval
            if op == "<":
                return lval < rval
            if op == ">":
                return lval > rval
            if op == "==":
                return lval == rval
            if op == "!=":
                return lval != rval
    return bool(eval_expr(expr, env))

# =============================
# Block Handling
# =============================
def collect_block(lines, start):
    """Collect a block starting at index `start` (inclusive).

    The function returns (block_lines, last_index).
    last_index is the index of the last line belonging to the block.
    The block ends either at a matching 'end' at depth 0, or immediately before a sibling
    'elif ...:' or 'else:' at depth 0.
    """
    block = []
    depth = 0
    i = start
    while i < len(lines):
        lineno, cur = lines[i]
        stripped = cur.strip()
        # treat 'end' as closing a block
        if stripped == "end":
            if depth == 0:
                return block, i
            depth -= 1
            block.append((lineno, cur))
            i += 1
            continue
        # If we encounter an elif/else at the same level, treat it as terminating
        # the current block so the caller can handle it.
        if depth == 0 and ( (stripped.startswith("elif ") and stripped.endswith(":")) or stripped == "else:" ):
            # block ends before this line
            return block, i - 1
        # detect nested block starters
        if stripped.endswith(":") and (stripped.startswith("repeat") or stripped.startswith("if") or stripped.startswith("while") or stripped.startswith("def")):
            depth += 1
            block.append((lineno, cur))
            i += 1
            continue
        block.append((lineno, cur))
        i += 1
    # If we exit without finding 'end', it's a syntax error
    raise SyntaxError("Missing 'end'")

# =============================
# Runner
# =============================
DEBUG_MODE = False

def run_lines(lines, env):
    i = 0
    while i < len(lines):
        lineno, line = lines[i]
        try:
            # ---------- function definition ----------
            if line.startswith("def ") and line.endswith(":"):
                sig = line[4:-1].strip()
                func_name, _, args = sig.partition("(")
                func_name = func_name.strip()
                params = [a.strip() for a in args.rstrip(")").split(",") if a.strip()]
                block, jump = collect_block(lines, i+1)
                # store function as dict with a reference to the defining env for closures
                fn_obj = {"__bpp_fn__": True, "params": params, "block": block, "env": env}
                env.set(func_name, fn_obj)
                i = jump

            # ---------- return (inside functions) ----------
            elif line.startswith("return"):
                rest = line[len("return"):].strip()
                if rest:
                    val = eval_expr(rest, env)
                else:
                    val = None
                # set __return__ in env and raise to unwind
                env.set("__return__", val)
                raise ReturnException(val)

            # ---------- export (module naming) ----------
            elif line.startswith("export "):
                # syntax: export "modname"
                arg = line[len("export "):].strip()
                # allow single or double quotes
                if (arg.startswith('"') and arg.endswith('"')) or (arg.startswith("'") and arg.endswith("'")):
                    modname = arg[1:-1]
                else:
                    modname = arg
                env.set("__export_name__", modname)

            # ---------- io & basic commands ----------
            elif line.startswith("say "):
                expr = line[4:].strip()
                print(eval_expr(expr, env))

            elif line.startswith("set ") and " to " in line:
                parts = line.split(" ", 3)
                if len(parts) == 4:
                    _, var, _, val = parts
                    env.set(var, eval_expr(val, env))
                else:
                    raise SyntaxError(f"Invalid set syntax")

            elif line.startswith("add ") and " to " in line:
                num, _, var = line[4:].partition(" to ")
                left = var.strip()
                left_val = env.get(left)
                add_val = eval_expr(num.strip(), env)
                try:
                    env.set(left, left_val + add_val)
                except TypeError:
                    raise TypeError(f"Cannot add non-compatible types to variable '{left}'")

            elif line.startswith("subtract ") and " from " in line:
                num, _, var = line[9:].partition(" from ")
                left = var.strip()
                left_val = env.get(left)
                sub_val = eval_expr(num.strip(), env)
                try:
                    env.set(left, left_val - sub_val)
                except TypeError:
                    raise TypeError(f"Cannot subtract non-numeric from variable '{left}'")

            elif line.startswith("multiply ") and " by " in line:
                var, _, num = line[9:].partition(" by ")
                left = var.strip()
                left_val = env.get(left)
                mul_val = eval_expr(num.strip(), env)
                try:
                    env.set(left, left_val * mul_val)
                except TypeError:
                    raise TypeError(f"Cannot multiply variable '{left}' by non-numeric")

            elif line.startswith("divide ") and " by " in line:
                var, _, num = line[7:].partition(" by ")
                left = var.strip()
                divisor = eval_expr(num.strip(), env)
                if isinstance(divisor, (int, float)) and divisor == 0:
                    raise ZeroDivisionError("Division by zero")
                left_val = env.get(left)
                try:
                    result = left_val / divisor
                except TypeError:
                    raise TypeError(f"Cannot divide non-numeric variable '{left}'")
                # convert float that is integer-valued to int
                if isinstance(result, float) and result.is_integer():
                    result = int(result)
                env.set(left, result)

            elif line.startswith("ask ") and " into " in line:
                prompt, _, var = line[4:].partition(" into ")
                prompt_val = eval_expr(prompt.strip(), env)
                user_input = input(str(prompt_val) + " ").strip()
                env.set(var.strip(), user_input)

            elif line.startswith("write ") and " to " in line:
                expr, _, fname = line[6:].partition(" to ")
                content = str(eval_expr(expr.strip(), env))
                filename = eval_expr(fname.strip(), env)
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(content)

            elif line.startswith("read ") and " into " in line:
                fname, _, var = line[5:].partition(" into ")
                filename = eval_expr(fname.strip(), env)
                with open(filename, "r", encoding="utf-8") as f:
                    content = f.read()
                env.set(var.strip(), content)

            # ---------- import handling (Python modules vs B++ .bpm mods) ----------
            elif line.startswith("import "):
                arg = line[len("import "):].strip()
                # B++ module import if quoted like import "modname"
                if (arg.startswith('"') and arg.endswith('"')) or (arg.startswith("'") and arg.endswith("'")):
                    modname = arg[1:-1]
                    # load file from modules/
                    src = load_module_file(modname)
                    mod_env = Env(is_module=True)
                    # run module code in its own env
                    run_lines(tokenize(src), mod_env)
                    # module name: prefer explicit export, else filename
                    exported_name = None
                    if "__export_name__" in mod_env.vars:
                        exported_name = mod_env.vars.get("__export_name__")
                        # remove the export marker from public vars
                        try:
                            del mod_env.vars["__export_name__"]
                        except KeyError:
                            pass
                    if exported_name:
                        module_name = exported_name
                    else:
                        module_name = os.path.splitext(modname)[0]
                    # store the module environment under module_name (so attribute access uses Env)
                    env.set(module_name, mod_env)

                else:
                    # Python import
                    pyname = arg
                    if SAFE_MODE and getattr(env, "__is_module_load__", False):
                        raise ImportError("Python imports are disabled in SAFE_MODE for modules.")
                    # basic import (no from/import as support for now)
                    module = __import__(pyname)
                    env.set(pyname, module)

            elif line.startswith("repeat ") and line.endswith(" times:"):
                count_expr = line[len("repeat "):-len(" times:")].strip()
                count = int(eval_expr(count_expr, env))
                block, jump = collect_block(lines, i+1)
                for _ in range(count):
                    try:
                        run_lines(block, env)
                    except ReturnException as re:
                        # return inside repeat outside function -> syntax error
                        raise SyntaxError("return outside function")
                i = jump

            elif line.startswith("while ") and line.endswith(":"):
                cond_expr = line[6:-1].strip()
                block, jump = collect_block(lines, i+1)
                while eval_condition(cond_expr, env):
                    try:
                        run_lines(block, env)
                    except ReturnException as re:
                        raise SyntaxError("return outside function")
                i = jump

            elif line.startswith("if ") and line.endswith(":"):
                cond_expr = line[3:-1].strip()
                cond_val = eval_condition(cond_expr, env)
                block, jump = collect_block(lines, i+1)
                if cond_val:
                    run_lines(block, env)
                    i = jump
                else:
                    handled = False
                    j = jump + 1
                    # scan for elif/else at the same level
                    while j < len(lines):
                        lnum2, nxt = lines[j]
                        if nxt.startswith("elif ") and nxt.endswith(":"):
                            cond_expr2 = nxt[5:-1].strip()
                            block2, jump2 = collect_block(lines, j+1)
                            if eval_condition(cond_expr2, env):
                                run_lines(block2, env)
                                j = jump2
                                handled = True
                                break
                            j = jump2 + 1
                            continue
                        elif nxt == "else:":
                            block2, jump2 = collect_block(lines, j+1)
                            run_lines(block2, env)
                            j = jump2
                            handled = True
                            break
                        else:
                            break
                    i = j if handled else jump

            if DEBUG_MODE:
                print(f"DEBUG (line {lineno}): {env.vars}")

        except ReturnException:
            # If we hit a ReturnException here, it means a `return` escaped
            # from a nested run_lines call into a non-function environment.
            # Only re-raise if current env is marked as a function local env.
            if getattr(env, "is_function", False):
                # let function caller handle it
                raise
            else:
                # return outside function
                print(f"[Line {lineno}] SyntaxError: 'return' outside function\n    --> {line}")
                return
        except Exception as e:
            print(f"[Line {lineno}] {type(e).__name__}: {e}\n    --> {line}")
            return
        i += 1

# =============================
# REPL & File Runner
# =============================
def repl():
    print("B++ REPL - type your code. Use 'end' to close blocks. Type 'start' to run. Ctrl+C to exit.")
    env, buffer, block_depth = Env(), [], 0
    while True:
        try:
            prompt = "B++> " if block_depth == 0 else "... "
            line = input(prompt)
            stripped = line.strip()
            if stripped == "start":
                if buffer:
                    run_lines(tokenize("\n".join(buffer)), env)
                    buffer, block_depth = [], 0
                continue
            if not stripped and not buffer:
                continue
            buffer.append(line)
            if stripped.endswith(":") and (stripped.startswith("repeat") or stripped.startswith("if") or stripped.startswith("while") or stripped.startswith("def") or stripped.startswith("elif") or stripped == "else:"):
                block_depth += 1
            elif stripped == "end":
                block_depth = max(0, block_depth - 1)
        except KeyboardInterrupt:
            print("\nExiting B++")
            break
        except Exception as e:
            print("Error:", e)
            buffer, block_depth = [], 0

def run_file(path):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    env = Env()
    run_lines(tokenize(src), env)

if __name__ == "__main__":
    if "--debug" in sys.argv:
        DEBUG_MODE = True
        sys.argv.remove("--debug")
    if len(sys.argv) > 1:
        run_file(sys.argv[1])
    else:
        repl()

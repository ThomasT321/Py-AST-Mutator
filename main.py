#!/usr/bin/env python3
"""
mutation_tool_fixed.py

Two-pass AST-based mutation testing tool (single-file).

Usage:
    python mutation_tool_fixed.py path/to/target_file.py
"""

import ast
import copy
import os
import shutil
import subprocess
import tempfile
import argparse


# Map ast class-name -> replacement ast class-name (for ops)
OP_MUTATION_MAP = {
    "Eq": "NotEq",
    "NotEq": "Eq",
    "Lt": "Gt",
    "Gt": "Lt",
    "Add": "Sub",
    "Sub": "Add",
    "Add":"Mult",
    "Mult":"Add",
}

# Boolean constant flips
CONST_BOOL_MUTATIONS = {
    True: False,
    False: True,
}

# Collector (find mutation sites)
class MutationSiteCollector(ast.NodeVisitor):
    """
    Walk AST once and record mutation sites as dicts.
    Each site contains:
      - kind: "Compare"/"BinOp"/"ConstantBool"
      - lineno, col_offset (location anchor)
      - extra info (op_index for Compare)
      - original name and target name for mutation
    """
    def __init__(self):
        self.sites = []

    def visit_Compare(self, node: ast.Compare):
        for idx, op in enumerate(node.ops):
            op_name = type(op).__name__
            if op_name in OP_MUTATION_MAP:
                site = {
                    "kind": "Compare",
                    "lineno": node.lineno,
                    "col_offset": node.col_offset,
                    "op_index": idx,
                    "orig": op_name,
                    "target": OP_MUTATION_MAP[op_name],
                }
                self.sites.append(site)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):
        op_name = type(node.op).__name__
        if op_name in OP_MUTATION_MAP:
            site = {
                "kind": "BinOp",
                "lineno": node.lineno,
                "col_offset": node.col_offset,
                "orig": op_name,
                "target": OP_MUTATION_MAP[op_name],
            }
            self.sites.append(site)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, bool):
            if node.value in CONST_BOOL_MUTATIONS:
                site = {
                    "kind": "ConstantBool",
                    "lineno": node.lineno,
                    "col_offset": node.col_offset,
                    "orig": node.value,
                    "target": CONST_BOOL_MUTATIONS[node.value],
                }
                self.sites.append(site)
        # no generic_visit needed for constants


# Applier (apply a single mutation to a copied AST)
class SingleSiteApplier(ast.NodeTransformer):
    """
    Apply exactly one mutation at the specified site.
    Matching is done by (kind, lineno, col_offset) and (for Compare) op_index.
    """
    def __init__(self, site):
        super().__init__()
        self.site = site
        self.applied = False

    def matches_location(self, node):
        # Some nodes (e.g., Compare) anchor at the Compare node lineno/col_offset.
        return getattr(node, "lineno", None) == self.site["lineno"] and getattr(node, "col_offset", None) == self.site["col_offset"]

    def visit_Compare(self, node: ast.Compare):
        if self.applied or self.site["kind"] != "Compare":
            return self.generic_visit(node)

        if self.matches_location(node):
            idx = self.site["op_index"]
            if 0 <= idx < len(node.ops):
                op = node.ops[idx]
                if type(op).__name__ == self.site["orig"]:
                    # construct new operator instance
                    new_op_cls = getattr(ast, self.site["target"])
                    node.ops[idx] = new_op_cls()
                    self.applied = True
                    return ast.fix_missing_locations(node)
        return self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):
        if self.applied or self.site["kind"] != "BinOp":
            return self.generic_visit(node)

        if self.matches_location(node):
            if type(node.op).__name__ == self.site["orig"]:
                new_op_cls = getattr(ast, self.site["target"])
                node.op = new_op_cls()
                self.applied = True
                return ast.fix_missing_locations(node)
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        if self.applied or self.site["kind"] != "ConstantBool":
            return node
        if self.matches_location(node):
            if isinstance(node.value, bool) and node.value == self.site["orig"]:
                # replace constant value
                return ast.copy_location(ast.Constant(value=self.site["target"]), node)
        return node


# Utility: generate mutants (two-pass)
def generate_mutants_from_source(source_code):
    """
    Returns list of (description, mutated_source) for every single-site mutant
    """
    root = ast.parse(source_code)
    collector = MutationSiteCollector()
    collector.visit(root)
    sites = collector.sites

    mutants = []
    for i, site in enumerate(sites):
        tree_copy = copy.deepcopy(root)
        applier = SingleSiteApplier(site)
        mutated_tree = applier.visit(tree_copy)
        # if applier didn't apply (shouldn't happen), skip
        if not getattr(applier, "applied", False):
            continue
        mutated_code = ast.unparse(mutated_tree)
        desc = f"{site['kind']} @ {site['lineno']}:{site['col_offset']} ({site['orig']} -> {site['target']})"
        mutants.append((desc, mutated_code))
    return mutants


# ----------------------------
# Test runner and execution
# ----------------------------
def run_pytest_in_cwd(timeout=10):
    """
    Run pytest in the current working directory.
    Return True if tests passed, False if tests failed OR timed out.
    If pytest times out, treat as test-failure detection (mutant considered killed).
    """
    try:
        subprocess.run(["pytest", "-q"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        # hanging tests -> consider mutant detected (killed)
        return False
    except subprocess.CalledProcessError:
        return False


def copy_project_to(tmpdir, exclude_names=None):
    """
    Copy current working dir (project root) to tmpdir.
    Excludes names in exclude_names (list of file/dir names).
    """
    if exclude_names is None:
        exclude_names = {".git", "__pycache__"}
    else:
        exclude_names = set(exclude_names)

    for item in os.listdir("."):
        if item in exclude_names:
            continue
        src = os.path.join(".", item)
        dst = os.path.join(tmpdir, item)
        try:
            if os.path.isdir(src):
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
            else:
                shutil.copy2(src, dst)
        except Exception as e:
            # non-fatal: skip problematic files (keeps tool robust for student projects)
            # but do not silently ignore during development; print for debugging.
            print(f"Warning: failed to copy {src}: {e}")


def mutate_file_and_test(path_to_file):
    """
    Orchestrates mutation of a single file:
      - read file
      - produce mutants
      - for each mutant, create temp copy of project, write mutant file, run pytest
    Returns list of result dicts: {"mutation": desc, "killed": bool}
    """
    with open(path_to_file, "r") as f:
        original = f.read()

    mutants = generate_mutants_from_source(original)

    results = []
    for desc, mutated_source in mutants:
        with tempfile.TemporaryDirectory() as tmpdir:
            copy_project_to(tmpdir)
            target_path = os.path.join(tmpdir, path_to_file)
            # ensure parent dir exists
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with open(target_path, "w") as tf:
                tf.write(mutated_source)

            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                passed = run_pytest_in_cwd(timeout=10)
            finally:
                os.chdir(cwd)

            results.append({"mutation": desc, "killed": not passed})
    return results

import os

def collect_python_files(path):
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        files = []
        for root, _, filenames in os.walk(path):
            for f in filenames:
                if f.endswith(".py"):
                    full = os.path.join(root, f)
                    files.append(full)
        return files
    else:
        print(f"Error: '{path}' is neither file nor directory.")
        return []


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Two-pass AST mutation tester (file or directory).")
    parser.add_argument("path", help="Path to a Python file or directory containing Python files.")
    args = parser.parse_args()

    targets = collect_python_files(args.path)
    if not targets:
        print("No Python files found.")
        return

    all_results = []

    for t in targets:
        print(f"\n=== Mutating {t} ===")
        res = mutate_file_and_test(t)

        if not res:
            print("  No mutation sites found.")
            continue

        # Print detailed results per mutant
        for r in res:
            status = "killed" if r["killed"] else "survived"
            print(f"  {r['mutation']}: {status}")

        all_results.append((t, res))


    print("\n=== Summary ===")
    total_killed = 0
    total_mutants = 0

    for filename, results in all_results:
        killed = sum(1 for r in results if r["killed"])
        total = len(results)
        total_killed += killed
        total_mutants += total
        print(f"{filename}: {killed}/{total} killed")

    print(f"\nOverall: {total_killed}/{total_mutants} killed")



if __name__ == "__main__":
    main()

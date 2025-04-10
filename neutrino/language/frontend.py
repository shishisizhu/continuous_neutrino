"""Parse and flatten Python Tracing DSL"""
from neutrino.language import DTYPES # neutrino/language/__init__.py
import ast
from typing import Optional
from dataclasses import dataclass

@dataclass
class Register:
    name: str
    dtype: str
    init: int

@dataclass
class Probe:
    name:   str                   # name is the key in TOML
    level:  str                   # level of the probe
    pos:    list[str]             # := tracepoint in the paper
    size:   Optional[int] = 0     # number of bytes per thread
    before: Optional[list] = None # snippet inserted before, one of before and after shall be given
    after:  Optional[list] = None # snippet inserted after,  one of before and after shall be given
    
allowed_nodes = {
    ast.Import,     # Imported Stuff
    ast.Module,     # the greatest start
    ast.Name,       # Name of Variable
    ast.Assign,     # Assign Value
    ast.UnaryOp,    # Unary Op, only negative
    ast.BinOp,      # Binary Op, +-*/
    ast.Call,       # Call function
    ast.Attribute,  # Access Attribute of Namespace
    ast.Constant,   # Constant Value
    ast.Expr        # Single Expression
}

binary_ops = {
    ast.Add:  "add",
    ast.Sub:  "sub",
    ast.Mult: "mul",
    ast.Div:  "div"
}

unary_ops = {
    ast.USub: "neg"
}

class NeutrinoVisitor(ast.NodeVisitor):
    def __init__(self, nl_name: str, regs: list[str]):
        super().__init__()
        self.nl_name = nl_name
        self.reg_counter = -1 # make it R0
        self.ir: list[tuple] = []
        self.reg_map: dict[str, str] = {reg: self.fresh_name() for reg in regs}
        # initialize and visit tree

    def fresh_name(self):
        self.reg_counter += 1
        return f"R{self.reg_counter}"

    def visit_Assign(self, node): # Lowered to mov 
        # we shall check if the target has a known name
        name = self.reg_map[node.targets[0].id]
        if isinstance(node.value, ast.Attribute):
            self.ir.append(["mov", name, self.visit(node.value)])
        else:
            new_name = self.visit(node.value) # this is the temporary name
            for inst in self.ir:
                for idx in range(len(inst)):
                    if inst[idx] == new_name:
                        inst[idx] = name
            self.reg_counter -= 1

    def process_operand(self, operand) -> str:
        if isinstance(operand, ast.Name):
            return self.reg_map[operand.id]
        elif isinstance(operand, ast.Constant):
            return operand.value
        elif isinstance(operand, (ast.Attribute, ast.Call, ast.BinOp, ast.UnaryOp)):
            return self.visit(operand)
        else:
            raise ValueError
    
    def visit_BinOp(self, node): # Lowered to add/sub
        lhs = self.process_operand(node.left)
        rhs = self.process_operand(node.right)
        new_name = self.fresh_name()
        if isinstance(node.op, ast.Add):
            self.ir.append(["add", new_name, lhs, rhs])
        elif isinstance(node.op, ast.Sub):
            self.ir.append(["sub", new_name, lhs, rhs])
        elif isinstance(node.op, ast.Mult):
            self.ir.append(["mul", new_name, lhs, rhs])
        elif isinstance(node.op, ast.Div):
            self.ir.append(["div", new_name, lhs, rhs])
        elif isinstance(node.op, ast.LShift):
            self.ir.append(["lsh", new_name, lhs, rhs])
        else:
            raise NotImplementedError()
        self.reg_map[new_name] = new_name
        return new_name

    def visit_UnaryOp(self, node): 
        value = self.process_operand(node.operand)
        new_name = self.fresh_name()
        if isinstance(node.op, ast.USub):
            self.ir.append(["neg", new_name, value])
        else:
            raise NotImplementedError()
        return new_name

    def visit_Call(self, node):
        func_name = self.visit(node.func)
        if func_name == "smid":
            new_name = self.fresh_name()
            self.ir.append(["smid", new_name])
            self.reg_map[new_name] = new_name
            return new_name
        elif func_name == "time":
            new_name = self.fresh_name()
            self.reg_map[new_name] = new_name
            self.ir.append(["time", new_name]) 
            return new_name
        elif func_name == "clock":
            new_name = self.fresh_name()
            self.reg_map[new_name] = new_name
            self.ir.append(["clock", new_name]) 
            return new_name
        elif func_name == "save":
            save_inst = ""
            regs = []
            for keyword in node.keywords:
                if keyword.arg == "dtype":
                    dtype = self.visit(keyword.value)
                    if dtype == "u32": 
                        save_inst = "stw"
                    elif dtype == "u64":
                        save_inst = "stdw"
                elif keyword.arg == "regs":
                    if isinstance(keyword.value, ast.Name):
                        regs.append(self.reg_map[keyword.value.id])
                    elif isinstance(keyword.value, (ast.Tuple, ast.List)):
                        for reg in keyword.value.elts:
                            regs.append(self.reg_map[reg.id])
            if len(regs) == 0 and len(node.args) > 0:
                if isinstance(node.args[0], (ast.Tuple, ast.List)):
                    for reg in node.args[0].elts:
                        regs.append(self.reg_map[self.visit(reg)])
                else:
                    regs.append(self.reg_map[self.visit(node.args[0])])
            for reg in regs:
                self.ir.append([save_inst, reg])
        else:
            raise NotImplementedError()
    
    def visit_Name(self, node):
        return node.id

    def visit_Attribute(self, node):
        if node.value.id == self.nl_name:
            return node.attr
        else:
            raise ValueError(f"can only refer to neutrino.language semantic but got {node.value.id}")

    def visit_Constant(self, node):
        return node

    def generic_visit(self, node):
        if type(node) not in allowed_nodes:
            raise NotImplementedError(f"{type(node).__name__} (lineno: {node.lineno})")
        super().generic_visit(node)


def parse(code: str) -> tuple[list[Register], list[Probe]]:
    """Parse the code into probes"""
    tree = ast.parse(code)
    nl_name: str = None # name of neutrino.language in the code
    regs:    list[Register] = []
    probes:  list[Probe] = []

    for node in tree.body:
        if type(node) is ast.Import and node.names[0].name == "neutrino.language":
            nl_name = node.names[0].asname
        elif type(node) is ast.AnnAssign and node.annotation:
            if node.annotation.value.id == nl_name and node.annotation.attr in DTYPES:
                regs.append(Register(node.target.id, node.annotation.attr, node.value.value)) 
        elif type(node) is ast.FunctionDef and node.decorator_list:
            name = node.name # take func name as probe name
            decorator = node.decorator_list[0]
            if decorator.func.value.id == nl_name and decorator.func.attr == "probe":
                pos, level, ret, size = None, None, False, 0
                for keyword in decorator.keywords:
                    if   keyword.arg == "pos":       pos = keyword.value.value
                    elif keyword.arg == "level":     level = keyword.value.value
                    elif keyword.arg == "ret":       ret = keyword.value.value
                    elif keyword.arg == "size":      size = keyword.value.value
                if not pos or not level: raise ValueError("position must be specified")
                # check if probe existed or not
                visitor = NeutrinoVisitor(nl_name=nl_name, regs=[reg.name for reg in regs])
                visitor.visit(ast.Module(body=node.body)) # Take it as independent code
                probe = Probe(name=name, pos=pos, level=level, size=size)
                if ret:
                    probe.before = visitor.ir
                else:
                    probe.after  = visitor.ir
                probes.append(probe)
    
    return regs, probes

# A Simple Test Case, not really used in production
if __name__ == "__main__":
    code = """
import neutrino
import neutrino.language as nl # API borrowed from Triton :)

gstart : nl.u64 = 0
gend   : nl.u64 = 0
elapsed: nl.u64 = 0

@nl.probe(pos="kernel", level="warp") # broadcast to warp leader
def block_sched_start():
    gstart = nl.clock()

@nl.probe(pos="kernel", ret=True, level="warp", size=16) # save 16 bytes per warp
def block_sched_end():
    gend = nl.clock()
    elapsed = gend - gstart 
    nl.save(gstart, dtype=nl.u64)
    nl.save((elapsed, nl.smid()), dtype=nl.u32) # auto casted"""
    
    regs, probes = parse(code)
    
    print(regs)
    print(probes)

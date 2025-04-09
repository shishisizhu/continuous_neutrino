"""Parse and flatten Python Tracing DSL"""
import ast

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

class RestrictedVisitor(ast.NodeVisitor):
    def __init__(self, regs: list[str]):
        super().__init__()
        self.nlname = "neutrino.language"
        self.temp_counter = -1 # make it R0
        self.ir = []
        self.name_map = {reg: self.fresh_name() for reg  in regs}
        print(self.name_map)

    def fresh_name(self):
        self.temp_counter += 1
        return f"R{self.temp_counter}"

    def visit_Import(self, node):
        for pkg in node.names:
            if pkg.name == self.nlname:
                self.nlname = pkg.asname
                return # find neutrino.language import
    
    def visit_Assign(self, node): # Lowered to mov 
        if node.targets[0].id in self.name_map:
            name = self.name_map[node.targets[0].id]
        else:
            name = self.fresh_name()
        if isinstance(node.value, ast.Attribute):
            new_name = self.fresh_name()
            self.ir.append(f"mov {new_name} {self.visit(node.value)}")
        else:
            new_name = self.visit(node.value)
        self.name_map[node.targets[0].id] = new_name

    def process_operand(self, operand) -> str:
        if isinstance(operand, ast.Name):
            return self.name_map[operand.id]
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
            self.ir.append(f"add {new_name} {lhs} {rhs}")
        elif isinstance(node.op, ast.Sub):
            self.ir.append(f"sub {new_name} {lhs} {rhs}")
        elif isinstance(node.op, ast.Mult):
            self.ir.append(f"mul {new_name} {lhs} {rhs}")
        elif isinstance(node.op, ast.Div):
            self.ir.append(f"div {new_name} {lhs} {rhs}")
        elif isinstance(node.op, ast.LShift):
            self.ir.append(f"lsh {new_name} {lhs} {rhs}")
        else:
            raise NotImplementedError()
        self.name_map[new_name] = new_name
        return new_name

    def visit_UnaryOp(self, node): 
        value = self.process_operand(node.operand)
        new_name = self.fresh_name()
        if isinstance(node.op, ast.USub):
            self.ir.append(f"neg {new_name} {value}")
        else:
            raise NotImplementedError()
        return new_name

    def visit_Call(self, node):
        func_name = self.visit(node.func)
        if func_name == "smid":
            new_name = self.fresh_name()
            self.ir.append(f"smid {new_name}")
            self.name_map[new_name] = new_name
            return new_name
        elif func_name == "time":
            local = False
            for keyword in node.keywords:
                if keyword.arg == "local":
                    local = keyword.value.value
            new_name = self.fresh_name()
            self.name_map[new_name] = new_name
            if local:
                self.ir.append(f"clock {new_name}") 
            else:
                self.ir.append(f"time {new_name}")
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
                        regs.append(self.name_map[keyword.value.id])
                    elif isinstance(keyword.value, (ast.Tuple, ast.List)):
                        for reg in keyword.value.elts:
                            regs.append(self.name_map[reg.id])
            if len(regs) == 0 and len(node.args) > 0:
                if isinstance(node.args[0], (ast.Tuple, ast.List)):
                    for reg in node.args[0].elts:
                        regs.append(self.name_map[self.visit(reg)])
                else:
                    regs.append(self.name_map[self.visit(node.args[0])])
            for reg in regs:
                self.ir.append(f"{save_inst} {reg}")
        else:
            raise NotImplementedError()
    
    def visit_Name(self, node):
        return node.id

    def visit_Attribute(self, node):
        if node.value.id == self.nlname:
            return node.attr
        else:
            raise ValueError(f"can only refer to neutrino.language semantic but got {node.value.id}")

    def visit_Constant(self, node):
        return node

    def generic_visit(self, node):
        if type(node) not in allowed_nodes:
            raise NotImplementedError(f"{type(node).__name__} (lineno: {node.lineno})")
        super().generic_visit(node)

# A Simple Test Case
if __name__ == "__main__":
    code = """
import neutrino.language as nl
gstart = nl.time(local=False) # use global timer
gend = nl.time(local=False)
elapsed = gend - gstart 
nl.save(gstart, dtype=nl.u64)
nl.save((elapsed, nl.smid()), dtype=nl.u32)"""

    try:
        tree = ast.parse(code)
        visitor = RestrictedVisitor(regs=["gstart", "gend", "elapsed"])
        visitor.visit(tree)
        print(visitor.ir)
        print(visitor.name_map)
    except ValueError as e:
        raise
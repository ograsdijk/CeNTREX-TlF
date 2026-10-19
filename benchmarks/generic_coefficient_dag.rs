// Appended to an isolated snapshot of eval.rs; all nodes are prepared data.
#[derive(Clone)]
struct DagNode {
    instruction: Instruction,
    args: Vec<usize>,
}
pub struct CoefficientDag {
    nodes: Vec<DagNode>,
    roots: Vec<usize>,
    values: Vec<Complex64>,
    pub original_ops: usize,
}
impl CoefficientDag {
    pub fn new(expressions: impl Iterator<Item = CompiledExpression>) -> Self {
        let mut nodes = Vec::new();
        let mut roots = Vec::new();
        let mut map = std::collections::HashMap::new();
        let mut original_ops = 0;
        for expression in expressions {
            assert!(expression.scalar_only);
            let mut stack: Vec<usize> = Vec::new();
            for ins in expression.instructions {
                original_ops += 1;
                let argc = match ins.op {
                    InstructionOp::Const | InstructionOp::Slot | InstructionOp::Time => 0,
                    InstructionOp::Neg | InstructionOp::Conj => 1,
                    InstructionOp::BuiltinFunc | InstructionOp::HelperFunc => ins.argc,
                    InstructionOp::Temp => {
                        panic!("coefficient temps unsupported in this prototype")
                    }
                    _ => 2,
                };
                assert!(argc <= 16);
                let args: Vec<usize> = stack.drain(stack.len() - argc..).collect();
                let key = format!("{:?}/{:?}", ins, args);
                let idx = *map.entry(key).or_insert_with(|| {
                    let idx = nodes.len();
                    nodes.push(DagNode {
                        instruction: ins,
                        args,
                    });
                    idx
                });
                stack.push(idx);
            }
            assert_eq!(stack.len(), 1);
            roots.push(stack[0]);
        }
        let values = vec![Complex64::ZERO; nodes.len()];
        Self {
            nodes,
            roots,
            values,
            original_ops,
        }
    }
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    pub fn evaluate(
        &mut self,
        slots: &[RuntimeValue],
        t: f64,
        out: &mut Vec<Complex64>,
    ) -> Result<(), String> {
        for (idx, node) in self.nodes.iter().enumerate() {
            let ins = &node.instruction;
            let mut args = [Complex64::ZERO; 16];
            for (a, &src) in node.args.iter().enumerate() {
                args[a] = self.values[src];
            }
            let a = args[0];
            let b = args[1];
            self.values[idx] = match ins.op {
                InstructionOp::Const => ins.value,
                InstructionOp::Slot => as_scalar(&slots[ins.index])?,
                InstructionOp::Time => Complex64::new(t, 0.),
                InstructionOp::Add => a + b,
                InstructionOp::Sub => a - b,
                InstructionOp::Mul => a * b,
                InstructionOp::Div => a / b,
                InstructionOp::Pow => a.powc(b),
                InstructionOp::Neg => -a,
                InstructionOp::Conj => a.conj(),
                InstructionOp::BuiltinFunc => apply_builtin_scalar(ins.function, a)?,
                InstructionOp::HelperFunc => apply_helper_scalar(ins.function, &args[..ins.argc])?,
                _ => return Err("unsupported scalar DAG opcode".into()),
            };
        }
        out.clear();
        out.extend(self.roots.iter().map(|&r| self.values[r]));
        Ok(())
    }
}

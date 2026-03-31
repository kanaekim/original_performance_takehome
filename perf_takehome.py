"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_packed(self, slots_list):
        instr = {}
        for engine, slot in slots_list:
            if engine not in instr:
                instr[engine] = []
            instr[engine].append(slot)
        if instr:
            self.instrs.append(instr)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch space at {self.scratch_ptr}"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        n_groups = batch_size // VLEN  # 32
        PIPE_SLOTS = 5
        GROUP_SPACING = 4
        LIFECYCLE = 19  # cycles per group
        ROUND_SPACING = (n_groups - 1) * GROUP_SPACING + GROUP_SPACING  # = 128

        # Compute header values at build time (no memory loads needed!)
        HEADER_SIZE = 7
        fvp_value = HEADER_SIZE
        iip_value = HEADER_SIZE + n_nodes
        ivp_value = HEADER_SIZE + n_nodes + batch_size

        # ============================================================
        # ALLOCATE SCRATCH
        # ============================================================
        s_load_addr = self.alloc_scratch("s_la")
        s_load_addr2 = self.alloc_scratch("s_la2")

        # Scalar constants
        zero_addr = self.alloc_scratch("zero")
        self.const_map[0] = zero_addr
        vlen_addr = self.alloc_scratch("vlen")
        self.const_map[VLEN] = vlen_addr
        one_addr = self.alloc_scratch("one")
        self.const_map[1] = one_addr
        two_addr = self.alloc_scratch("two")
        self.const_map[2] = two_addr
        two_vlen_addr = self.alloc_scratch("two_vlen")
        self.const_map[2 * VLEN] = two_vlen_addr
        fvp_addr = self.alloc_scratch("fvp")
        nnodes_addr = self.alloc_scratch("n_nodes")
        ivp_addr = self.alloc_scratch("ivp")

        zero = zero_addr
        one = one_addr
        two = two_addr
        vlen_const = vlen_addr
        two_vlen = two_vlen_addr

        # Build list of add_imm operations for all scalars during vload phase
        addimm_ops = [
            (one_addr, 1),
            (two_addr, 2),
            (two_vlen_addr, 2 * VLEN),
            (fvp_addr, fvp_value),
            (nnodes_addr, n_nodes),
            (ivp_addr, ivp_value),
        ]

        # Hash scalar constants
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                multiplier = (1 + (1 << val3)) % (2**32)
                if multiplier not in self.const_map:
                    addr = self.alloc_scratch(f"hm_{hi}")
                    self.const_map[multiplier] = addr
                    addimm_ops.append((addr, multiplier))
                if val1 not in self.const_map:
                    addr = self.alloc_scratch(f"ha_{hi}")
                    self.const_map[val1] = addr
                    addimm_ops.append((addr, val1))
            else:
                if val1 not in self.const_map:
                    addr = self.alloc_scratch(f"hc1_{hi}")
                    self.const_map[val1] = addr
                    addimm_ops.append((addr, val1))
                if val3 not in self.const_map:
                    addr = self.alloc_scratch(f"hc3_{hi}")
                    self.const_map[val3] = addr
                    addimm_ops.append((addr, val3))

        # Vector constants
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_fvp = self.alloc_scratch("v_fvp", VLEN)
        v_nnodes = self.alloc_scratch("v_nnodes", VLEN)

        v_hash_ma_mul = [None] * 6
        v_hash_ma_add = [None] * 6
        v_hash_c1 = [None] * 6
        v_hash_c3 = [None] * 6

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                multiplier = (1 + (1 << val3)) % (2**32)
                v_hash_ma_mul[hi] = self.alloc_scratch(f"v_ma_mul_{hi}", VLEN)
                v_hash_ma_add[hi] = self.alloc_scratch(f"v_ma_add_{hi}", VLEN)
            else:
                v_hash_c1[hi] = self.alloc_scratch(f"v_hc1_{hi}", VLEN)
                v_hash_c3[hi] = self.alloc_scratch(f"v_hc3_{hi}", VLEN)

        # Per-group scratch (indices and values, persistent across rounds)
        g_idx = [self.alloc_scratch(f"gi_{g}", VLEN) for g in range(n_groups)]
        g_val = [self.alloc_scratch(f"gv_{g}", VLEN) for g in range(n_groups)]

        # Pipeline working registers (5 slots recycled every GROUP_SPACING)
        pipe_na = [self.alloc_scratch(f"pna{i}", VLEN) for i in range(PIPE_SLOTS)]
        pipe_nv = [self.alloc_scratch(f"pnv{i}", VLEN) for i in range(PIPE_SLOTS)]
        pipe_t1 = [self.alloc_scratch(f"pt1{i}", VLEN) for i in range(PIPE_SLOTS)]
        pipe_t2 = [self.alloc_scratch(f"pt2{i}", VLEN) for i in range(PIPE_SLOTS)]

        # ============================================================
        # INIT: 1 setup cycle + 32 vload cycles = 33 cycles
        # Scratch starts at zero - no need to explicitly load zero!
        # All constants loaded via add_imm (flow engine) during vloads.
        # All broadcasts done via valu during vloads.
        # ============================================================
        # Cycle 0: const(la, iip_value), const(vlen, VLEN), add_imm(la2, zero, ivp_value)
        # zero_addr is never written, stays 0 from scratch initialization
        self.add_packed([
            ("load", ("const", s_load_addr, iip_value)),
            ("load", ("const", vlen_const, VLEN)),
            ("flow", ("add_imm", s_load_addr2, zero, ivp_value)),
        ])

        # Build broadcast list with source address tracking
        all_broadcasts = []
        all_broadcasts.append(("valu", ("vbroadcast", v_one, one_addr)))
        all_broadcasts.append(("valu", ("vbroadcast", v_two, two_addr)))
        all_broadcasts.append(("valu", ("vbroadcast", v_fvp, fvp_addr)))
        all_broadcasts.append(("valu", ("vbroadcast", v_nnodes, nnodes_addr)))
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                multiplier = (1 + (1 << val3)) % (2**32)
                all_broadcasts.append(("valu", ("vbroadcast", v_hash_ma_mul[hi], self.const_map[multiplier])))
                all_broadcasts.append(("valu", ("vbroadcast", v_hash_ma_add[hi], self.const_map[val1])))
            else:
                all_broadcasts.append(("valu", ("vbroadcast", v_hash_c1[hi], self.const_map[val1])))
                all_broadcasts.append(("valu", ("vbroadcast", v_hash_c3[hi], self.const_map[val3])))

        # Track when each scalar becomes available for broadcast
        addr_available = {zero_addr: 0}  # zero available from start
        addimm_idx = 0
        pending_broadcasts = list(all_broadcasts)

        for g in range(n_groups):
            ops = [
                ("load", ("vload", g_idx[g], s_load_addr)),
                ("load", ("vload", g_val[g], s_load_addr2)),
            ]
            if g < n_groups - 1:
                ops.append(("alu", ("+", s_load_addr, s_load_addr, vlen_const)))
                ops.append(("alu", ("+", s_load_addr2, s_load_addr2, vlen_const)))

            # Add_imm for next constant (1 per cycle, flow engine)
            if addimm_idx < len(addimm_ops):
                dest, val = addimm_ops[addimm_idx]
                ops.append(("flow", ("add_imm", dest, zero, val)))
                addr_available[dest] = g + 1  # available next cycle
                addimm_idx += 1

            # Broadcasts (check if source scalar is available)
            valu_count = 0
            still_pending = []
            for bc_op in pending_broadcasts:
                if valu_count >= 6:
                    still_pending.append(bc_op)
                    continue
                src_addr = bc_op[1][2]  # vbroadcast(dest, src) -> src
                if src_addr in addr_available and g >= addr_available[src_addr]:
                    ops.append(bc_op)
                    valu_count += 1
                else:
                    still_pending.append(bc_op)
            pending_broadcasts = still_pending

            # Pack pause into last vload cycle if flow engine is free
            pause_packed = False
            if g == n_groups - 1 and not any(o[0] == "flow" for o in ops):
                ops.append(("flow", ("pause",)))
                pause_packed = True

            self.add_packed(ops)

        assert addimm_idx == len(addimm_ops), f"Not all consts loaded: {addimm_idx}/{len(addimm_ops)}"
        assert len(pending_broadcasts) == 0, f"Not all broadcasts done: {len(pending_broadcasts)} remaining"

        if not pause_packed:
            self.add("flow", ("pause",))

        # ============================================================
        # BUILD UNIFIED PIPELINE SCHEDULE (all rounds, inter-round overlap)
        # ============================================================
        total_pipeline_cycles = (rounds - 1) * ROUND_SPACING + (n_groups - 1) * GROUP_SPACING + LIFECYCLE
        schedule = [[] for _ in range(total_pipeline_cycles)]

        for r in range(rounds):
            round_offset = r * ROUND_SPACING
            is_last_round = (r == rounds - 1)
            for g in range(n_groups):
                t = round_offset + g * GROUP_SPACING
                s = (r * n_groups + g) % PIPE_SLOTS
                na = pipe_na[s]
                nv = pipe_nv[s]
                t1 = pipe_t1[s]
                t2 = pipe_t2[s]
                idx = g_idx[g]
                val = g_val[g]

                # T+0: nodeaddr = idx + forest_values_p
                schedule[t].append(("valu", ("+", na, idx, v_fvp)))

                # T+1..T+4: gather (8 load_offsets, 2 per cycle)
                for off in range(0, VLEN, 2):
                    cy = t + 1 + off // 2
                    schedule[cy].append(("load", ("load_offset", nv, na, off)))
                    schedule[cy].append(("load", ("load_offset", nv, na, off + 1)))

                # T+5: XOR val ^= node_val
                schedule[t + 5].append(("valu", ("^", val, val, nv)))

                # T+6..T+14: Hash (9 cycles: 3*MA + 3*2-cycle)
                cycle = t + 6
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    if v_hash_ma_mul[hi] is not None:
                        schedule[cycle].append(("valu", ("multiply_add", val, val, v_hash_ma_mul[hi], v_hash_ma_add[hi])))
                        cycle += 1
                    else:
                        schedule[cycle].append(("valu", (op1, t1, val, v_hash_c1[hi])))
                        schedule[cycle].append(("valu", (op3, t2, val, v_hash_c3[hi])))
                        cycle += 1
                        schedule[cycle].append(("valu", (op2, val, t1, t2)))
                        cycle += 1

                # Skip index update on last round (only values matter for output)
                if not is_last_round:
                    # T+14: MA(na, idx, 2, 1) = 2*idx+1 (overlaps with last hash combine)
                    schedule[cycle - 1].append(("valu", ("multiply_add", na, idx, v_two, v_one)))
                    # T+15: bit = val & 1
                    schedule[cycle].append(("valu", ("&", nv, val, v_one)))
                    cycle += 1
                    # T+16: idx = (2*idx+1) + bit
                    schedule[cycle].append(("valu", ("+", idx, na, nv)))
                    cycle += 1
                    # T+17: cond = idx < n_nodes
                    schedule[cycle].append(("valu", ("<", t1, idx, v_nnodes)))
                    cycle += 1
                    # T+18: idx = idx * cond (wraps to 0 if past tree)
                    schedule[cycle].append(("valu", ("*", idx, idx, t1)))

        # ============================================================
        # EMBED STORES: last round's values stored during pipeline tail
        # ============================================================
        last_round = rounds - 1
        addr_setup_cycle = last_round * ROUND_SPACING + 1 * GROUP_SPACING + 14
        schedule[addr_setup_cycle].append(("alu", ("+", s_load_addr, ivp_addr, zero)))
        schedule[addr_setup_cycle].append(("alu", ("+", s_load_addr2, ivp_addr, vlen_const)))

        for gp in range(0, n_groups, 2):
            ready_cycle = last_round * ROUND_SPACING + (gp + 1) * GROUP_SPACING + 14
            store_cycle = ready_cycle + 1
            if store_cycle >= total_pipeline_cycles:
                break
            schedule[store_cycle].append(("store", ("vstore", s_load_addr, g_val[gp])))
            schedule[store_cycle].append(("store", ("vstore", s_load_addr2, g_val[gp + 1])))
            if gp + 2 < n_groups:
                schedule[store_cycle].append(("alu", ("+", s_load_addr, s_load_addr, two_vlen)))
                schedule[store_cycle].append(("alu", ("+", s_load_addr2, s_load_addr2, two_vlen)))

        # ============================================================
        # VERIFY & EMIT
        # ============================================================
        for cy, ops in enumerate(schedule):
            if not ops:
                continue
            counts = {}
            for engine, slot in ops:
                counts[engine] = counts.get(engine, 0) + 1
            for engine, count in counts.items():
                limit = SLOT_LIMITS.get(engine, 999)
                assert count <= limit, f"Cycle {cy}: {engine} has {count} ops (limit {limit}), ops: {[(e,s[0]) for e,s in ops]}"

        # Pack schedule[0] into the last init instruction (overlap nodeaddr with pause/vload)
        if schedule[0]:
            last_init = self.instrs[-1]
            for engine, slot in schedule[0]:
                if engine not in last_init:
                    last_init[engine] = []
                last_init[engine].append(slot)
        for ops in schedule[1:]:
            if ops:
                self.add_packed(ops)


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


if __name__ == "__main__":
    unittest.main()

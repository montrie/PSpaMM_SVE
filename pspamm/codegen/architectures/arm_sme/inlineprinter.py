from typing import List
from pspamm.codegen.ast import *
from pspamm.codegen.visitor import Visitor
from pspamm.codegen.operands import *
from pspamm.codegen.precision import *
from pspamm.codegen.architectures.arm_sme.operands import z
# TODO: p_string should not end in ', ', that should be part of the final asm string a visit function assembles

class InlinePrinter(Visitor):
    show_comments = True
    indent = "  "
    depth = 0
    lmargin = 0
    rmargin = 60
    vpadding = False
    output = None
    stack = None

    def __init__(self, precision: Precision):
        self.output = []
        self.stack = []
        self.precision = precision

    def show(self):
        print("\n".join(self.output))

    def addLine(self, stmt: str, comment: str):
        line = " " * self.lmargin + self.indent * self.depth

        if stmt is not None and comment is not None and self.show_comments:
            stmt = '"' + stmt + '\\r\\n"'
            line += stmt.ljust(self.rmargin) + "// " + comment

        elif stmt is not None:
            line += '"' + stmt + '\\r\\n"'

        elif stmt is None and comment is not None:
            line += "// " + comment

        self.output.append(line)

    def visitFma(self, stmt: FmaStmt):
        b = stmt.bcast_src.ugly
        m = stmt.mult_src.ugly
        a = stmt.add_dest.ugly_mem_vector_group_fmla
#        if "z0" in m:
#            a = a.replace("#0", "#1")
#        if "z4" in m:
#            a = a.replace("#0", "#3")
#        if "z8" in m:
#            a = a.replace("#0", "#")
#        if "z12" in m:
#            a = a.replace("#0", "#")
        group_num = int(a[-2])
        zn_index = int(m[1:-2])  # should give us the number of the SVE vector
        zn_end_index = zn_index + group_num - 1
        zn_end_reg = z(zn_end_index, stmt.mult_src.ugly_precision)
        zn_end = zn_end_reg.ugly
        mult_str = "{{{}-{}}}".format(m, zn_end)
        p = self.p_string(stmt.pred)

        s = "fmla {}, {}, {}".format(a, mult_str, b)

        #TODO: QEMU DOES NOT IMPLEMENT THE SME FMLA BC ITS A SME2 FEATURE
        # s = "FMLA NOT IN QEMU"

        self.addLine(s, stmt.comment)

    def visitFmopa(self, stmt: FmopaStmt):
        # floating-point outer product and accumulate:
        # FMOPA <ZAda>.D, <Pn>/M, <Pm>/M, <Zn>.D, <Zm>.D
        # if we have to reuse fma: put 2 predicates into stmt.pred, add_dest is ZA tile, bcast_src is B_reg, mult_src is A_reg
        za = stmt.za.ugly_register
        mult = stmt.mult_src.ugly
        mult2 = stmt.mult_src2.ugly
        p = self.p_string(stmt.pred)
        p2 = self.p_string(stmt.pred2)

        s = "fmopa {}, {}{}{}, {}".format(za, p, p2, mult, mult2)

        self.addLine(s, stmt.comment)


    def visitMul(self, stmt: MulStmt):
        b = stmt.src.ugly
        m = stmt.mult_src.ugly
        a = stmt.dest.ugly
        p = self.p_string(stmt.pred)

        if a != b:
            s1 = "mov {}, {}{}".format(a, p, b)
            self.addLine(s1, "move tile slice {} into {}".format(stmt.src.ugly_offset, a))
            # s2 = "movprfx {}, {}".format(a.split(".")[0], b.split(".")[0])
            # self.addLine(s2, "move {} into {}".format(b, a))
            b = a

        # TODO: alpha*A*B will remain as an fmul instruction
        #       beta*C: there seems to be no multiplication of ZA slices, we need to mov/ld a C vector into an SVE register, then multiply it with beta and move the result back into ZA
        s = "fmul {}, {}{}, {}".format(a, p, b, m)
        self.addLine(s, stmt.comment)

    def visitBcst(self, stmt: BcstStmt):
        # Used to broadcast a scalar register into a vector register
        b = stmt.bcast_src.ugly
        a = stmt.dest.ugly
        # make sure the src register is a W register when using single precision
        if self.precision == Precision.SINGLE:
            b = "w" + b[1:]
        s = "dup {}, {}".format(a, b)
        self.addLine(s, stmt.comment)

    def visitAdd(self, stmt: AddStmt):
        if isinstance(stmt.src, Constant) and (stmt.src.value > 4095 or stmt.src.value < -4095):
            # This condition is probably related to immediate values being restricted to 12 bits for add instructions
            # https://developer.arm.com/documentation/dui0802/a/A64-General-Instructions/ADD--immediate-
            # https://developer.arm.com/documentation/ddi0596/2020-12/Base-Instructions/ADD--immediate---Add--immediate--
            if (stmt.src.value >> 16) & 0xFFFF > 0 and stmt.src.value < 0:
                s = "mov x11, #-1"
                s1 = "movk x11, #{}".format((stmt.src.value) & 0xFFFF)
                val = ((stmt.src.value >> 16) & 0xFFFF)
                s2 = "movk x11, #{}, lsl #16".format(val)

                self.addLine(s, "")
                self.addLine(s1, "load lower 16 bit of immediate that requires more than 16 bit")
                self.addLine(s2, "load upper 16 bit of immediate that requires more than 16 bit")
            elif (stmt.src.value >> 16) != 0:
                s1 = "mov x11, #{}".format((stmt.src.value) & 0xFFFF)
                val = ((stmt.src.value >> 16) & 0xFFFF)
                s2 = "movk x11, #{}, lsl #16".format(val)
                self.addLine(s1, "load lower 16 bit of immediate that requires more than 16 bit")
                self.addLine(s2, "load upper 16 bit of immediate that requires more than 16 bit")
            else:
                s = "mov x11, {}".format(stmt.src.ugly)
                self.addLine(s, "load lower 16 bit of immediate ")

            if stmt.dest.ugly != "x11":
                s = "add {}, {}, x11".format(stmt.dest.ugly, stmt.dest.ugly)
                self.addLine(s, stmt.comment)
            if stmt.additional is not None:
                s = "add {}, {}, {}".format(stmt.dest.ugly, stmt.dest.ugly, stmt.additional.ugly)
                self.addLine(s, stmt.comment)
        else:
            # if stmt.src is a Constant but outside of the above range of value < -4095 or value > 4095
            # we can simply add the Constant to a register
            if stmt.additional is not None:
                s = "add {}, {}, {}".format(stmt.dest.ugly, stmt.additional.ugly, stmt.src.ugly)
            elif stmt.dest.ugly.startswith("z"):
                # TODO: is there a better way to determine whether dest is a ZA tile slice?
                p = "p7/m, "# self.p_string(stmt.pred)
                dest = stmt.dest.ugly
                src = stmt.src.ugly
                s = "fadd {}, {}{}, {}".format(dest, p, dest, src)
            else:
                s = "add {}, {}, {}".format(stmt.dest.ugly, stmt.dest.ugly, stmt.src.ugly)
            self.addLine(s, stmt.comment)

    def visitLabel(self, stmt: LabelStmt):
        s = "{}:".format(stmt.label.ugly)
        self.addLine(s, stmt.comment)

    def visitCmp(self, stmt: CmpStmt):
        s = "cmp {}, {}".format(stmt.rhs.ugly, stmt.lhs.ugly)
        self.addLine(s, stmt.comment)

    def visitJump(self, stmt: JumpStmt):
        s = "b.lo {}".format(stmt.destination.ugly)
        self.addLine(s, stmt.comment)

    def visitMov(self, stmt: MovStmt):
        if isinstance(stmt.src, Label):
            src_str = ("#" + stmt.src.ugly).split("_")[0]
        else: 
            src_str = stmt.src.ugly
        if stmt.typ == AsmType.f64x8:
            p = self.p_string(stmt.pred)
            if not isinstance(stmt.src, Label):
                # test = stmt.src.typeinfo
                # test2 = stmt.dest.typeinfo
                # testtype = AsmType.za
                # testtype2 = AsmType.za
                # testresult = testtype == testtype2
                # print(test == testtype)
                # print(testresult)
                # test3 = stmt.src.typeinfo.name
                # testresult3 = test3.startswith('za')
                # TODO: why does the line below return false??
                # if stmt.src.typeinfo == AsmType.za:
                # if stmt.src.typeinfo.name.startswith('za'):
                #     # self.addLine("src is za", "DEBUG")
                #     src_str = stmt.dest.ugly
                #     dest_str = stmt.dest.ugly
                # if stmt.dest.typeinfo.name.startswith('za'):
                #     # self.addLine("dest is za", "DEBUG")
                #     src_str = stmt.src.ugly
                #     dest_str = stmt.dest.ugly
                # predicate is only used when we move data into/out of the ZA register
                # TODO: use MOVA instead?
                if stmt.comment == "Move C to matrix register":
                    mem_acc_str, base, offs, abs_offs = self.za_abs_offs(stmt.dest)
                    tile = self.get_fma_tile_slice(stmt.dest, base, offs, abs_offs)
                    dest_str = "za{}h.{}[{}]".format(tile, stmt.dest.ugly_precision, mem_acc_str)
                else:
                    dest_str = stmt.dest.ugly
                if stmt.comment == "Move C back to contiguous tile":
                    # adjust src_str after the fmla instr. in matmul
                    mem_acc_str, base, offs, abs_offs = self.za_abs_offs(stmt.src)
                    tile = self.get_fma_tile_slice(stmt.src, base, offs, abs_offs)
                    src_str = "za{}h.{}[{}]".format(tile, stmt.dest.ugly_precision, mem_acc_str)
                # s = "mov {}, {}{}".format(stmt.dest.ugly, p, src_str)
                s = "mov {}, {}{}".format(dest_str, p, src_str)
            else:
                # s = "fmov {}, {}".format(stmt.dest.ugly, src_str)
                s = "zero {za}"
        else:
            if stmt.comment == "Setup base za register":
                # adjust src_str for the setup of the za base register in FMLA
                src_str = src_str.replace("x14", "x15")
            s = "mov {}, {}".format(stmt.dest.ugly, src_str)
        self.addLine(s, stmt.comment)

    def visitLoad(self, stmt: LoadStmt):
        if isinstance(stmt.src, Label):
            src_str = "#" + stmt.src.ugly
# TODO: ugly_offset and scalar_offs might be helpful to include the

        elif stmt.src.ugly_offset != "0" and stmt.scalar_offs:
            self.addLine("mov {}, #{}".format(stmt.add_reg.ugly, stmt.src.ugly_offset), "move immediate offset into {}".format(stmt.add_reg.ugly))
            # TODO: adapt ugly_lsl_shift to account for possible single precision instead of double precision
            src_str = "[{}, {}, LSL #{}]".format(stmt.src.ugly_base, stmt.add_reg.ugly, stmt.dest.ugly_lsl_shift)
        else:
            src_str = stmt.src.ugly if not stmt.is_B else stmt.src.ugly_no_vl_scaling

        p = self.p_string(stmt.pred)
        # TODO: this can be done better
        prec = "d" if stmt.dest.ugly_precision == "d" else "w"
        is_B = "r" if stmt.is_B else ""

        if stmt.typ == AsmType.i64:
            s = "add {}, {}, {}".format(stmt.dest.ugly, stmt.dest.ugly, src_str)
# TODO: stmt.dest is prob. ZA tile -> we can only load a slice so do we loop over the X tile slices here?
#       we don't have access to the size of C here so the looping (+ adjustment of access base reg) should happen in generator.py
# TODO: maybe we can assign C_reg the X amount of different ZA slices that exist
# TODO: do we still load only one element of B and broadcast it across a whole SVE vector?
# TODO: maybe use str if dest.ugly_offset and za.ugly_offset are equal?

        elif stmt.typ == AsmType.f64x8 and stmt.aligned:
            if stmt.za != None:
                # if stmt.src.ugly_offset == stmt.za.ugly_offset:
                #     s = "ldr {}, {}".format(stmt.za.ugly, src_str)
                # else: 
                #     s = "ld1{}{} {}, {}{}".format(is_B, prec, stmt.dest.ugly, p, src_str)
                if stmt.src.ugly_offset == "0":
                        src_str = "[{}]".format(stmt.src.ugly_base)
                s = "ld1{}{} {{{}}}, {}{}".format(is_B, prec, stmt.dest.ugly, p, src_str)
            else:
                # if stmt.is_B:
                #     s = "ld1r{} {}, {}{}".format(prec, stmt.dest.ugly, p, src_str)
                # else:
                s = "ld1{}{} {}, {}{}".format(is_B, prec, stmt.dest.ugly, p, src_str)
        else:
            raise NotImplementedError()
        self.addLine(s, stmt.comment)

    def visitStore(self, stmt: StoreStmt):
        if isinstance(stmt.src, Label):
            src_str = "#" + stmt.src.ugly
        elif stmt.dest.ugly_offset != "0" and stmt.scalar_offs:
            self.addLine("mov {}, #{}".format(stmt.add_reg.ugly, stmt.dest.ugly_offset),
                         "move immediate offset into {}".format(stmt.add_reg.ugly))
            # TODO: adapt ugly_lsl_shift to account for possible single precision instead of double precision
            dest_str = "[{}, {}, LSL #{}]".format(stmt.dest.ugly_base, stmt.add_reg.ugly, stmt.src.ugly_lsl_shift)
        else:
            dest_str = stmt.dest.ugly

        p = self.p_string(stmt.pred)
        prec = "d" if stmt.src.ugly_precision == "d" else "w"

        if stmt.typ == AsmType.i64:
            s = "add {}, {}, {}".format(stmt.dest.ugly, stmt.dest.ugly, dest_str)
# TODO: same concerns as for the load instruction
# TODO: there IS a store instruction that directly stores a row of ZA to memory, we DON'T need to MOVA the ZA slice
#       into a Z register and then store that register
#       src is the SVE register, dest is the memory we store to, za is the ZA tile slice
# TODO: maybe use str if dest.ugly_offset and za.ugly_offset are equal?
        elif stmt.typ == AsmType.f64x8 and stmt.aligned:
            if stmt.za != None:
                if stmt.scalar_offs:
                    if stmt.dest.ugly_offset == "0":
                        dest_str = "[{}]".format(stmt.dest.ugly_base)
                    s = "st1{} {{{}}}, {}{}".format(prec, stmt.src.ugly, p, dest_str)
                elif stmt.dest.ugly_offset == stmt.za.ugly_offset:
                    s = "str {}, {}".format(stmt.za.ugly_slice, dest_str)
                else:
                    s = "st1{} {}, {}{}".format(prec, stmt.src.ugly, p, dest_str)
            else:
                s = "st1{} {}, {}{}".format(prec, stmt.src.ugly, p, dest_str)
        else:
            raise NotImplementedError()
        self.addLine(s, stmt.comment)

    def visitPrefetch(self, stmt: PrefetchStmt):
        # https://stackoverflow.com/questions/37070/what-is-the-meaning-of-non-temporal-memory-accesses-in-x86#:~:text=Data%20referenced%20by%20a%20program,%2C%20is%20often%20non%2Dtemporal.
        cache_level = "L1"  # specify cache level to which we prefetch
        temporality = "KEEP"  # could use "STRM" for non-temporal prefetching if needed
        xn = stmt.dest.ugly_base
        offset = stmt.dest.ugly_offset
        src_string = "[{}, {}, MUL VL]".format(xn, offset)
        p = self.p_string(stmt.pred)
        prec = "d" if stmt.precision == Precision.DOUBLE else "w"
        s = "prf{} P{}{}{}, {}{}".format(prec, stmt.access_type, cache_level, temporality, p.split('/')[0], src_string)
        self.addLine(s, "prefetch from memory")

    def visitBlock(self, block: Block):
        self.stack.append(block)
        self.depth += 1
        if self.show_comments:
            self.addLine(None, block.comment)
        for stmt in block.contents:
            stmt.accept(self)
        self.depth -= 1
        self.stack.pop()

    def p_string(self, predicate: Register):
        # returns "pk{/z or /m}, " or an empty string "" with contents in {} being optional
        # at this point the contents are already generated, we simply turn them into a string
        return predicate.value + ", " if predicate is not None else ""

    def za_abs_offs(self, za_slice: Register):
        base = int(za_slice.ugly_base[1:])
        offset = int(za_slice.ugly_offset)
        abs_offs = (base - 12) * za_slice.ugly_max_tile_slice_offset + offset
        # TODO: for doubles, the fmla instruction skips a row to store the result of the VGx4 result
        # maybe for singles it skips none/ another number of rows?
        # ugly_max_tile_slice_offset is 2 for doubles and 4 for singles, idk if the 4 leads to correct 
        # calculations in the case of single precision
        za_base = str(abs_offs % (4 // za_slice.ugly_max_tile_slice_offset * za_slice.ugly_max_tile_slice_offset) + 12)
#        za_offset = str(abs_offs // (4 // za_slice.ugly_max_tile_slice_offset * za_slice.ugly_max_tile_slice_offset)) # str(abs_offs % za_slice.ugly_max_tile_slice_offset)
        # alle 8 offsets landen wir wieder in der selben tile aber ugly_max_tile_slice_offset zeilen weiter
        # after 8 vector processed by fma, we go back to a tile that we already visited, the offset calculation below is 
        # therefore increased by 1 compared to the last time we visited the same tile -> this should check out with how ZA is implemented in the cpu
        za_offset = str(abs_offs // 8) # * za_slice.ugly_max_tile_slice_offset)
        return "w{}, #{}".format(za_base, za_offset), za_base, za_offset, abs_offs

    def get_fma_tile_slice(self, za_slice: Register, base: str, offs: str, abs_offs: int):
        prec_to_tile = {Precision.DOUBLE.value: [1, 3, 5, 7],
                        Precision.SINGLE.value: [1, 3, 1, 3],
                        # Precision.HALFWORD.value: [1, 1] <- in case we add half-precision functionality
                       }
        tile = prec_to_tile[self.precision.value][abs_offs // 4] # 4 because we use 4 source vectors during fma
        return tile
        

def render(s: AsmStmt):
    p = InlinePrinter()
    s.accept(p)
    return "\n".join(p.output)

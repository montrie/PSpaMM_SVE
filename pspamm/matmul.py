from typing import Tuple

from pspamm.codegen.ast import *
from pspamm.codegen.sugar import *
from pspamm.codegen.forms import *
from pspamm.codegen.precision import *

import pspamm.scripts.old_arm
import pspamm.scripts.max_bn_knl
import pspamm.scripts.max_bn_hsw
import pspamm.scripts.max_arm_sve

from pspamm.cursors import *

import pspamm.architecture
import numpy


def decompose_pattern(k, n, pattern:Matrix[bool], bk:int, bn:int, is_sme:bool) -> Tuple[Matrix[int], List[Matrix[bool]]]:
    Bk,Bn = k//bk, n//bn
    patterns = []
    x = 0

    n_overhead = n % bn
    k_overhead = k % bk

    if n_overhead > 0:
        Bn += 1
    if k_overhead > 0:
        Bk += 1

    blocks = Matrix.full(Bk,Bn,-1)# if not is_sme else Matrix.full(Bn,Bk,-1)

    if is_sme:
        # TODO: for now we just copy the else part from below, we don't have overheads using SME
        for Bni in range(Bn):
            for Bki in range(Bk):
                block = pattern[(Bki*bk):((Bki+1)*bk), (Bni*bn):((Bni+1)*bn)]
                # block = pattern[(Bni*bn):((Bni+1)*bn), (Bki*bk):((Bki+1)*bk)]
                blocks[Bki,Bni] = x
                x += 1
                patterns.append(block)

    else:
        for Bni in range(Bn):
            for Bki in range(Bk):
                if Bni + 1 == Bn and n_overhead > 0 and Bki + 1 == Bk and k_overhead > 0:
                    block = pattern[(Bki*bk):((Bki+1)*bk+k_overhead), (Bni*bn):((Bni)*bn+n_overhead)]
                elif Bni + 1 == Bn and n_overhead > 0:
                    block = pattern[(Bki*bk):((Bki+1)*bk), (Bni*bn):((Bni)*bn+n_overhead)]
                elif Bki + 1 == Bk and k_overhead > 0:
                    block = pattern[(Bki*bk):((Bki+1)*bk+k_overhead), (Bni*bn):((Bni+1)*bn)]
                else:
                    block = pattern[(Bki*bk):((Bki+1)*bk), (Bni*bn):((Bni+1)*bn)]
                
                blocks[Bki,Bni] = x
                x += 1
                patterns.append(block)

    mtx_overhead = [0] * n

    for i in range(n):
        for j in range(k, pattern.rows):
            if pattern[j, i]:
                mtx_overhead[i] += 1

    return blocks, patterns, mtx_overhead

class MatMul:
    def __init__(self,
                 m: int, 
                 n: int, 
                 k: int, 
                 lda: int, 
                 ldb: int, 
                 ldc: int,
                 alpha: str,
                 beta: str,
                 mtx_filename: str,
                 mtx_format: str = 'any',
                 output_funcname: str = None,
                 output_filename: str = None,
                 output_overwrite: bool = False,
                 bm: int = None, 
                 bn: int = None, 
                 bk: int = None,
                 arch: str = 'knl',
                 precision: str = 'd',
                 prefetching: str = None,
                 **kwargs  # Accept and ignore args which don't belong
                 ) -> None:

        self.m = m
        self.n = n
        self.k = k

        self.lda = lda
        self.ldb = ldb
        self.ldc = ldc

        try:
          self.alpha = float(alpha)
        except:
          self.alpha = 'generic'
        try:
          self.beta = float(beta)
        except:
          self.beta = 'generic'

        if arch == 'skx':
          arch = 'knl'
        
        # hacky implementation of multi-register length for SVE and SME
        if arch.startswith('arm_'):
          if len(arch) == 7:
            v_len_regs = 4 # compatibility: arm_sve == arm_sve512, arm_sme == arm_sme512
          else:
            v_len_bits = int(arch[7:])
            assert v_len_bits % 128 == 0 and v_len_bits <= 2048
            v_len_regs = v_len_bits // 128
          arch = arch[:7]

        self.arch = arch
        assert precision.lower() in ['s', 'd']
        self.precision = Precision.DOUBLE if precision.lower() == 'd' else Precision.SINGLE

        pspamm.architecture.init()
        pspamm.architecture.arch = arch
        pspamm.architecture.Generator = pspamm.architecture.get_class("pspamm.codegen.architectures." + arch + ".generator").Generator
        pspamm.architecture.operands = pspamm.architecture.get_class("pspamm.codegen.architectures." + arch + ".operands")

        self.generator = pspamm.architecture.Generator(self.precision)

        # flags that determine if a matmul kernel uses sve/sme instructions, needed for sve predicates
        self.is_sve = arch.startswith("arm_")
        self.is_sme = arch == "arm_sme"
        # define which architectures need to use an explicit broadcast, necessary for alpha/beta values
        self.use_bcst = arch in ["arm", "arm_sve", "arm_sme", "hsw"]

        if self.is_sve:
          self.generator.v_len = v_len_regs

        self.v_size = self.generator.get_v_size()

        if bk == None:
            bk = 2 if arch == 'knl' else 1

        if bm == None or bn == None:
            if arch == 'knl':
                (self.bm, self.bn) = pspamm.scripts.max_bn_knl.getBlocksize(m, n, bk, self.v_size)
            elif arch == 'hsw':
                (self.bm, self.bn) = pspamm.scripts.max_bn_hsw.getBlocksize(m, n, bk, self.v_size)
            elif arch == 'arm':
                (self.bm, self.bn) = pspamm.scripts.old_arm.getBlocksize(m, n, bk, self.v_size)
            elif arch == 'arm_sve':
                (self.bm, self.bn) = pspamm.scripts.max_arm_sve.getBlocksize(m, n, bk, self.v_size)
            elif arch == 'arm_sme':
                (self.bm, self.bn, self.bk) = pspamm.scripts.max_arm_sme.getBlocksize(m, n, bk, self.v_size)
        else: 
            self.bm = bm
            self.bn = bn

        self.bk = bk

        self.prefetching = prefetching

        self.mtx_filename = mtx_filename
        self.mtx_format = mtx_format

        self.output_funcname = output_funcname
        self.output_filename = output_filename
        self.output_overwrite = output_overwrite

        if ldb == 0:
            pattern = Matrix.load(mtx_filename)
            if self.is_sve:
                self.generator.set_sparse()
        else:
            mtx = numpy.zeros((k, n))
            for i in range(k):
                for j in range(n):
                    mtx[i, j] = 1
            pattern = Matrix(mtx)

        blocks,patterns,mtx_overhead = decompose_pattern(self.k, self.n, pattern, self.bk, self.bn, self.is_sme)

        self.nnz = 0
        self.flop = 0

        if ldb == 0:
            for i in range(n):
                for j in range(k):
                    if pattern[j,i]:
                        self.nnz += 1
            self.flop = self.nnz * m * 2
            self.nnz += sum(mtx_overhead)
        else:
            self.nnz = ldb * self.n
            self.flop = m * n * k * 2

        #if prefetching is not None:
        #    prefetchReg = self.generator.init_prefetching(self.prefetching)
        #else:
        #    prefetchReg = None
        prefetchReg = self.generator.init_prefetching(self.prefetching)

        # if matrices are always padded to multiple of v_size, we can remove the if-part and execute the assert for SVE/SME too
        if not self.is_sve:
            assert(self.m % self.v_size == 0)

        self.A_regs, self.B_regs, self.C_regs, self.starting_regs, self.alpha_reg, self.beta_reg, self.loop_reg, self.additional_regs = self.generator.make_reg_blocks(self.bm, self.bn, self.bk, self.v_size, self.nnz, self.m, self.n, self.k)

        self.alpha_bcst_reg, self.beta_bcst_reg = self.starting_regs[3], self.starting_regs[4]


        # create transposed cursor of A for SME outer product
        if not self.is_sme:
            self.A = DenseCursor("A", self.starting_regs[0], self.m, self.k, self.lda, self.bm, self.bk, self.precision.value)
        else:
            self.A = DenseCursor("A", self.starting_regs[0], self.k, self.m, self.lda, self.bk, self.bm, self.precision.value)
        self.B = BlockCursor("B", self.starting_regs[1], self.k, self.n, self.ldb, self.bk, self.bn, self.precision.value, blocks, patterns,mtx_overhead)
        self.C = DenseCursor("C", self.starting_regs[2], self.m, self.n, self.ldc, self.bm, self.bn, self.precision.value)
        self.C_pf = DenseCursor("C_pf", prefetchReg, self.m, self.n, self.ldc, self.bm, self.bn, self.precision.value) if prefetchReg else None


    def make_nk_unroll(self):

        asm = block("Unrolling over bn and bk")
        A_ptr = CursorLocation()
        B_ptr = self.B.start()
        C_ptr = CursorLocation()
        C_pf_ptr = CursorLocation()

        Bn = self.n // self.bn
        Bk = self.k // self.bk
        # handle fringe case of SVE -> allow bm < v_size
        vm = self.bm // self.v_size if not self.is_sve else self.generator.ceil_div(self.bm, self.v_size)

        n_overhead = self.n % self.bn
        k_overhead = self.k % self.bk

        if n_overhead > 0:
            Bn += 1
        if k_overhead > 0:
            Bk += 1

        asm.add(self.generator.make_b_pointers(self.starting_regs[1], self.additional_regs, self.nnz))

        for Bni in range(0, Bn):
            
            regs = self.C_regs

            if Bni + 1 == Bn and n_overhead > 0:
                regs = self.C_regs[0:vm, 0:n_overhead]

            if self.alpha == 1.0 and self.beta != 0.0:
                asm.add(self.generator.move_register_block(self.C, C_ptr, Coords(), regs, self.v_size, self.additional_regs, None, False))
                if self.beta != 1.0:
                    if self.use_bcst:
                        asm.add(bcst(self.beta_bcst_reg, self.beta_reg[1], "Broadcast beta"))
                    for ic in range(regs.shape[1]):
                        for ir in range(regs.shape[0]):
                            pred_m = None 
                            # SME needs a separate pred_m because the row block dimension is bk instead of bm
                            if self.is_sme:
                                # is there a better way to handle SME not allowing multiplication of ZA rows with vector registers?
                                pred_m = self.generator.pred_n_trues(self.bk - ir * self.v_size, self.v_size, "m")
                                # swapped indices of A_regs[ir,ic] because we transposed A for SME
                                asm.add(mov(regs[ir,ic], self.A_regs[ic,ir], True, "move ZA row to vector register", pred=pred_m))
                                asm.add(mul(self.A_regs[ic,ir], self.beta_reg[1], self.A_regs[ic,ir], "C = beta * C", pred=pred_m))
                                asm.add(mov(self.A_regs[ic,ir], regs[ir,ic], True, "move vector register to ZA row", pred=pred_m))
                            elif self.is_sve:# and not self.is_sme:
                                # SME has bk as block row dimension, the declaration below might cause problems for SME
                                pred_m = self.generator.pred_n_trues(self.bm - ir * self.v_size, self.v_size, "m")
                            else:
                                asm.add(mul(regs[ir,ic], self.beta_reg[1], regs[ir,ic], "C = beta * C", pred=pred_m))                    
            else:
                asm.add(self.generator.make_zero_block(regs, self.additional_regs))

            for Bki in range(0,Bk):
                to_A = Coords(right=Bki) if not self.is_sme else Coords(down=Bki) 
                to_B = Coords(right=Bni, down=Bki, absolute=True)

                if self.B.has_nonzero_block(B_ptr, to_B):
                    asm.add(self.generator.make_microkernel(self.A, self.B, A_ptr, B_ptr, self.A_regs, self.B_regs, regs, self.v_size, self.additional_regs, to_A, to_B))

            if self.alpha != 1.0:
                store_block = block("")

                if self.use_bcst:
                    store_block.add(bcst(self.alpha_bcst_reg, self.alpha_reg[1] if not self.is_sme else self.beta_reg[1], "Broadcast alpha"))
                    if self.beta != 0.0 and self.beta != 1.0:
                        # SME needs to use the alpha_reg here to broadcast beta, this is helpful for the SME version of FMLA
                        store_block.add(bcst(self.beta_bcst_reg, self.beta_reg[1], "Broadcast beta"))

                if self.is_sme:
                    step_size = self.A_regs.shape[0]
                else:
                    step_size = self.A_regs.shape[1]
                for x in range(0, regs.shape[1], step_size):
                    A_regs_cut = self.A_regs[0:min(self.A_regs.shape[0], regs.shape[0]), 0:regs.shape[1]-x]
                    if self.is_sme:
                        A_regs_cut = self.A_regs[0:min(self.A_regs.shape[0], regs.shape[1]), 0:regs.shape[1]-x]
                        B_regs_cut = self.B_regs[0:min(self.B_regs.shape[0], regs.shape[1]), 0:regs.shape[1]-x]
                    if self.beta != 0.0:
                        store_block.add(self.generator.move_register_block(self.C, C_ptr, Coords(), A_regs_cut if not self.is_sme else B_regs_cut, self.v_size, self.additional_regs, None, False, None, self.ldc * x))

                    for ir in range(A_regs_cut.shape[0]):
                        for ic in range(A_regs_cut.shape[1]):
                            pred_m = None
                            if self.is_sme:
                                pred_m = self.generator.pred_n_trues(self.bk - ic*self.v_size, self.v_size, "m")
                            elif self.is_sve:
                                pred_m = self.generator.pred_n_trues(self.bm - ir*self.v_size, self.v_size, "m")
                            
                            if self.beta != 0.0 and self.beta != 1.0:
                                if self.is_sme:
                                # helpful for SME FMLA: beta is in alpha_reg, C is in B_regs_cut
                                    store_block.add(bcst(self.beta_bcst_reg, self.alpha_reg[min(1, ir // 4)], "Broadcast beta"))
                                    store_block.add(mul(B_regs_cut[ir,ic], self.alpha_reg[min(1, ir // 4)], B_regs_cut[ir,ic], "beta * C", pred=pred_m))
                                else:
                                    store_block.add(mul(A_regs_cut[ir,ic], self.beta_reg[1], A_regs_cut[ir,ic], "beta * C", pred=pred_m))
                            if self.beta == 0.0:
                                mul_regs_cut = regs[ir, x + ic] if not self.is_sme else regs[x + ic, ir] 
                                store_block.add(mul(mul_regs_cut, self.alpha_reg[1] if not self.is_sme else self.beta_reg[1], A_regs_cut[ir, ic], "C = C + alpha * AB", pred=pred_m))
                            else:
                                if self.is_sme:
                                    store_block.add(mov(regs[ic, x + ir], A_regs_cut[ir, ic], True, "Move AB to vector register", pred=pred_m))
                                    store_block.add(mov(B_regs_cut[ir, ic], regs[ic, x + ir], True, "Move C to matrix register", pred=pred_m))
                                    # FMLA uses vector group of size 4 as source vectors
                                    if (ir + 1) % 4 == 0:
                                        store_block.add(bcst(self.alpha_bcst_reg, self.alpha_reg[min(1, ir // 4)], "Broadcast alpha"))
                                        # this creates valid bases indices for double and single precision, tested on SME512
                                        # subsequent slices are determined using a stride dependent on vector group size and vector length
                                        za_base_row_idx = ir // 4 * 2 + 1
                                        store_block.add(mov(za_base_row_idx, self.additional_regs[3], False, "Setup base za register"))
                                        store_block.add(fma(self.alpha_reg[min(1, ir // 4)], A_regs_cut[ir + 1 - 4, ic], regs[ic, x + ir + 1 - 4], "C = C + alpha * AB", False, pred=pred_m))
                                else:
                                    store_block.add(fma(regs[ir, x + ic], self.alpha_reg[1], A_regs_cut[ir, ic], "C = C + alpha * AB", False, pred=pred_m))
                    if self.is_sme and self.beta != 0.0:
                        # move the results of the fma instruction back into the right za0h.{s,d} slices
                        pred_m = self.generator.pred_n_trues(self.bk - ic*self.v_size, self.v_size, "m")
                        for ir in range(A_regs_cut.shape[0]):
                            for ic in range(A_regs_cut.shape[1]):
                                store_block.add(mov(regs[ic, x + ir], B_regs_cut[ir, ic], True, "Move C back to contiguous tile", pred=pred_m))
                                store_block.add(mov(B_regs_cut[ir, ic], regs[ic, x + ir], True, pred=pred_m))
                        A_regs_cut = regs[:,:]
                    store_block.add(self.generator.move_register_block(self.C, C_ptr, Coords(), A_regs_cut, self.v_size, self.additional_regs, None, True, self.prefetching, self.ldc * x))
                asm.add(store_block)

            else:
                asm.add(self.generator.move_register_block(self.C, C_ptr, Coords(), regs, self.v_size, self.additional_regs, None, True, self.prefetching))

            if (Bni != Bn-1):
                move_C, C_ptr = self.C.move(C_ptr, Coords(right=1), self.is_sme)
                asm.add(move_C)
                if self.C_pf:
                  move_C_pf, C_pf_ptr = self.C_pf.move(C_pf_ptr, Coords(right=1), self.is_sme)
                  asm.add(move_C_pf)


        return asm



    def make(self):
        
        A_ptr = CursorLocation()
        C_ptr = CursorLocation()
        C_pf_ptr = CursorLocation()

        Bm = self.m // self.bm
        Bn = self.n // self.bn
        Bk = self.k // self.bk

        if self.n % self.bn != 0:
            Bn += 1
        # allow defining Bk such that the final block can be shorter than Bk
#        if self.k % self.bk != 0:
#            Bk += 1

        loopBody = [
          self.make_nk_unroll(),
          *([self.A.move(A_ptr, Coords(down=1), self.is_sme)[0]] if not self.is_sme else [self.A.move(A_ptr, Coords(right=1), self.is_sme)[0]]),
          self.C.move(C_ptr, Coords(down=1, right=1-Bn), self.is_sme)[0]
        ]
        if self.C_pf:
          loopBody.append(self.C_pf.move(C_pf_ptr, Coords(down=1, right=1-Bn), self.is_sme)[0])

        asm = block("unrolled_{}x{}x{}".format(self.m,self.n,self.k),
            self.generator.bcst_alpha_beta(self.alpha_reg, self.beta_reg),
            self.generator.make_scaling_offsets(self.additional_regs, self.nnz),
            loop(self.loop_reg, 0, Bm, 1).body(*loopBody)
        )

        vm_overhead = (self.m % self.bm) // self.v_size

        if vm_overhead > 0:
            self.m = self.m % self.bm
            self.bm = self.m % self.bm
            self.A_regs = self.A_regs[0:self.bm // self.v_size, 0:self.bk]
            self.C_regs = self.C_regs[0:self.bm // self.v_size, 0:self.bn]
            self.A.r = self.m
            asm.add(self.make_nk_unroll())


        return asm

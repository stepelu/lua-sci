--------------------------------------------------------------------------------
-- BLAS support for sci.alg module.
--
-- Copyright (C) 2011-2014 Stefano Peluchetti. All rights reserved.
--
-- Features, documentation and more: http://www.scilua.org .
--
-- This file is part of the SciLua library, which is released under the MIT 
-- license: full text in file LICENSE.TXT in the library's root folder.
--------------------------------------------------------------------------------

local ffi = require "ffi"

local cfg = require "sci.alg.cfg"
local header = cfg.blasheader
local lib    = cfg.blaslib

local cdef = require(header).cdef
ffi.cdef(cdef)
local BLAS = ffi.load(lib)

-- Use one single thread: multiple threads speed-up computations only when
-- sizes are large. Moreover we want to enforce a model where each LuaJIT 
-- process is always single-threaded. High-level parallelization model with 
-- message-passing between LuaJIT computational units (nodes).
BLAS.openblas_set_num_threads(1) 

local ORDER   = BLAS.CblasRowMajor
local trans   = BLAS.CblasTrans
local notrans = BLAS.CblasNoTrans

-- C <- alpha*A*B + beta*C.
function dgemm(transA, transB, alpha, A, B, beta, C)
  local tA = transA and trans or notrans
  local tB = transB and trans or notrans
  local m, n, k = C:nrow(), C:ncol(), transA and A:nrow() or A:ncol()
  local lda = A:ncol()
  local ldb = B:ncol()
  BLAS.cblas_dgemm(ORDER, tA, tB, m, n, k, alpha, A._p, lda, B._p, ldb, beta, 
    C._p, n)
end

-- y <- alpha*A*x + beta*y.
function dgemv(transA, alpha, A, x, beta, y)
  local m, n = A:nrow(), A:ncol()
  local tA = transA and trans or notrans
  BLAS.cblas_dgemv(ORDER, tA, m, n, alpha, A._p, n, x._p, 1, beta, y._p, 1)
end

return {
  dgemm = dgemm,
  dgemv = dgemv,
}
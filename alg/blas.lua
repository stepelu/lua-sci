--------------------------------------------------------------------------------
-- BLAS support.
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

local function warn(...)
  io.stderr:write(...)
end

if not cfg.blas.enable then
  if cfg.blas.banner then
    warn("[BLAS --- disabled in sci.alg.cfg]\n")
  end
  return false
end

local _L = { }
local search = package.searchpath

local ok, BLAS = pcall(function()
  if jit.os == "Windows" then
    local path = package.path:gsub("%.lua", "%.dll")
    _L[-2] = ffi.load(search("sci.alg.windows.libgcc_s_dw2-1", path))
    _L[-1] = ffi.load(search("sci.alg.windows.libquadmath-0",  path))
    _L[0]  = ffi.load(search("sci.alg.windows.libgfortran-3",  path))
    _L[1]  = ffi.load(search("sci.alg.windows.libopenblas",    path))
  elseif jit.os == "OSX" then
    _L[1] = ffi.load("blas") -- Accelerate Framework.
  elseif jit.os == "Linux" then
    local path = package.path:gsub("%.lua", "%.so")
    _L[1] = ffi.load(search("sci.alg.linux.libopenblas", path))
  else
    error("BLAS not supported on OS="..jit.os)
  end
  return _L[1]
end)
if not ok then
  if cfg.blas.banner then
    warn("[BLAS --- dynamic library loading failed: '"..BLAS.."']\n")
  end
  return false
end
if cfg.blas.banner then
  warn("[BLAS --- enabled]\n")
end

ffi.cdef(require("sci.alg.cblas_h"))

local ORDER   = BLAS.CblasRowMajor
local TRANS   = BLAS.CblasTrans
local NOTRANS = BLAS.CblasNoTrans

-- C = m*n.
-- A = m*k.
-- B = k*n.
-- C <- alpha*A*B + beta*C, no dimension checks.
local function new_gemm(BLAS_call)
  return function(C, A, B, transA, transB, alpha, beta)
    alpha = alpha or 1
    beta  = beta  or 0
    local tA = transA and TRANS or NOTRANS
    local tB = transB and TRANS or NOTRANS
    local m, n, k = C:nrow(), C:ncol(), transA and A:nrow() or A:ncol()
    local lda = A:ncol()
    local ldb = B:ncol()
    local ldc = C:ncol()
    BLAS[BLAS_call](ORDER, tA, tB, m, n, k, alpha, A:data(), lda, B:data(), ldb, 
      beta, C:data(), ldc)
  end
end

-- y <- alpha*A*x + beta*y, no dimension checks.
local function new_gemv(BLAS_call)
  return function(y, A, x, transA, alpha, beta)
    alpha = alpha or 1
    beta  = beta  or 0
    local tA = transA and TRANS or NOTRANS
    local m, n = A:nrow(), A:ncol()
    local lda = A:ncol()
    BLAS[BLAS_call](ORDER, tA, m, n, alpha, A:data(), lda, x:data(), 1, beta, 
      y:data(), 1)
  end
end

return {
  sgemm = new_gemm("cblas_sgemm"),
  dgemm = new_gemm("cblas_dgemm"),
  zgemm = new_gemm("cblas_zgemm"),
  sgemv = new_gemv("cblas_sgemv"),
  dgemv = new_gemv("cblas_dgemv"),
  zgemv = new_gemv("cblas_zgemv"),
  _L    = _L,
}
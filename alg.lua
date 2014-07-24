--------------------------------------------------------------------------------
-- Algebra module.
--
-- Copyright (C) 2011-2014 Stefano Peluchetti. All rights reserved.
--
-- Features, documentation and more: http://www.scilua.org .
--
-- This file is part of the SciLua library, which is released under the MIT
-- license: full text in file LICENSE.TXT in the library's root folder.
--------------------------------------------------------------------------------

--[[ Implementation Documentation ----------------------------------------------

TODO!

CEVEATS:
elements cannot be VLS/VLA
elements cannot require manual memory management (i.e pointers)
When reference semantics (struct) via indexing (due to FFI) and value 
semantics is desired instead then a elnew function/ctype must be provided;
notice that this implies immutability of the elements!

We keep :new() because we want to support users that prefers standard Lua 

PRINCIPLE:
The most standard and basic and the more simple and permanent it should be

------------------------------------------------------------------------------]]

-- TODO: Add removable tests for assumptions in low-level functions like 
-- TODO: *_memcpy, checks on type agreement, size agreement, bound checks.
-- TODO: row, col, sub, diag for matrix.

local ffi  = require "ffi"
local bit  = require "bit"
local xsys = require "xsys"
local blas = require("sci.alg.blas")
local cfg  = require "sci.alg.cfg"

local new, copy, fill, sizeof, typeof, metatype = xsys.from(ffi,
     "new, copy, fill, sizeof, typeof, metatype")
local type, select, tonumber, rawequal = type, select, tonumber, rawequal

local width = xsys.string.width
local concat = table.concat

local double_ct, complex_ct = typeof("double"), typeof("complex")
local float_ct = typeof("float")

local floor = math.floor
local band = bit.band

local unroll, buffer = cfg.unroll, cfg.buffer

-- Vector ----------------------------------------------------------------------
local function vec_alloc(ct, n)
  local v = new(ct, n) -- PERF: Default initialization of VLS, compiled.
  v._n, v._p = n, v._a
  return v -- VLS are automatically zero-filled for default initializer case.
end

local function vec_memmap(ct, n, p)
  local v = new(ct, 0) -- PERF: Default initialization of VLS, compiled.
  v._n, v._p = n, p
  return v
end

local function vec_memcpy(y, x)
  copy(y._p, x._p, sizeof(y:elct())*y._n)
end

local function vec_memcpy_offset(y, x, o)
  copy(y._p, x._p + o, sizeof(y:elct())*y._n)
end

local function vec_set(y, x)
  if x:elct() == y:elct() then
    vec_memcyp(y, x)
  else
    local elnew = y:elnew()
    if elnew then
      for i=0,#y-1 do
        y._p[i] = elnew(x._p[i])
      end
    else
      for i=0,#y-1 do
        y._p[i] = x._p[i]
      end      
    end
  end
end

-- TODO: Use vec_set (modified to handle offset) when not table.
local function vec_tab_set(y, x, off, n)
  for i=1,n do 
    y[i+off] = x[i]
  end
end

-- TODO: Specialize over number of arguments.
local function vec_tovec(ct, ...)
  local narg = select("#", ...)
  local a = { ... }
  local n = 0
  for i=1,narg do
    n = n + #a[i]
  end
  local v = vec_alloc(ct, n) -- PERF: Default initialization of VLS, compiled.
  local off = 0
  for i=1,narg do
    local ni = #a[i]
    vec_tab_set(v, a[i], off, ni)
    off = off + ni
  end
  return v
end

local function new_tovec(vec_ct)
  return function(...)
    return vec_tovec(vec_ct, ...)
  end
end

local function new_vec_ct(elct, elnew, stack)
  local vec_mf = {
    elct = function()
      return elct
    end,
    elnew = function()
      return elnew
    end,
    stack = stack,
    data = function(self)
      return self._p
    end,
    new = function(self)
      return vec_alloc(self, self._n)
    end,
    copy = function(self)
      local v = vec_alloc(self, self._n)
      vec_memcpy(v, self)
      return v
    end,
    sub = function(self, f, l) -- Allows for l == f-1 --> 0-sized array.
      f = f or 1
      l = l or self._n
      if f < 1 or f - 1 > l or l > self._n or f > self._n then
        error("out of bounds range: f="..f..", l="..l..", #="..self._n)
      end
      local v = vec_alloc(self, l - f + 1)
      vec_memcpy_offset(v, self, f - 1)
      return v
    end,
    clear = function(self)
      fill(self._p, sizeof(self:elct())*self._n)
    end,
    totable = function(self)
      local o = { }
      for i=1,self._n do
        o[i] = self[i]
      end
      return o
    end,
    width = function(self, chars)
      local o = { }
      for i=1,self._n do
        o[i] = width(self[i], chars)
      end
      return concat(o, ",")
    end,
  }
  local vec_mt = {
    __new = function(ct, n)
      if not type(n) == "number" then
        error("size must be a number")
      end
      if not (n >= 0) then
        error("size must be non-negative, size="..n)
      end
      return vec_alloc(ct, n)
    end,
    __len = function(self)
      return self._n
    end,
    __index = elnew and function(self, k)
      if type(k) == "number" then
        if not (1 <= k and k <= self._n) then
          error("out of bounds index: i="..k..", #="..self._n)
        end
        return elnew(self._p[k-1]) -- To have value semantics.
      else    
        return vec_mf[k]
      end
    end or function(self, k)
      if type(k) == "number" then
        if not (1 <= k and k <= self._n) then
          error("out of bounds index: i="..k..", #="..self._n)
        end
        return self._p[k-1]
      else    
        return vec_mf[k]
      end
    end,
    __newindex = elnew and function(self, i, v)
      if not (1 <= i and i <= self._n) then
        error("out of bounds index: i="..i..", #="..self._n)
      end
      self._p[i-1] = elnew(v) -- For conversions.
    end or function(self, i, v)
      if not (1 <= i and i <= self._n) then
        error("out of bounds index: i="..i..", #="..self._n)
      end
      self._p[i-1] = v 
    end,
    __tostring = function(self)
      local o = { }
      for i=1,self._n do
        o[i] = tostring(self[i])
      end
      return "{"..concat(o, ",").."}"
    end,
  }
  local ct = typeof("struct { int32_t _n; $* _p; $ _a[?]; }", elct, elct)
  return metatype(ct, vec_mt)
end

-- Matrix ----------------------------------------------------------------------
local function mat_alloc(ct, n, m)
  local v = new(ct, n*m) -- PERF: Default initialization of VLS, compiled.
  v._n, v._m, v._p = n, m, v._a
  return v
end

local function mat_memmap(ct, n, m, p)
  local v = new(ct, 0) -- PERF: Default initialization of VLS, compiled.
  v._n, v._m, v._p = n, m, p
  return v
end

local function mat_memcpy(y, x)
  copy(y._p, x._p, sizeof(y:elct())*y._n*y._m)
end

local function mat_set(y, x)
  if x:elct() == y:elct() then
    mat_memcpy(y, x)
  else
    local elnew, n = y:elnew(), y._n*y._m
    if elnew then
      for i=0,n-1 do
        y._p[i] = elnew(x._p[i])
      end
    else
      for i=0,n-1 do
        y._p[i] = x._p[i]
      end      
    end
  end
end

local function mat_tab_dim(x)
  if type(x) == "table" then
    return #x, #x[1]
  else
    return x:nrow(), x:ncol()
  end
end

-- TODO: Use mat_set (modified to handle offset) when not table.
local function mat_tab_set(y, x, off, n, m)
  for r=1,n do for c=1,m do
    y[r+off][c] = x[r][c]
  end end
end

-- TODO: Specialize over number of arguments.
local function mat_aggregate(ct, ...)
  local narg = select("#", ...)
  local a = { ... }
  local n, m = 0
  for i=1,narg do
    local ni, mi = mat_tab_dim(a[i])
    if m and (mi ~= m) then
      error("all arguments must have the same number of columns")
    end
    n, m = n + ni, mi
  end
  local v = mat_alloc(ct, n, m) -- PERF: Default initialization of VLS, compiled.
  local off = 0
  for i=1,narg do
    local ni = mat_tab_dim(a[i])
    mat_tab_set(v, a[i], off, ni, m)
    off = off + ni
  end
  return v
end

local function new_tomat(mat_ct)
  return function(...)
    return mat_aggregate(mat_ct, ...)
  end
end

-- Do not call directly, called by new_mat_ct already:
local function new_row_ct(elct, elnew)
  local row_mt = {
    __index = elnew and function(self, i)
      if not (1 <= i and i <= self._m) then
        error("out of bounds col index: col="..i..", #col="..self._m)
      end
      return elnew(self._p[i-1]) -- To have value semantics.
    end or function(self, i)
      if not (1 <= i and i <= self._m) then
        error("out of bounds col index: col="..i..", #col="..self._m)
      end
      return self._p[i-1]
    end,
    __newindex = elnew and function(self, i, v)
      if not (1 <= i and i <= self._m) then
        error("out of bounds col index: col="..i..", #col="..self._m)
      end
      self._p[i-1] = elnew(v) -- For conversions.
    end or function(self, i, v)
      if not (1 <= i and i <= self._m) then
        error("out of bounds col index: col="..i..", #col="..self._m)
      end
      self._p[i-1] = v
    end,
  }
  return metatype(typeof("struct { int32_t _m; $* _p; }", elct), row_mt)
end

local function new_mat_ct(elct, elnew, stack)
  local row_ct = new_row_ct(elct, elnew)
  local mat_mf = {
    elct = function()
      return elct
    end,
    elnew = function()
      return elnew
    end,
    stack = stack,
    data = function(self)
      return self._p
    end,
    new = function(self)
      return mat_alloc(self, self._n, self._m)
    end,
    copy = function(self)
      local v = mat_alloc(self, self._n, self._m)
      mat_memcpy(v, self)
      return v
    end,
    clear = function(self)
      fill(self._p, sizeof(self:elct())*self._n*self._m)
    end,
    nrow = function(self)
      return self._n
    end,
    ncol = function(self)
      return self._m
    end,
    totable = function(self)
      local o = { }
      for i=1,self._n do
        local oo = { }
        for j=1,self._m do
          oo[j] = self[i][j]
        end
        o[i] = oo
      end
      return o
    end,
    width = function(self, chars)
      local o = { }
      for i=1,self._n do
        local oo = { }
        for j=1,self._m do
          oo[j] = width(self[i][j], chars)
        end
        o[i] = concat(oo, ",")
      end
      return concat(o, "\n")
    end,
  }
  local mat_mt = {
    __new = function(ct, n, m)
      if not (type(n) == "number" and type(m) == "number") then
        error("sizes must be numbers")
      end
      if not (n >= 0 and m >= 0) then
        error("size must be non-negative, size="..n.."x"..m)
      end
      return mat_alloc(ct, n, m)
    end,
    __index = function(self, k)
      if type(k) == "number" then
        if not (1 <= k and k <= self._n) then
          error("out of bounds row index: row="..k..", #row="..self._n)
        end
        return row_ct(self._m, self._p + (k-1)*self._m)
      else    
        return mat_mf[k]
      end
    end,
    __tostring = function(self)
      local o = { }
      for i=1,self._n do
        local oo = { }
        for j=1,self._m do
          oo[j] = tostring(self[i][j])
        end
        o[i] = "{"..concat(oo, ",").."}"
      end
      return "{"..concat(o, ",").."}"
    end,
  }
  local ct = typeof("struct { int32_t _n, _m; $* _p; $ _a[?]; }", elct, elct)
  return metatype(ct, mat_mt)
end

-- Stack -----------------------------------------------------------------------
-- TODO: Consider having a growth-able stack limited by _max, the growth would
-- TODO: be triggered in clear (to avoid invalidating *_map objects) starting 
-- TODO: from a minimum value (configurable as well) and doubling each time up
-- TODO: to _max.
local function stack_data(stack, n)
  stack._n = stack._n + n
  return stack._p + (stack._n - n)
end

local function new_stack_ct(elct, size, vec_ct, mat_ct)
  local ct = typeof("struct { int32_t _max, _n; $ _p[?]; }", elct)
  local stack = ct(size, size)
  local stack_mt = {
    clear = function()
      stack._n = 0
    end,
    vec = function(n)
      if stack._n + n <= stack._max then
        return vec_memmap(vec_ct, n, stack_data(stack, n))
      else
        return vec_alloc(vec_ct, n)
      end
    end,
    mat = function(n, m)
      if stack._n + n*m <= stack._max then
        return mat_memmap(mat_ct, n, m, stack_data(stack, n*m))
      else
        return mat_alloc(mat_ct, n, m)
      end
    end,
  }
  stack_mt.__index = stack_mt  
  return metatype("struct { }", stack_mt)
end

-- Algorithms ------------------------------------------------------------------
-- In all algorithms a prefix _ means that it's the private version:
-- + no dimension checks
-- + not alias safe
-- + they are guaranteed not to obtain new stack or call stack.clear(), 
--   hence they are passed a stack when necessary
-- + they are guaranteed not to throw an error, but can return nil, err

-- We also try to avoid using math functions, just operators. This way the 
-- algorithms are compatible with types different from double.

-- Multiply --------------------------------------------------------------------
local function chk_eq_square(X, Y)
  local n, m = X:nrow(), X:ncol()
  local ny, my = Y:nrow(), Y:ncol()
  if not (n == m and n == ny and n == my) then
    error("arguments must be square matrices of equal size")
  end
  return n, m
end

-- Generic version when no BLAS or unsupported element type, loop ordering is
-- more cache-friendly than intuitive version.
local function _mulmm_generic(C, A, B, transA, transB)
  local M, N, K = C:nrow(), C:ncol(), not transA and A:ncol() or A:nrow()
  C:clear()
  if not transA and not transB then
    for k=1,K do for i=1,M do for j=1,N do
      C[i][j] = C[i][j] + A[i][k]*B[k][j]
    end end end
  elseif transA and not transB then
    for k=1,K do for i=1,M do for j=1,N do
      C[i][j] = C[i][j] + A[k][i]*B[k][j]
    end end end
  elseif not transA and transB then
    for k=1,K do for i=1,M do for j=1,N do
      C[i][j] = C[i][j] + A[i][k]*B[j][k]
    end end end
  elseif transA and transB then
    for k=1,K do for i=1,M do for j=1,N do
      C[i][j] = C[i][j] + A[k][i]*B[j][k]
    end end end
  end
end

local function _mulmv_generic(y, A, x, transA)
  local M, N = #y, not transA and A:ncol() or A:nrow()
  for i=1,M do y[i] = 0 end
  if not transA then
    for i=1,M do for j=1,N do
      y[i] = y[i] + A[i][j]*x[j]
    end end
  else
    for j=1,N do for i=1,M do
      y[i] = y[i] + A[j][i]*x[j]
    end end   
  end
end

local function iseltype2(ct, x, y)
  return x:elct() == ct and y:elct() == ct
end

local function _mulmm_blas(C, A, B, transA, transB)
  if iseltype2(double_ct, A, B) then
    blas.dgemm(C, A, B, transA, transB)
  elseif iseltype2(complex_ct, A, B) then
    blas.zgemm(C, A, B, transA, transB)
  elseif iseltype2(float_ct, A, B) then
    blas.sgemm(C, A, B, transA, transB)
  else
    _mulmm_generic(C, A, B, transA, transB)
  end
end

local function _mulmv_blas(y, A, x, transA)
  if iseltype2(double_ct, A, x) then
    blas.dgemv(y, A, x, transA)
  elseif iseltype2(complex_ct, A, x) then
    blas.zgemv(y, A, x, transA)
  elseif iseltype2(float_ct, A, x) then
    blas.sgemv(y, A, x, transA)
  else
    _mulmv_generic(y, A, x, transA)
  end
end

local _mulmm = blas and _mulmm_blas or _mulmm_generic
local _mulmv = blas and _mulmv_blas or _mulmv_generic

-- Exponentiation by squaring algorithm:
local function _rec_powms(A, s, n, stack)
  local T = stack.mat(n, n) -- Cannot alias A.
  if s == 1 then
    -- Cannot return A because could generate aliasing between R and T below:
    mat_memcpy(T, A)
    return T
  elseif s == 2 then
    _mulmm(T, A, A)
    return T
  elseif band(s, 1) == 0 then -- Even.
    _mulmm(T, A, A)
    return _rec_powms(T, s/2, n, stack)
  else
    _mulmm(T, A, A)
    local R = _rec_powms(T, (s - 1)/2, n, stack) -- R cannot alias T.
    _mulmm(T, R, A)
    return T
  end
end

-- TODO: Use SVD decomposition for large s and allow positive real s.
local function _powms(B, A, s, stack)
  local n = B:nrow()
  if s == 0 then
    B:clear()
    for i=1,n do B[i][i] = 1 end
  elseif s == 1 then
    mat_set(B, A)
  else
    local T = _rec_powms(A, s, n, stack)
    mat_memcpy(B, T)
  end
end

local function powms(B, A, s)
  chk_eq_square(B, A)
  if s < 0 or floor(s) ~= s then
    error("NYI: matrix exponentiation supported only for non-negative integers")
  end
  local stack = B:stack()
  _powms(B, A, s, stack)
  stack.clear()
end

-- Dimension checks, aliasing safe:
local function mulmm(C, A, B, transA, transB)
  local Cn, Cm = C:nrow(), C:ncol()
  local An, Am = A:nrow(), A:ncol() 
  if transA then
    An, Am = Am, An
  end
  local Bn, Bm = B:nrow(), B:ncol() 
  if transB then
    Bn, Bm = Bm, Bn
  end
  if Cn ~= An or Cm ~= Bm or Am ~= Bn then
    error("incompatible dimensions in matrix-matrix multiplication")
  end
  if rawequal(C, A) or rawequal(C, B) then
    local CS = C:stack().mat(Cn, Cm)
    _mulmm(CS, A, B, transA, transB)
    mat_memcpy(C, CS)
    C:stack().clear()
  else
    _mulmm(C, A, B, transA, transB)
  end
end

-- Dimension checks, aliasing safe:
local function mulmv(y, A, x, transA)
  local An, Am = A:nrow(), A:ncol() 
  if transA then
    An, Am = Am, An
  end
  if #y ~= An or Am ~= #x then
    error("incompatible dimensions in matrix-vector multiplication")
  end
  if rawequal(y, x) then
    local ys = y:stack().vec(#y)
    _mulmv(ys, A, x, transA)
    vec_memcpy(y, ys)
    y:stack().clear()
  else
    _mulmv(y, A, x, transA)
  end
end

-- Factor ----------------------------------------------------------------------
-- Perform a LT Cholesky factorization of PD (positive-definite) matrix A,
-- that is A = L*L:t() in 1/6*n^3 operations.
-- LT has better performance (more cache friendly even if more divisions, more
-- LuaJIT-friendly) then UT version. For the UT version see:
-- "Matrix Inversion Using Cholesky Decomposition", 2011.
-- L *can* alias A (in-place Cholesky), only LT part of L written (alias safe).
-- TODO: add 'kind' option: lt or ut.
local function _cholesky(L, A)
  local n = A:nrow()
  for i=1,n do
    -- Strictly lower diagonal part.
    for j=1,i-1 do
    local sum = 0
      for k=1,j-1 do
        sum = sum + L[i][k]*L[j][k]
      end
      L[i][j] = (A[i][j] - sum)/L[j][j]
    end
    -- Diagonal part.
    local sum = 0
    for k=1,i-1 do
      sum = sum + L[i][k]^2
    end
    L[i][i] = (A[i][i] - sum)^0.5 -- Works with non-double as well.
    if L[i][i] ~= L[i][i] then
      return nil, "input matrix is not positive-definite"
    end
  end
  return true
end

local function factor(Y, X, kind)
  if kind == "posdef" then
    chk_eq_square(Y, X)
    local ok, err = _cholesky(Y, X)
    if not ok 
      then return nil, err
    end
    -- Must fill with 0 the upper triangular part.
    local n = Y:nrow()
    for r=1,n do for c=r+1,n do Y[r][c] = 0 end end
    return true
  else
    error("NYI: only factorization of 'posdef' matrices is implemented")
  end
end

-- Invert ----------------------------------------------------------------------
-- Only LT parts of X and Y are used, Y *must not* alias X.
local function _invltmat(Y, X)
  local n = X:nrow()
  for i=1,n do
    Y[i][i] = 1/X[i][i]
  end
  for r=2,n do
    for c=1,r-1 do
      local sum = 0
      for k=1,r-1 do
        sum = sum + X[r][k]*Y[k][c]
      end
      Y[r][c] = -sum/X[r][r] 
    end
  end
  return true
end

local function _invertposdef(Y, X, stack)
  local ok, err = _cholesky(Y, X)
  if not ok then 
    return nil, err
  end
  local T = stack.mat(X:nrow(), X:ncol())
  T:clear()
  _invltmat(T, Y)
  _mulmm(Y, T, T, true)
  return true
end

local function invert(Y, X, kind)
  chk_eq_square(Y, X)
  if kind == "posdef" then
    local stack = Y:stack()
    local ok, err = _invertposdef(Y, X, stack)
    stack.clear()
    if ok then
      return true
    else
      return nil, err
    end
  else
    error("NYI: only inversion of 'posdef' matrices is implemented")
  end
end

-- Typeof ----------------------------------------------------------------------
local alg_elct = { }

local function alg_typeof(elct, elnew)
  elct = typeof(elct) -- Allows for string definitions.
  local elctnum = tonumber(elct)
  if alg_elct[elctnum] then
    return alg_elct[elctnum]
  end
  local stack_elct
  local function stack()
    return stack_elct
  end
  local vec_ct   = new_vec_ct(elct, elnew, stack)
  local mat_ct   = new_mat_ct(elct, elnew, stack)
  local tovec    = new_tovec(vec_ct)
  local tomat    = new_tomat(mat_ct)
  local maxstack = buffer/sizeof(elct)
  stack_elct = new_stack_ct(elct, maxstack, vec_ct, mat_ct)
  alg_elct[elctnum] = { 
    vec   = vec_ct, 
    mat   = mat_ct,
    tovec = tovec,
    tomat = tomat,
    stack = stack_elct,
  }
  return alg_elct[elctnum]
end

local alg_double = alg_typeof("double")

return {
  typeof = alg_typeof,
  mulmm  = mulmm,
  powms  = powms,
  mulmv  = mulmv,
  factor = factor,
  invert = invert,
  vec    = alg_double.vec,
  mat    = alg_double.mat,
  tovec  = alg_double.tovec,
  tomat  = alg_double.tomat,
  stack  = alg_double.stack,
}

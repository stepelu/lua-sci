--------------------------------------------------------------------------------
-- Matrix and vector algebra module.
--
-- Copyright (C) 2011-2014 Stefano Peluchetti. All rights reserved.
--
-- Features, documentation and more: http://www.scilua.org .
--
-- This file is part of the SciLua library, which is released under the MIT 
-- license: full text in file LICENSE.TXT in the library's root folder.
--------------------------------------------------------------------------------

-- TODO: Stack.
-- TODO: Use only the algorithms in OpenBLAS: support only the supported types.
-- TODO: Views: sub, row, col, diag
-- TODO: Custom allocator
-- TODO: Can bound checks be optimized?
-- TODO: Can access be optimized (contiguous memory + $** / no shifts) ?
-- TODO: Can access to BLAS functions be optimized?
-- TODO: For vectors: r = c = 0 or r = 1, c = n ? (totable then changes)
-- TODO: Minimize dimension and type checks in BLAS operations.
-- TODO: Can the request to the stack be optimized (is the pointer sunk?) ?
-- TODO: Is it faster to check for aliasing with raweq ?
-- TODO: better dimension reporting.
-- TODO: Use column vectors: think of matrix vector ,multiplication.
-- TODO: Just mul() instead of mulmv() and mulmm().
-- TODO: Consider avoiding type checks via calls like x:_method_element_type().

-- Notes:
-- + BLAS requires contiguous memory and allows only for aliasing between inputs
-- + to decide faster way a reasonable number of benchmarks must be available, 
--   including ones that perform allocations.

local STACK_PER_ELEMENT_BUFFER = 10e6
assert(STACK_PER_ELEMENT_BUFFER >= 0)
local JOIN_UNROLL = 5
assert(JOIN_UNROLL > 0)

local ffi  = require 'ffi'
local bit  = require 'bit'
local xsys = require 'xsys'

local type, setmetatable = type, setmetatable
local band = bit.band
local floor = math.floor

local template = xsys.template
local width = xsys.string.width

-- Array memory ops ------------------------------------------------------------
-- Invariant: n == r*c.
local function array_alloc(ct, n, r, c)
  local a = ffi.new(ct, n) -- Default initialization of VLS, compiled.
  a._n, a._r, a._c = n, r, c
  a._p = a._v
  return a -- VLS are automatically zero-filled for default initializer case.
end

local function array_map(ct, n, r, c, p)
  local a = ffi.new(ct, 0) -- Default initialization of VLS, compiled.
  a._n, a._r, a._c = n, r, c
  a._p = p
  return a -- VLS are automatically zero-filled for default initializer case.
end

local function array_copy_data(dest, source)
  local source_size = ffi.sizeof(source:elementct())*source._n
  ffi.copy(dest._p, source._p, source_size)
end

local function array_copy_data_offset(dest, source, offset)
  local source_size = ffi.sizeof(source:elementct())*source._n
  ffi.copy(dest._p + offset, source._p, source_size)
end

local function array_clear(x)
  local x_size = ffi.sizeof(x:elementct())*x._n
  ffi.fill(x._p, x_size)
end

-- Stack -----------------------------------------------------------------------
-- TODO: Use malloc.
-- TODO: Allow growth.

local stack_struct = 'struct { int32_t _max, _n; $ _p[?]; }'

local function mem_stack_data(self, n)
  self._n = self._n + n
  return self._p + (self._n - n)
end

local function new_mem_stack_ct(element_ct)
  local stack_mt = {
    __new = function(ct, maxsize)
      local o = ffi.new(ct, maxsize)
      o._max = maxsize
      return o
    end,
    clear = function(self)
      array_clear(self)
      self._n = 0
    end,
    request = function(self, n)
      return self._n + n <= self._max and mem_stack_data(self, n)
    end,
    elementct = function()
      return element_ct
    end,
  }
  stack_mt.__index = stack_mt 

  local stack_ct = ffi.typeof(stack_struct, element_ct) 
  return ffi.metatype(stack_ct, stack_mt)
end

-- BLAS ------------------------------------------------------------------------
local blas_element_code = template([[
local ffi     = require 'ffi'
local cblas_h = require 'sci._cblas_h'

ffi.cdef(cblas_h)
local blas = ffi.load('libopenblas')

local complex_a1 = ffi.typeof('complex[1]')
local compflo_a1 = ffi.typeof('complex float[1]')

local compfloa, compflob = compflo_a1(), compflo_a1()
local complexa, complexb = complex_a1(), complex_a1()

return {
| for ELEMENT_NAME, BLAS in pairs{
|   float             = { PREFIX = 's' },
|   double            = { PREFIX = 'd' },
|   ['complex float'] = { PREFIX = 'c', ALPHA = 'compfloa', BETA = 'compflob' },
|   complex           = { PREFIX = 'z', ALPHA = 'complexa', BETA = 'complexb' },
| } do
  [tonumber(ffi.typeof('${ELEMENT_NAME}'))] = {
    gemm = function(C, A, B, At, Bt, alpha, beta)
      ${BLAS.ALPHA and BLAS.ALPHA..'[0] = alpha'}
      ${BLAS.BETA and BLAS.BETA..'[0] = beta'}
      blas.cblas_${BLAS.PREFIX}gemm(
        blas.CblasRowMajor, 
        At and blas.CblasTrans or blas.CblasNoTrans, 
        Bt and blas.CblasTrans or blas.CblasNoTrans, 
        C:nrow(), 
        C:ncol(), 
        At and A:nrow() or A:ncol(), 
        ${BLAS.ALPHA and BLAS.ALPHA or 'alpha'},
        A:data(), 
        A:ncol(), 
        B:data(), 
        B:ncol(), 
        ${BLAS.BETA and BLAS.BETA or 'beta'},
        C:data(), 
        C:ncol()
      )
    end,
    gemv = function(y, A, x, At, alpha, beta)
      ${BLAS.ALPHA and BLAS.ALPHA..'[0] = alpha'}
      ${BLAS.BETA and BLAS.BETA..'[0] = beta'}
      blas.cblas_${BLAS.PREFIX}gemv(
        blas.CblasRowMajor, 
        At and blas.CblasTrans or blas.CblasNoTrans, 
        A:nrow(), 
        A:ncol(),
        ${BLAS.ALPHA and BLAS.ALPHA or 'alpha'},
        A:data(), 
        A:ncol(),
        x:data(), 
        1,
        ${BLAS.BETA and BLAS.BETA or 'beta'},
        y:data(),
        1
      )
    end,
  },
| end
}
]])()

local blas_element_ct = assert(loadstring(blas_element_code))()

local function same_type_2_check(x, y)
  if x:elementct() ~= y:elementct() then
    error('constant element type required')
  end
end

local function same_type_3_check(x, y, z)
  local ct = x:elementct()
  if ct ~= y:elementct() or ct ~= z:elementct() then
    error('constant element type required')
  end
end

local function dimensions_mat(A, At)
  local Ar, Ac = A:nrow(), A:ncol()
  if At then
    Ar, Ac = Ac, Ar
  end
  return Ar, Ac
end

local function dimensions_mul_check(A, B, At, Bt)
  local Ar, Ac = dimensions_mat(A, At)
  local Br, Bc = dimensions_mat(B, Bt)
  if Ac ~= Br then
    error("incompatible dimensions in matrix-matrix multiplication")
  end
  return Ar, Bc
end

local function dimensions_mat_same_check(A, At, nrow, ncol)
  local Ar, Ac = dimensions_mat(A, At)
  if Ar ~= nrow or Ac ~= ncol then
    error('unexpected matrix dimensions')
  end
end

local function dimensions_mat_square_check(A)
  if A:nrow() ~= A:ncol() then
    error('square matrix expected')
  end
end

local function mul(C, A, B, At, Bt)
  same_type_3_check(C, A, B)
  local Cr, Cc = dimensions_mul_check(A, B, At, Bt)
  dimensions_mat_same_check(C, false, Cr, Cc)
  local T = C:_stack_array(Cr*Cc, Cr, Cc)
  if Cc == 1 then
    T:_gemv(A, B, At, 1, 0)
  else
    T:_gemm(A, B, At, Bt, 1, 0)
  end
  array_copy_data(C, T)
  C:_stack_clear()
end

-- Exponentiation by squaring algorithm:
local function pow_recursive(A, s, n)
  local T = A:_stack_array(n*n, n, n)
  if s == 1 then
    -- Cannot return A because could generate aliasing between R and T below.
    array_copy_data(T, A)
    return T
  elseif s == 2 then
    T:_gemm(A, A, false, false, 1, 0)
    return T
  elseif band(s, 1) == 0 then -- Even.
    T:_gemm(A, A, false, false, 1, 0)
    return pow_recursive(T, s/2, n)
  else
    T:_gemm(A, A, false, false, 1, 0)
    local R = pow_recursive(T, (s - 1)/2, n) -- R cannot alias T.
    T:_gemm(R, A, false, false, 1, 0)
    return T
  end
end

local function pow_dispatch(B, A, s)
  local n = B:nrow()
  if s == 0 then
    array_clear(B)
    for i=1,n do B[{i,i}] = 1 end
  elseif s == 1 then
    array_copy_data(B, A)
  else
    local T = pow_recursive(A, s, n)
    array_copy_data(B, T)
  end
end

-- TODO: Use SVD decomposition for large s and allow positive real s.
local function pow(B, A, s)
  same_type_2_check(B, A)
  dimensions_mat_square_check(B, A)
  if s < 0 or floor(s) ~= s then
    error('NYI: matrix exponentiation supported only for nonnegative integers')
  end
  pow_dispatch(B, A, s)
  B:_stack_clear()
end

-- Join ------------------------------------------------------------------------
local function rep(what, first, last, sep)
  sep = sep or ', '
  local increment = last >= first and 1 or -1
  local o = { }
  for i=first,last,increment do
    o[#o + 1] = what:gsub('@', i)
  end
  return table.concat(o, sep)
end

local concat_code = template([[
local setmetatable = setmetatable

local concat_n_mt = {
  _new = function(self, n, r, c)
    return self[1]:_new(n, r, c)
  end,
  nrow = function(self)
    return self[1]:nrow()
  end,
  ncol = function(self)
    local nc = 0
    for i=1,self[0] do
      nc = nc + self[i]:ncol()
    end
    return nc
  end,
  elementct = function(self)
    return self[1]:elementct()
  end,
  _concat_dispatch = function(self, lhs)
    local na = self[0]
    self[na + 1] = lhs
    self[0] = na + 1
    return self
  end,
  _copy_into = function(self, out, offset)
    local na, nr = self[0], self[1]._r
    for r=1,nr do
      for a=na,1,-1 do
        local nc = self[a]._c
        for c=1,nc do
          out._p[offset + c - 1] = self[a]._p[(r-1)*nc + c - 1]
        end
        offset = offset + nc
      end
    end
    return offset
  end,
}
concat_n_mt.__index = concat_n_mt

| for N=JOIN_UNROLL,2,-1 do
local concat_${N}_mt = {
  _new = function(self, n, r, c)
    return self[1]:_new(n, r, c)
  end,
  nrow = function(self)
    return self[1]:nrow()
  end,
  ncol = function(self)
    return ${R('self[@]:ncol()', 1, N, ' + ')}
  end,
  elementct = function(self)
    return self[1]:elementct()
  end,
  _concat_dispatch = function(self, lhs)
| if N == JOIN_UNROLL then
    self[0] = ${N + 1}
    return setmetatable({ ${R('self[@]', 1, N)}, lhs }, concat_n_mt)
| else
    return setmetatable({ ${R('self[@]', 1, N)}, lhs }, concat_${N + 1}_mt)
| end
  end,
  _copy_into = function(self, out, offset)
    for r=1,self[1]:nrow() do
| for I=N,1,-1 do
      local nc = self[${I}]._c
      for c=1,nc do
        out._p[offset + c - 1] = self[${I}]._p[(r-1)*nc + c - 1]
      end
      offset = offset + nc
| end
    end
    return offset
  end,
}
concat_${N}_mt.__index = concat_${N}_mt

| end
return concat_2_mt
]])({ JOIN_UNROLL = JOIN_UNROLL, R = rep })

local concat_2_mt = assert(loadstring(concat_code))()

local join_code = template([[
local select = select
local error = error

local function join_1(x1)
  local nr, nc = x1:nrow(), x1:ncol()
  local a = x1:_new(nr*nc, nr, nc)
  x1:_copy_into(a, 0)
  return a
end

| for N=2,JOIN_UNROLL do
local function join_${N}(${R('x@', 1, N)})
  local nr, nc, ct = x1:nrow(), x1:ncol(), x1:elementct()
  if ${R('x@:elementct() ~= ct', 2, N, ' or ')} then
    error('constant element type required')
  end
  if ${R('x@:ncol() ~= nc', 2, N, ' or ')} then
    error('constant number of columns required')
  end
  nr = nr + ${R('x@:nrow()', 2, N, ' + ')}
  local a = x1:_new(nr*nc, nr, nc)
  local offset = 0
| for I=1,N do
  offset = x${I}:_copy_into(a, offset)
| end
  return a
end

| end
local function join_n(n, ...)
  local arg = { ... }
  local nr, nc, ct = arg[1]:nrow(), arg[1]:ncol(), arg[1]:elementct()
  for i=2,n do
    if arg[i]:elementct() ~= ct then
      error('constant element type required')
    end
    if arg[i]:ncol() ~= nc then
      error('constant number of columns required')
    end
    nr = nr + arg[i]:nrow()
  end
  local a = arg[1]:_new(nr*nc, nr, nc)
  local offset = 0
  for i=1,n do
    offset = arg[i]:_copy_into(a, offset)
  end
  return a
end

return function(...)
  local n = select('#', ...)
  if n == 1 then
    return join_1(...)
| for I=2,JOIN_UNROLL do
  elseif n == ${I} then
    return join_${I}(...)
| end
  else
    return join_n(n, ...)
  end
end
]])({ JOIN_UNROLL = JOIN_UNROLL, R = rep })

local join = assert(loadstring(join_code))()

-- Array -----------------------------------------------------------------------
local array_struct = 'struct { int32_t _n, _r, _c; $* _p; $ _v[?]; }'

local function unsupported_element_ct(self)
  error('operation not supported for element type '..tostring(self:elementct()))
end

local function new_array_ct(element_ct, element_copy)
  local array_mt
  array_mt = {
    new = function(self)
      return array_alloc(self, self._n, self._r, self._c)
    end,
    copy = function(self)
      local a = self:new()
      array_copy_data(a, self)
      return a
    end,
    _new = function(self, n, r, c)
      return array_alloc(self, n, r, c)
    end,
    _copy_into = function(self, out, offset)
      array_copy_data_offset(out, self, offset)
      return offset + self._n
    end,
    _concat_dispatch = function(self, lhs) -- Concatenating two array_ct.
      return setmetatable({ [0] = 2, self, lhs }, concat_2_mt)
    end,
    __concat = function(lhs, rhs)
      if lhs:nrow() ~= rhs:nrow() then
        error('constant number of rows required')
      end
      same_type_2_check(lhs, rhs)
      return rhs:_concat_dispatch(lhs)
    end,
    sub = function(self, f, l)
      if f < 1 or f - 1 > l or l > self._n then
        error('out of bounds first: '..f..', last: '..l..', length: '..self._n)
      end
      if self._n ~= 0 and self._c ~= 1 then
        error('single-column array required')
      end
      local a = array_alloc(self, l - f + 1, l - f + 1, 1)
      array_copy_data_offset(a, self, f - 1)
      return a
    end,
    __len = function(self)
      return self._n
    end,
    nrow = function(self)
      return self._r
    end,
    ncol = function(self)
      return self._c
    end,
    __index = element_copy and function(self, k)
      if type(k) == 'number' then
        if k < 1 or k > self._n then
          error('out of bounds index: '..k..', length: '..self._n)
        end
        return element_copy(self._p[k-1])
      elseif type(k) == 'table' then
        local r, c = k[1], k[2]
        if r < 1 or r > self._r then
          error('out of bounds row: '..r..', number of rows: '..self._r)
        end
        if c < 1 or c > self._c then
          error('out of bounds column: '..c..', number of columns: '..self._c)
        end
        return element_copy(self._p[(r-1)*self._c + (c-1)])
      else    
        return array_mt[k]
      end
    end or function(self, k)
      if type(k) == 'number' then
        if k < 1 or k > self._n then
          error('out of bounds index: '..k..', length: '..self._n)
        end
        return self._p[k-1]
      elseif type(k) == 'table' then
        local r, c = k[1], k[2]
        if r < 1 or r > self._r then
          error('out of bounds row: '..r..', number of rows: '..self._r)
        end
        if c < 1 or c > self._c then
          error('out of bounds column: '..c..', number of columns: '..self._c)
        end
        return self._p[(r-1)*self._c + (c-1)]
      else    
        return array_mt[k]
      end
    end,
    __newindex = element_copy and function(self, k, v)
      if type(k) == 'number' then
        if k < 1 or k > self._n then
          error('out of bounds index: '..k..', length: '..self._n)
        end
        self._p[k-1] =  element_copy(v)
      elseif type(k) == 'table' then
        local r, c = k[1], k[2]
        if r < 1 or r > self._r then
          error('out of bounds row: '..r..', number of rows: '..self._r)
        end
        if c < 1 or c > self._c then
          error('out of bounds column: '..c..', number of columns: '..self._c)
        end
        self._p[(r-1)*self._c + (c-1)] = element_copy(v)
      end
    end or function(self, k, v)
      if type(k) == 'number' then
        if k < 1 or k > self._n then
          error('out of bounds index: '..k..', length: '..self._n)
        end
        self._p[k-1] = v
      elseif type(k) == 'table' then
        local r, c = k[1], k[2]
        if r < 1 or r > self._r then
          error('out of bounds row: '..r..', number of rows: '..self._r)
        end
        if c < 1 or c > self._c then
          error('out of bounds column: '..c..', number of columns: '..self._c)
        end
        self._p[(r-1)*self._c + (c-1)] = v
      end
    end,
    totable = function(self)
      local o = { }
      for i=1,self:nrow() do
        o[i] = { }
        for j=1,self:ncol() do
          o[i][j] = self[{i, j}]
        end
      end
      return o
    end,
    __tostring = function(self)
      local o = { }
      for i=1,self:nrow() do
        o[i] = { }
        for j=1,self:ncol() do
          o[i][j] = width(self[{i, j}])
        end
        o[i] = table.concat(o[i], ",")
      end
      return table.concat(o, "\n")
    end,
    elementct = function()
      return element_ct
    end,
    data = function(self)
      return self._p
    end,
  }

  local element_ct_id = tonumber(element_ct)

  local blas_algo = blas_element_ct[element_ct_id]
  if blas_algo then
    local stack_ct = new_mem_stack_ct(element_ct)
    local maxsize = STACK_PER_ELEMENT_BUFFER/ffi.sizeof(element_ct)
    local stack = stack_ct(maxsize)

    array_mt._stack_array = function(self, n, r, c)
      local p = stack:request(n)
      return p and array_map(self, n, r, c, p) or array_alloc(self, n, r, c)
    end
    array_mt._stack_clear = function()
      stack:clear()
    end
    array_mt._gemm = blas_algo.gemm
    array_mt._gemv = blas_algo.gemv
  else
    array_mt._stack_array = unsupported_element_ct
    array_mt._stack_clear = unsupported_element_ct
    array_mt._gemm        = unsupported_element_ct
  end
  
  local ct = ffi.typeof(array_struct, element_ct, element_ct)
  return ffi.metatype(ct, array_mt)
end

-- Typeof ----------------------------------------------------------------------

-- To preserve value semantics.
local allowed_element_ct = { }

local diff = require 'sci.diff'

for ct_name in pairs{
  bool              = true,
  char              = true,
  int8_t            = true,
  int16_t           = true,
  int32_t           = true,
  int64_t           = true,
  uint8_t           = true,
  uint16_t          = true,
  uint32_t          = true,
  uint64_t          = true,
  float             = true,
  double            = true,
  ['complex float'] = true,
  complex           = true,
  [diff.dn]         = true,
} do
  local ct_id = tonumber(ffi.typeof(ct_name))
  allowed_element_ct[ct_id] = true
end

local alg_element_ct = { }

local function alg_typeof(element_ct)
  element_ct = ffi.typeof(element_ct) -- Allow for strings, now it's ctype.
  local element_ct_id = tonumber(element_ct)
  if not allowed_element_ct[element_ct_id] then
    error('element type "'..tostring(element_ct)..'" not allowed')
  end
  
  if alg_element_ct[element_ct_id] then
    return alg_element_ct[element_ct_id]
  end

  local is_diff_dn = element_ct == diff.dn
  local array_ct = new_array_ct(element_ct, is_diff_dn and element_ct)

  local function vec(n)
    if n < 0 then
      error('length '..n..' is negative')
    end
    return array_alloc(array_ct, n, n, 1)
  end

  local function mat(r, c)
    if r < 0 then
      error('number of rows '..r..' is negative')
    end
    if c < 0 then
      error('number of columns '..c..' is negative')
    end
    return array_alloc(array_ct, r*c, r, c)
  end

  local function tovec(t)
    if type(t) ~= 'table' then
      error('table argument expected, got '..type(t))
    end
    local n = #t
    local a = vec(n)
    for i=1,n do
      a[i] = t[i]
    end
    return a
  end

  local function tomat(t)
    if type(t) ~= 'table' then
      error('table argument expected, got '..type(t))
    end
    local r, c = #t, #t > 0 and #t[1] or 0
    local a = mat(r, c)
    for i=1,r do
      for j=1,c do
        if #t[i] ~= c then
          error('all rows of the table must have the same number of elements')
        end
        a[{i, j}] = t[i][j]
      end
    end
    return a
  end

  local alg = { 
    vec = vec, 
    mat = mat,
    tovec = tovec,
    tomat = tomat,
    arrayct = array_ct,
  }

  alg_element_ct[element_ct_id] = alg
  return alg_element_ct[element_ct_id]
end

--------------------------------------------------------------------------------

local alg_double = alg_typeof('double')

return {
  typeof = alg_typeof,

  vec = alg_double.vec,
  mat = alg_double.mat,
  tovec = alg_double.tovec,
  tomat = alg_double.tomat,
  arrayct = alg_double.arrayct,

  join = join,

  mul = mul,
  pow = pow,
}

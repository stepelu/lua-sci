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

Requirements: 
-------------
* never return a view associated to some data:
  + if table then cannot provide indexing with same performance of originating
    object as hash part not empty. Storing in [0] is not 'safe' in terms of 
    out-of bounds checking ([0] can be accidentally overwritten). Weak tables 
    defeat allocation sinking.
  + thus cdata for performance, requires either weak table or __gc for lifetime 
    management of originating object => cannot be sunk. Caching is impractical 
    and 'strong' caching results in a 'self reference problem' meaning neither 
    the views nor the originating object are ever collected.
  + even if views could be made efficient, we would still have to restrict their
    use in set() because of next point (always using extra copy really is 
    inefficient).    
* results must always be correct even in presence of aliasing
* allow for easy mapping to BLAS (or Eigen) primitives
* have a clear, manageable, compact implementation which cover most use cases 
  and is efficient

Conclusions:
------------
* only works with a linearly representable (potentially with stride) algebra 
  objects => almost all operation can be computed with a single loop => simple
  code and efficient implementation. Excluded:
  + A <- B*x (BLAS)
  + A <- B*C (BLAS)
  + A <- x*x:t() (BLAS)
  + transposed to not transposed matrices
* do not allow views but selectors
* avoid specializing expressions per type
  
Id tables:
----------
The minimum requirement for identifying uniquely an expression is a string
consisting of only result names 'M + M'. With an expression parser this would be
all we need and it's what is used for caching.
We augment this information with the corresponding element position which is 
always increasing left-to-right for a simpler logic 'M<1> + M<2>'. Notice that
in [M<1> + M<2>]*[M<3> + M<4>] both sub-expressions in brackets correspond to 
'M + M'. For variables (terminal nodes in expressions) we omit positional info.
We aggregate this information (esp) together with other other fields in a (id)
table:
* res : result of the expressions, used to determine una-op and bin-op case.
* tra : transposed or not, cannot easily be obtained otherwise, [V*t(V)] is a 
        not transposed matrix; same use as res.
* elw : element-wise operator or not, for convenience and symmetry wrt tra.
* op  : operator (not necessarily a string), used to determine when grouping
        via () is required based on operator precedence and fixity.
* tbe : needs to be evaluated, i.e. esp contains a a [] pair.
* env : environment associated with expression, needed for element-wise 
        functional expressions.
We refer to x, y as the first and second argument to a binary operator.
 
Aliasing notes:
---------------
Vector issues when aliasing is present:
* for i=2,#x do x[i] = x[i-1] end --> x[2] = x[1], x[3] = x[2]
Matrix issues when aliasing is present:
* A <- A*B
* A <- A^n
Cannot alias in this API:
* matrices with vectors
* objects with different element types
* objects of different sizes
We always make a copy (on a buffer for 'small' objects) whenever aliasing could
be a problem. This also helps to simplify logic as BLAS kernels cannot operate
on expressions.

Operations:
-----------
reduce: maxel, minel, sum, prod, argmax, argmin.
fill
set

Selectors:
----------
They apply to reduce, fill, set (not to inner_prod or comparison operators).
Vectors:
* range
Matrices:
* row, col, diag

Reduce:
Argmin and argmax return index for original vector, really are 'restrictors'.
v = max(x, range(2, 4))
v = x:max(range(2, 4))
v = sum(A, diag()) 
v = A:sum(diag())
x:set(sum(A, byrow()))
x:set(A:max(bycol(2,4))) -- From col 2 to col 4.

Fill:
x:fill(c, range(2, 4))
A:fill(c, col(3))

Set: principle is dimensionality reduction, applied to which of the two is of
     higher dimensionality.
Vector-vector: 
x, y = vec(5), vec(3)
x:set(y, range(2, 4)) -- x[2:4] = y
y:set(x, range(2, 4)) -- y = x[2:4]
Matrix-matrix: no selectors.
Matrix-vector: no ambiguity, only matrix selectors are allowed.
A:set(2*x, row(3))
x:set(A/2, row(3))

Examples of views issues (all these may result in aliasing problems, either
forbid them, but it's unintuitive, or warn, but dangerous, or always do copy, 
but kills performance => do without them).
x:range(1,2):set(y:range(2,3))
x:set(y:range(1,2)*x)
A:col(1):set(A:col(2) + A:col(3))

BLAS notes:
Not worth trying to embed the scalar in the matrix-matrix and matrix-vector 
multiplication as buffers are used anyway and an extra loop is required to copy 
the result to the lhs (to avoid potential aliasing problems).
------------------------------------------------------------------------------]]

-- TODO: Consider 'struct { int32_t _n; double* _p; double _a[?]; }' for 
-- TODO: vectors (similar one for matrices); just one type for both temporary
-- TODO: and normal objects and for small and large objects. Con is worse 
-- TODO: alias analysis?

-- TODO: better unrolling strategy / loop blocking for array computations.

-- TODO: vec_argmax --> i, v[i].
-- TODO: mat_argmax --> r, c, v[r][c].

-- TODO: use ffi.copy when possible, check correctness.

-- TODO: The following code will be generated for 'A:set((B+C+2)*D)':
-- TODO: ...
-- TODO: local t2 = (e1+e2+e3):_eval(stack)
-- TODO: local t1 = (t2*e4):_eval(stack)
-- TODO: t1._set(y, t1)
-- TODO: This requires the sub-expressions e1*e2 and t1*e3 to be created. 
-- TODO: Better to to this instead:
-- TODO: ...
-- TODO: local t2 = t2_eval(stack, e1, e2, e3)
-- TODO: local t1 = t1_eval(stack, t2, e4)
-- TODO: t1._set(y, t1)

-- TODO: For small sizes generic multiplies are already faster, optimize via 
-- TODO: unrolling and use them instead.

local ffi  = require "ffi"
local xsys = require "xsys"
local math = require "sci.math"
local cfg  = require "sci.alg.cfg"

-- User-definable settings:
local UNROLL = cfg.unroll
local BUFFER = cfg.buffer

local HAS_BLAS, BLAS = pcall(require, "sci.alg.blas")

local double_ct = ffi.typeof("double")

local exec, template = xsys.exec, xsys.template
local merge, join, insert = xsys.table.merge, xsys.table.join, table.insert
local type, setmetatable, select, assert, error, tostring = xsys.from(_G,
     "type, setmetatable, select, assert, error, tostring")
local istype, typeof, sizeof, new, metatype = xsys.from(ffi, 
     "istype, typeof, sizeof, new, metatype ")
local min, max, abs, ceil, floor, step = xsys.from(math, 
     "min, max, abs, ceil, floor, step")
local concat, width = table.concat, xsys.string.width

-- Utility ---------------------------------------------------------------------
local function op_to_str(op)
  if type(op) == "string" then 
    return op
  elseif type(op) == "function" then
    return "f_"..tostring(op):sub(11):gsub("#", "")
  elseif type(op) == "cdata" then
    return "f_"..tostring(op):sub(7,-2):gsub(" ", "")
  else
    error("unexpected type '"..type(op).."', function or cdata expected")
  end
end

local function env_merge(x, y)
  local o = {}
  for k,v in pairs(x) do
    o[k] = v
  end
  for k,v in pairs(y) do
    if type(o[k]) ~= "nil" then
      assert(o[k] == v)
    end
    o[k] = v
  end
  return o
end

local alg_ct_id_cache = setmetatable({ }, { __index = function()
  return "S"
end })

local function id_of(x)
  if type(x) == "cdata" then
    return alg_ct_id_cache[tonumber(typeof(x))]
  elseif type(x) == "number" 
    then return "S" 
  else
    return x._id
  end
end

local atypeof_cache = { }

-- Array Kernels ---------------------------------------------------------------
-- Kernels always assume array (possibly strided) data. This representation is 
-- accessible via :_at() and :_to() for matrices as well.

local alg_pre = { UNROLL = UNROLL }

local ker_template = template([[
return function(n,x${args and ','..args})
  |if init then
  ${'local v = '..init}
  |end
  |if UNROLL == 0 then
  for i=0,n-1 do
    ${elw('i')}
  end
  |else
  |for n=1,UNROLL do
  ${n==1 and 'if' or 'elseif'} n == ${n} then
    |for i=0,n-1 do  
    ${elw(i)}
    |end
  |end
  else
    for i=0,n-1 do
      ${elw('i')}
    end
  end
  |end  
  |if init then
  ${'return v'}
  |end
end
]])

local ker_env = { assert = assert, error = error, max = max, min = min }

local function ker_compile(name, pre, multi)
  pre = merge(alg_pre, pre)
  local ker = exec(ker_template(pre), name, ker_env)
  if not multi then
    return ker
  else
    local elw = pre.elw
    name = name.."_o"  
    pre.args = pre.args and pre.args..",o" or "o"
    pre.elw = function(i) return elw(i.." + o") end
    local ker_o = exec(ker_template(pre), name, ker_env)
    name = name.."s"
    pre.args = pre.args..",s"
    pre.elw = function(i) return elw(i.."*s + o") end
    local ker_os = exec(ker_template(pre), name, ker_env)
    return ker, ker_o, ker_os
  end
end

local ker_maxel, ker_maxel_o, ker_maxel_os = ker_compile("maxel", {
  init = "-1/0", -- Return -1/0 if size is 0.
  elw  = function(i) return "v = max(v, x:_at("..i.."))" end,
}, true) 

local ker_minel, ker_minel_o, ker_minel_os = ker_compile("minel", {
  init = "1/0", -- Return 1/0 if size is 0.
  elw  = function(i) return "v = min(v, x:_at("..i.."))" end,
}, true) 

local ker_sum, ker_sum_o, ker_sum_os = ker_compile("sum", {
  init = "0", -- Return 0 if size is 0.
  elw  = function(i) return "v = v + x:_at("..i..")" end,  
}, true) 

local ker_prod, ker_prod_o, ker_prod_os = ker_compile("prod", {
  init = "1", -- Return 1 if size is 0.
  elw  = function(i) return "v = v * x:_at("..i..")" end,  
}, true) 

local ker_eq = ker_compile("eq", {
  args = "y",
  init = "true", -- Return true if size is 0.
  elw  = function(i) 
    return "if not (x:_at("..i..") == y:_at("..i..")) then v = false end" 
  end,  
}) 

local ker_lt = ker_compile("lt", {
  args = "y",
  init = "true", -- Return true if size is 0.
  elw  = function(i) 
    return "if not (x:_at("..i..") < y:_at("..i..")) then v = false end" 
  end,  
}) 

local ker_le = ker_compile("le", {
  args = "y",
  init = "true",  -- Return true if size is 0.
  elw  = function(i) 
    return "if not (x:_at("..i..") <= y:_at("..i..")) then v = false end" 
  end,  
})

local ker_inner_prod = ker_compile("inner_prod", {
  args = "y",
  init = "0", -- Return 0 if size is 0.
  elw  = function(i) return "v = v + x:_at("..i..")*y:_at("..i..")" end,  
})

local ker_set = ker_compile("set", { 
  args = "y",
  elw = function(i) return "x:_to("..i..", y:_at("..i.."))" end,  
})
local ker_set_xo = ker_compile("set_xo", { 
  args = "y,o",
  elw = function(i) return "x:_to("..i.." + o, y:_at("..i.."))" end,  
})
local ker_set_xos = ker_compile("set_xos", { 
  args = "y,o,s",
  elw = function(i) return "x:_to("..i.."*s + o, y:_at("..i.."))" end,  
})
local ker_set_yo = ker_compile("set_yo", { 
  args = "y,o",
  elw = function(i) return "x:_to("..i..", y:_at("..i.." + o))" end,  
})
local ker_set_yos = ker_compile("set_yos", { 
  args = "y,o,s",
  elw = function(i) return "x:_to("..i..", y:_at("..i.."*s + o))" end,  
})

local ker_fillc, ker_fillc_o, ker_fillc_os = ker_compile("fillc", { 
  args = "c",
  elw = function(i) return "x:_to("..i..", c)" end,
}, true)

local ker_fillf, ker_fillf_o, ker_fillf_os = ker_compile("fillf", { 
  args = "f",
  elw = function(i) return "x:_to("..i..", f())" end,
}, true)

-- Dimension Checks ------------------------------------------------------------
local function check_v(x)
  if x._m then
    error("vector expected, got matrix")
  end
end

local function check_m(x)
  if not x._m then
    error("matrix expected, got vector")
  end
end

local function check_v_v(x, y)
  if x._n ~= y._n then
    error("different sizes: "..x._n.." and "..y._n)
  end
end

local function check_m_m(x, y)
  if x._n ~= y._n or x._m ~= y._m then
    error("different sizes: "..x._n.."x"..x._m.." and "..y._n.."x"..y._m)
  end
end

local function check_mmv(x, y)
  if x._m ~= y._n then
    error("incompatible sizes: "..x._n.."x"..x._m.." and "..y._n)
  end
end

local function check_mmm(x, y)
  if x._m ~= y._n then
    error("incompatible sizes: "..x._n.."x"..x._m.." and "..y._n.."x"..y._m)
  end
end

local function check_square(x)
  if x._n ~= x._m then
    error("matrix "..x._n.."x"..x._m.." is not square")
  end
end

local function check_range(x, sel)
  local f, l, n = sel._f, sel._l, x._n
  if not (1 <= f and f - 1 <= l and l <= n) then -- Allow for 0-size case.
    error("invalid range: first="..f..", last="..l..", #x="..x._n)
  end
end

local function check_row(x, sel)
  if not (1 <= sel._r and sel._r <= x._n) then
    error("out of bounds row "..sel._r..", #rows is "..x._n)
  end
end

local function check_col(x, sel)
  if not (1 <= sel._c and sel._c <= x._m) then
    error("out of bounds col "..sel._c..", #cols is "..x._m)
  end
end

local function check_diag(x, sel)
  check_square(x)
  if not (abs(sel._d) < x._n) then
    error("out of bounds diagonal "..sel._d..", n is "..x._n)
  end
end

-- Selectors and Set -----------------------------------------------------------
local range = typeof("struct { int32_t _f, _l; }")
local row   = typeof("struct { int32_t _r; }")
local col   = typeof("struct { int32_t _c; }")
local diag  = typeof("struct { int32_t _d; }")

local function all_set(x, y, sel)
  if not sel then
    y._set(x, y)
  else
    if istype(range, sel) then
      check_v(x)
      check_v(y)
      local n = sel._l - sel._f + 1
      local o = sel._f - 1
      if x._n > y._n then
        check_range(x, sel)
        if y._n ~= n then
          error("range size must agree with one of the vectors")
        end
        ker_set_xo(n, x, y, o)   
      else
        check_range(y, sel)
        if x._n ~= n then
          error("range size must agree with one of the vectors")
        end
        ker_set_yo(n, x, y, o)   
      end    
    else -- All these must be of mixed matrix - vector type.
      if x._m and y._m then
        error("one of two arguments must be a vector and the other a matrix")
      end
      local lhs, rhs = x, y
      local ker_set_o, ker_set_os
      if x._m then
        ker_set_o, ker_set_os = ker_set_xo, ker_set_xos
        x, y = x, y
      else
        ker_set_o, ker_set_os = ker_set_yo, ker_set_yos
        x, y = y, x
      end
      if istype(row, sel) then
        check_row(x, sel)
        if not (x._m == y._n) then
          error("incompatible sizes")
        end
        local o = (sel._r - 1)*x._m
        ker_set_o(x._m, lhs, rhs, o)
      elseif istype(col, sel) then
        check_col(x, sel)
        if not (x._n == y._n) then
          error("incompatible sizes")
        end
        local o, s = sel._c - 1, x._m
        ker_set_os(x._n, lhs, rhs, o, s)
      elseif istype(diag, sel) then
        check_diag(x, sel)
        local d = sel._d
        if not (x._n - abs(d) == y._n) then
          error("incompatible sizes")
        end
        local o = step(d)*(d + d*x._n) - d*x._n
        local s = x._n + 1
        ker_set_os(x._n - abs(d), lhs, rhs, o, s)
      else
        error("unexpected selector")
      end
    end
  end
end

-- Vector Computations ---------------------------------------------------------
local function vec_eq(x, y) check_v_v(x, y) return ker_eq(x._n, x, y) end
local function vec_lt(x, y) check_v_v(x, y) return ker_lt(x._n, x, y) end
local function vec_le(x, y) check_v_v(x, y) return ker_le(x._n, x, y) end

local function inner_prod(x, y) check_v_v(x, y)
  return ker_inner_prod(x._n, x, y)
end

local function vec_reduce(ker, ker_o)
  return function(x, sel)
    if not sel then
      return ker(x._n, x)
    else
      if not istype(range, sel) then
        error("unexpected selector")
      end
      check_range(x, sel)
      local n, o = sel._l - sel._f + 1, sel._f - 1
      return ker_o(n, x, o)
    end
  end
end

local function vec_reduce2(ker, ker_o)
  return function(x, y, sel)
    if not sel then
      return ker(x._n, x, y)
    else
      if not istype(range, sel) then
        error("unexpected selector")
      end
      check_range(x, sel)
      local n, o = sel._l - sel._f + 1, sel._f - 1
      return ker_o(n, x, y, o)
    end
  end
end

local vec_maxel  = vec_reduce(ker_maxel, ker_maxel_o)
local vec_minel  = vec_reduce(ker_minel, ker_minel_o)
local vec_sum    = vec_reduce(ker_sum,   ker_sum_o)
local vec_prod   = vec_reduce(ker_prod,  ker_prod_o)
local vec_fillc = vec_reduce2(ker_fillc, ker_fillc_o)
local vec_fillf = vec_reduce2(ker_fillf, ker_fillf_o)

local function vec_fill(x, y, sel)
  if type(y) == "function" then
    vec_fillf(x, y, sel)
  else
    vec_fillc(x, y, sel)
  end
end

local function vec_set(x, y) 
  check_v(x) -- We know y is vec.
  check_v_v(x, y) 
  ker_set(x._n, x, y) 
end

local function vec_ker(x, y) ker_set(x._n, x, y) end

-- Matrix Computations ---------------------------------------------------------
local function mat_eq(x, y) check_m_m(x, y) return ker_eq(x._n*x._m, x, y) end
local function mat_lt(x, y) check_m_m(x, y) return ker_lt(x._n*x._m, x, y) end
local function mat_le(x, y) check_m_m(x, y) return ker_le(x._n*x._m, x, y) end

local function mat_reduce(ker, ker_o, ker_os)
  return function(x, sel)
    if not sel then
      return ker(x._n*x._m, x)
    else
      if istype(row, sel) then
        check_row(x, sel)
        local o = (sel._r - 1)*x._m
        return ker_o(x._m, x, o)
      elseif istype(col, sel) then
        check_col(x, sel)
        local o, s = sel._c - 1, x._m
        return ker_os(x._n, x, o, s)
      elseif istype(diag, sel) then
        check_diag(x, sel)
        local d = sel._d
        local o = step(d)*(d + d*x._n) - d*x._n
        local s = x._n + 1
        return ker_os(x._n - abs(d), x, o, s)
      else
        error("unexpected selector")
      end
    end
  end
end

local function mat_reduce2(ker, ker_o, ker_os)
  return function(x, y, sel)
    if not sel then
      return ker(x._n*x._m, x, y)
    else
      if istype(row, sel) then
        check_row(x, sel)
        local o = (sel._r - 1)*x._m
        return ker_o(x._m, x, y, o)
      elseif istype(col, sel) then
        check_col(x, sel)
        local o, s = sel._c - 1, x._m
        return ker_os(x._n, x, y, o, s)
      elseif istype(diag, sel) then
        check_diag(x, sel)
        local d = sel._d
        local o = step(d)*(d + d*x._n) - d*x._n
        local s = x._n + 1
        return ker_os(x._n - abs(d), x, y, o, s)
      else
        error("unexpected selector")
      end    
    end
  end
end

local mat_maxel  = mat_reduce(ker_maxel, ker_maxel_o, ker_maxel_os)
local mat_minel  = mat_reduce(ker_minel, ker_minel_o, ker_minel_os)
local mat_sum    = mat_reduce(ker_sum,   ker_sum_o,   ker_sum_os)
local mat_prod   = mat_reduce(ker_prod,  ker_prod_o,  ker_prod_os)
local mat_fillc = mat_reduce2(ker_fillc, ker_fillc_o, ker_fillc_os)
local mat_fillf = mat_reduce2(ker_fillf, ker_fillf_o, ker_fillf_os)

local function mat_fill(x, y, sel)
  if type(y) == "function" then
    mat_fillf(x, y, sel)
  else
    mat_fillc(x, y, sel)
  end
end

local function mat_set(x, y) 
  check_m(x) -- We know y is mat.
  check_m_m(x, y) 
  ker_set(x._n*x._m, x, y) 
end

local function mat_ker(x, y) ker_set(x._n*x._m, x, y) end

-- Matrix Kernels --------------------------------------------------------------

local function ker_mmm_generic(transA, transB, A, B, C)
  local M, N, K = C:nrow(), C:ncol(), not transA and A:ncol() or A:nrow()
  C:fill(0)
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

-- C <- A*B.
local function ker_mmm(transA, transB)
  return function(x, y)
    local C, A, B = x, y._e1, y._e2
    -- assert(C._p ~= A._p) -- Alias check.
    -- assert(C._p ~= B._p) -- Alias check.
    if HAS_BLAS and istype(double_ct, C.ct) then
      BLAS.dgemm(transA, transB, 1, A, B, 0, C)
    else
      ker_mmm_generic(transA, transB, A, B, C)
    end
  end
end

local function ker_mmv_generic(transA, A, x, y)
  local M, N = #y, not transA and A:ncol() or A:nrow()
  y:fill(0)
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

-- y <- A*x.
local function ker_mmv(transA)
  return function(x, y)
    local y, A, x = x, y._e1, y._e2
    -- assert(x._p ~= y._p) -- Alias check.
    if HAS_BLAS and istype(double_ct, y.ct) then
      BLAS.dgemv(transA, 1, A, x, 0, y)
    else
      ker_mmv_generic(transA, A, x, y)
    end
  end
end

-- Exponentiation by squaring algorithm:
local function rec_matrix_pow(A, n, stack)
  if n == 1 then
    return A
  elseif n == 2 then
    return (A*A):_eval(stack)    
  elseif n == 3 then
    local A2 = (A*A):_eval(stack)
    return (A2*A):_eval(stack)
  elseif n == 4 then
    local A2 = (A*A):_eval(stack)
    return (A2*A2):_eval(stack)      
  elseif n % 2 == 0 then
    local A2 = (A*A):_eval(stack)
    return rec_matrix_pow(A2, n/2, stack)
  else
    local A2 = (A*A):_eval(stack)
    return (rec_matrix_pow(A2, (n - 1)/2, stack)*A):_eval(stack)
  end
end

-- B <- A^n.
-- TODO: Use SVD decomposition for large n, allow positive real n.
local function ker_mps(x, y)
  local B, A, n = x, y._e1, y._e2
  if n < 0 or floor(n) ~= n then
    error("NYI: matrix power supported only for non-negative integers")
  end
  if n == 0 then
    B:fill(0)
    B:fill(1, diag())
  else
    local stack = B:_stack()
    B:_ker(rec_matrix_pow(A, n, stack))
  end
end

-- Expressions -----------------------------------------------------------------
local bin_op, una_op
local __unm, __t, __elw

local function set_arith_ops(mt, id)
  mt.__add = bin_op("+",  id)
  mt.__sub = bin_op("-",  id)
  mt.__mul = bin_op("*",  id)
  mt.__div = bin_op("/",  id)
  mt.__pow = bin_op("^",  id)
  mt.__mod = bin_op("%",  id)
  mt.__unm = __unm
  mt.t     = __t
  mt.elw   = __elw
end

local function set_reductions(mt, res)
  if res == "V" then
    mt.maxel = vec_maxel
    mt.minel = vec_minel
    mt.sum   = vec_sum
    mt.prod  = vec_prod
  elseif res == "M" then
    mt.maxel = mat_maxel
    mt.minel = mat_minel
    mt.sum   = mat_sum
    mt.prod  = mat_prod
  end
end

local function set_order_ops(mt, res)
  if res == "V" then
    mt.__eq = vec_eq
    mt.__lt = vec_lt
    mt.__le = vec_le
  elseif res == "M" then
    mt.__eq = mat_eq
    mt.__lt = mat_lt
    mt.__le = mat_le
  end
end

local op_precedence = setmetatable({
["+"]  = 1,
["-"]  = 1,
["*"]  = 2,
["/"]  = 2,
["%"]  = 2,
["1-"] = 3, -- Unary minus.
["^"]  = 4, -- Right-to-left fixity.
}, { __index = function(self, k)
  return 5  -- Everything else: t(), elwf(), variables.
end })

local esp_cache = {
  ["S"] = {
    esp = "S", 
    res = "S", 
    tra = false,
    elw = false,
    op  = false,
    tbe = false,
    env = { },
  },
  ["V"] = { 
    esp = "V", 
    res = "V", 
    tra = false,
    elw = false,
    op  = false,
    tbe = false,
    env = { },
  },
  ["M"] = {
    esp = "M", 
    res = "M", 
    tra = false,
    elw = false,    
    op  = false,
    tbe = false,
    env = { },
  }
}


-- 'M<1> + M<2>' --> 2.
local function esp_count(esp)
  local _,n = esp:gsub("<%d+>", "")
  return n
end

-- 'M<1> + M<2>' --> 'M<1+n> + M<2+n>'.
local function esp_shift(esp, n)
  return (esp:gsub("<(%d+)>", function(s)
    local i = tonumber(s)
    return "<"..(i + n)..">"
  end))
end

-- 'M<1> + M<2>' --> '<1> + <2>'.
local function esp_pos(esp)
  return (esp:gsub("%u", ""))
end

-- 'M<1> + M<2>' --> e1 + e2.
local function esp_var(esp)
  return (esp:gsub("%u<(%d+)>", "e%1"))
end

-- 'M<1> + M<2>' --> 'M + M'.
local function esp_id(esp)
  return (esp:gsub("<%d+>", ""))
end

local function cache_vars(n)
  local lhs, rhs = {}, {}
  for i=1,n do
    lhs[i], rhs[i] = "e"..i, "y._e"..i
  end
  return "local "..concat(lhs, ",").." = "..concat(rhs, ",")
end

local function expr_eval(self, stack)
  if self._m then
    local x = stack:mat(self._n, self._m)
    self._ker(x, self)
    return x
  else
    local x = stack:vec(self._n)
    self._ker(x, self)
    return x
  end
end

local expr_ntbe_set_template = template([[
return function(x, y)
  local stack = x:_stack()
  stack:clear()
  ${locals}
${steps}
  local final = ${final}
  final._set(x, final)
end
]])

local function esp_ntbe_set(esp)
  local steps, c = {}, 0
  local function process(esp)
    while true do
      local n = 0
      esp, n = esp:gsub("(%b[])", function(s)
        s = s:sub(2,-2)
        c = c + 1
        local mc = c
        insert(steps, "  local t"..mc.." = ("..esp_var(process(s))..
               "):_eval(stack)")
        return "t"..mc -- All these enter the computation of y.
      end)
      if n == 0 then 
        return esp -- No brackets: can be evaluated.
      end 
    end
  end
  local final = esp_var(process(esp))
  return exec(expr_ntbe_set_template({
    locals = cache_vars(esp_count(esp)),
    steps  = concat(steps, "\n"),
    final  = final,
  }), "expr_ntbe_set<"..esp_id(esp)..">", {})
end

local function esp_is_var_or_tvar(esp)
  esp = esp_pos(esp)
  return esp == "<1>" or esp == "(<1>):t()"
end

local function esp_bracket(binop, op, esp, tbe, side)
  assert(side == "left" or side == "right")
  if tbe then
    if esp:sub(1,1) ~= "[" then -- Avoid double square brackets: [[  ]].
      return "["..esp.."]"
    else
      return esp
    end
  end
  if op_precedence[binop] > op_precedence[op] then 
    return "("..esp..")"
  elseif op_precedence[binop] < op_precedence[op] then
    return esp
  else
    if binop ~= "^" then -- Left-to-right is default fixity.
      return side == "left" and esp or "("..esp..")"      
    else -- Right-to-left is default fixity.
      return side == "right" and esp or "("..esp..")"
    end
  end
end

local function expr_tnew_var(nvarx, nvary)
  local c, o = 0, {}
  if nvarx == 0 then
    c = c + 1
    o[c] = "_e"..c.."=x"
  else
    for i=1,nvarx do
      c = c + 1
      o[c] = "_e"..c.."=x._e"..i
    end
  end
  if nvary then
    if nvary == 0 then
      c = c + 1
      o[c] = "_e"..c.."=y"
    else
      for i=1,nvary do
        c = c + 1
        o[c] = "_e"..c.."=y._e"..i
      end
    end
  end
  return concat(o, ",")
end

local expr_ct_template = template([[
return function(x, y)
  return setmetatable(${tnew}, mt)
end
]])

local expr_ct_env = { setmetatable = setmetatable }

local function expr_ct_compile(mt, name, pre)
  local env = merge(expr_ct_env, { mt = mt })
  return exec(expr_ct_template(pre), "expr_new<"..name..">", env)
end

-- TODO: Can this be made more efficient?
local function mat_tra_at(self, i)
  local r, c = self._n, self._m
  local j = floor(i/c)
  local k = i % c
  local i = k*r + j
  return self._e1:_at(i)
end

local function expr_at(esp, env)
  if esp == "(M<1>):t()" then
    return mat_tra_at
  else
    local s = esp:gsub("%((.*)%):t%b()", "%1")
    s = s:gsub("(%u)<(%d+)>", function(res, i) 
      local acc = "x._e"..i
      if res ~= "S" then
        acc = acc..":_at(i)"
      end
      return acc
    end)
    return exec("return function(x, i) return "..s.." end", 
      "expr_at<"..esp_id(esp)..">", env)
  end
end

local function expr_tostring(self) 
  return "expression<"..self._id..">" 
end

local function expr_new(op, idx, idy)
  local esp, res, tra, env, ker, set
  local tnew_var, tnew_dim, check
  local elw = false
  local optbe -- Whether [] needs to be added to this op (esp).
  -- Unary operators:
  if not idy then 
    tnew_var = expr_tnew_var(esp_count(idx.esp))
    local espx = #idx.esp > 1 and idx.esp or idx.esp.."<1>"    
    res = idx.res
    if op == "t" then -- Transposition.
      if idx.tra then
        error("nested transposition is not allowed")
      end
      if idx.elw then
        error("cannot mix transposition and element-wise specifiers")
      end
      if idx.res == "S" or idx.res == "M" and espx ~= "M<1>" then
        error("scalars and matrix expressions cannot be transposed")
      end
      tra = true
      esp = "("..espx.."):t()" -- Easier if we week the surrounding ().
      env = idx.env 
      tnew_dim = res == "V" and "_n=x._n" or "_n=x._m,_m=x._n"
    else
     if idx.tra then
       error("transposition not allowed in unary '-' or function application"..
             ", rearrange or group")
     end
     tra = false
      if op == "1-" then -- Unary minus.
        esp = "-"..esp_bracket(op, idx.op, espx, false, "right")
        env = idx.env
      elseif op == "e" then -- Element-wise operation.
        if idx.res ~= "M" then
          error("element-wise specifier only applicable to matrix expressions")
        end
        elw = true
        esp = "("..espx..")"
        env = idx.env
      else -- Function application
        esp = op_to_str(op).."("..espx..")"
        env = env_merge(idx.env, { [op_to_str(op)] = op })
      end
      tnew_dim = res == "V" and "_n=x._n" or "_n=x._n,_m=x._m"
    end
  -- Binary operators:    
  else
    local tbex, tbey -- Whether [] needs to be added to x, y of this op (esp).
    tnew_var = expr_tnew_var(esp_count(idx.esp), esp_count(idy.esp))
    local espx = #idx.esp > 1 and idx.esp or idx.esp.."<1>"  
    local espy = #idy.esp > 1 and idy.esp or idy.esp.."<1>"  
    env = env_merge(idx.env, idy.env)
    
    local resx, trax, resy, tray = idx.res, idx.tra, idy.res, idy.tra
     -- Utility function to simplify case matching in binary operators:
    local function case(opin, resxin, resyin)
      return opin:find(op, 1, true) and resxin:find(resx, 1, true)
                                    and resyin:find(resy, 1, true)
    end
    -- Cases involving transposition:
    if     case("*", "M", "M")        then res, tra = "M", false
      if not idx.elw then
        check    = check_mmm
        tnew_dim = "_n=x._n,_m=y._m"
        tbex = not esp_is_var_or_tvar(espx)
        tbey = not esp_is_var_or_tvar(espy)
        optbe = true
        if not (tbex or tbey) then
          ker = ker_mmm(trax, tray)
        end
      else
        if trax or tray then
          error("transposition not allowed in element-wise matrix product")
        end        
        check    = check_m_m
        tnew_dim = "_n=x._n,_m=x._m"
      end
    elseif case("*", "M", "V")        then res, tra = "V", false
      if tray then
        error("not allowed operation: M*V:t(), allowed: M*V")
      end
      check    = check_mmv
      tnew_dim = "_n=x._n"
      tbex = not esp_is_var_or_tvar(espx)
      tbey = not esp_is_var_or_tvar(espy)
      optbe = true
      if not (tbex or tbey) then        
        ker = ker_mmv(trax)
      end
    elseif case("*", "V", "M")        then
      error("V*M and t(V)*M are both not allowed; for inner product "..
            "V:t()*M*V group the M*V part: V:t()*(M*V)")
    elseif case("*", "V", "V")        then
      if trax and not tray            then return inner_prod
      elseif not trax and tray        then res, tra = "M", false        
        tnew_dim = "_n=x._n,_m=y._n"
        tbex = not esp_is_var_or_tvar(espx)
        tbey = not esp_is_var_or_tvar(espy)
        optbe = true
        error("NYI: matrix via V*V:t()")
      elseif trax and tray then
        error("not allowed operation: V:t()*V:t()")
      else                                 res, tra = "V", false
        check    = check_v_v
        tnew_dim = "_n=x._n"
      end      
    -- Cases not involving transposition:
    else 
      if trax or tray then
        error("transposition not allowed in: "..resx..op..resy)
      end
      tra = false
      -- Matrix power:
      if     case("^", "M", "S")      then res = "M"
        if not idx.elw then
          check    = check_square
          tnew_dim = "_n=x._n,_m=x._m"
          tbex = not esp_is_var_or_tvar(espx)
          optbe = true
          if not tbex then
            if trax then 
              error("NYI: matrix power with transposition") 
            end
            ker = ker_mps
          end
        else
          tnew_dim = "_n=x._n,_m=x._m"
        end
      -- Matrix-matrix ops (not confusing ones):
      elseif case("+-*", "M", "M")    then res = "M"
        check    = check_m_m
        tnew_dim = "_n=x._n,_m=x._m"
      -- Scalar-matrix ops (not confusing ones):
      elseif case("+-*", "S", "M")    then res = "M"
        tnew_dim = "_n=y._n,_m=y._m"
      -- Matrix-scalar ops:      
      elseif case("+-*/%", "M", "S")  then res = "M"
        tnew_dim = "_n=x._n,_m=x._m"
      -- Vector-vector, scalar-vector, vector-scalar ops:
      elseif case("+-/^%", "V", "V")  then res = "V" 
        check    = check_v_v
        tnew_dim = "_n=x._n"
      elseif case("+-*/^%", "S", "V") then res = "V"
        tnew_dim = "_n=y._n"
      elseif case("+-*/^%", "V", "S") then res = "V"
        tnew_dim = "_n=x._n"
      end 
    end    
    if not res then 
      error("not allowed operation: "..resx..op..resy)
    end    
    espx = esp_bracket(op, idx.op, espx, tbex, "left")
    espy = esp_shift(espy, esp_count(espx))    
    espy = esp_bracket(op, idy.op, espy, tbey, "right")
    esp = espx..op..espy   
  end
  esp = optbe and "["..esp.."]" or esp
  local id = esp_id(esp)  
  -- If expression id is cached return it, otherwise build new expression:
  if esp_cache[id] then
    return esp_cache[id].new
  else
    local tbe = esp:find("%b[]") and true or false
    local at
    if not tbe then
      assert(not ker)
      assert(not set)
      set = res == "V" and vec_set or mat_set
      ker = res == "V" and vec_ker or mat_ker
      at = expr_at(esp, env)
    else
      set = set or esp_ntbe_set(esp)      
    end
    assert(set)
    -- Expression metatable:
    local mt = { 
      _id        = id,
      __tostring = expr_tostring,
      _at        = at,
      _ker       = ker,
      _set       = set,
      _eval      = expr_eval,
    }
    mt.__index = mt
    set_arith_ops(mt, id)
    if not tbe then
      set_reductions(mt, res)
      set_order_ops(mt, res)
    end
    
    -- Expression constructor:
    assert(tnew_var ~= nil)
    assert(tnew_dim ~= nil)
    local tnew = "{"..tnew_dim..","..tnew_var.."}"
    local ct = expr_ct_compile(mt, id, {
      tnew = tnew
    })
    local new = check and function(x, y) check(x, y); return ct(x, y) end or ct
    -- Cache result and return it:
    assert(esp ~= nil)
    assert(res ~= nil)
    assert(tra ~= nil)
    assert(elw ~= nil)
    assert(op  ~= nil)
    assert(tbe ~= nil)
    assert(env ~= nil)
    esp_cache[id] = {
      esp = esp,
      res = res,
      tra = tra,
      elw = elw,
      op  = op,
      tbe = tbe,
      env = env,
      -- Extra fields for expressions:
      new = new, -- Constructor when using operators.
    }
    return new
  end
end

local function bin_op_dispatcher(op, idx)
  return setmetatable({}, { 
    __index = function(self, idy)  
      return function(x, y)
        local f = expr_new(op, esp_cache[idx], esp_cache[idy])
        self[idy] = f  -- Cache.
        return f(x, y) -- Invoke.
      end
    end,
  })
end

bin_op = function(op, idx)
  local disp_idy = bin_op_dispatcher(op, idx) 
  local scal_idx
  return function(x, y)
    if id_of(x) == "S" then
      scal_idx = scal_idx or expr_new(op, esp_cache["S"], esp_cache[idx])
      return scal_idx(x, y)
    else
      return disp_idy[id_of(y)](x, y)
    end
  end
end

local function una_op_dispatcher(op)
  return setmetatable({}, { 
    __index = function(self, idx)  
      return function(x)
        local f = expr_new(op, esp_cache[idx])
        self[idx] = f  -- Cache.
        return f(x)    -- Invoke.
      end
    end,
  })
end

una_op = function(op)
  local disp_idx = una_op_dispatcher(op)
  return function(x)
    return disp_idx[id_of(x)](x)
  end
end

__unm = una_op("1-")
__t   = una_op("t")
__elw = una_op("e")

-- Excludes functions with more than 1 argument or that return more than 1 
-- value, constants, math.random, math.randomseed.
local no_math = {
  atan2      = true,
  fmod       = true,
  huge       = true,
  ldexp      = true,
  frexp      = true,
  max        = true,
  min        = true,
  modf       = true,
  pi         = true,
  pow        = true,
  random     = true,
  randomseed = true,
  
  beta       = true,
  logbeta    = true,
}

local amath = {}
for k,v in pairs(math) do
  if not no_math[k] then
    amath[k] = una_op(v)
  end
end

-- Array -----------------------------------------------------------------------
local function array_at(self, i)
  -- assert(0 <= i and i < self._n)
  return self._p[i]
end
local function array_copy_at(self, i)
  -- assert(0 <= i and i < self._n)
  return self._p[i]:copy()
end

local function array_to(self, i, v)
  -- assert(0 <= i and i < self._n)
  self._p[i] = v
end

local function array_stack(self)
  return atypeof_cache[tonumber(typeof(self.ct))]._stack
end

-- Vector ----------------------------------------------------------------------
local function vec_new_std(ct, n, c)
  if not (n >= 0) then
    error("vector size must be non-negative, size="..n)
  end
  -- PERF: Default initialization (compiled)!
  local v = new(ct, n)
  v._n = n
  if (n > 0) and c then
    if type(c) == "function" then
      for i=1,n do
        v[i] = c(i)
      end
    else
      v:fill(c)        
    end
  end -- VLS are automatically zero-filled for default initializer case.
  return v
end

-- TODO: review/improve:
local function vec_dim_any(x)
  local n = x._n
  if n then
    check_v(x)
    return n
  else
    return #x
  end
end
-- TODO: review/improve:
local function vec_set_any(x, t, f, l)
  if t._at then
    for i=f,l do
      x:_to(i-1, t:_at(i-f))
    end
  else
    for i=f,l do
      x:_to(i-1, t[i-f+1] )
    end
  end
end
-- TODO: review/improve:
local function vec_new_aggregate(ct, ...)
  local narg = select("#", ...)
  local a1 = ...
  local n1 = vec_dim_any(a1)
  local n = n1
  for i=2,narg do
    local ai = select(i, ...)
    n = n + vec_dim_any(ai)
  end
  -- PERF: Default initialization (compiled)!
  local v = new(ct, n)
  v._n = n  
  local f, l = 1, n1
  if f <= l then
    vec_set_any(v, a1, f, l)    
  end
  for i=2,narg do
    local ai = select(i, ...)
    f, l = l + 1, l + vec_dim_any(ai)
    if f <= l then
      vec_set_any(v, ai, f, l)      
    end
  end
  return v
end

local function vec_tostring(self, tostringel)
  tostringel = tostringel or tostring
  local o, maxlen = { }, 0
  for i=1,self._n do 
    o[i] = tostringel(self[i])
    maxlen = max(maxlen, #o[i])
  end
  for i=1,self._n do
    local pre = i == 1 and "{ " or ""
    local post = i == self._n and " }" or ","
    o[i] = pre..o[i]..(" "):rep(maxlen - #o[i])..post
  end
  return concat(o, "")
end

local vec_mt = {
  __new = function(ct, ...)
    local n, c = ...
    if type(n) == "number" then
      return vec_new_std(ct, n, c)
    else
      return vec_new_aggregate(ct, ...)
    end
  end,
  copy = function(self, sel)
    if not sel then
      -- PERF: Default initialization (compiled)!
      local v = new(typeof(self), self._n)
      v._n = self._n
      -- copy(v._p, self._p, sizeof(self._p, self._n*self._m))
      v:set(self)
      return v
    elseif istype(range, sel) then
      check_range(self, sel)
      local f = sel._f
      local l = sel._l
      if l == f - 1 then -- Zero-sized case.
        return new(typeof(self), 0)
      else 
        -- PERF: Default initialization (compiled)!
        local v = new(typeof(self), l - (f - 1))
        v._n = l - (f - 1)
        -- copy(v._p, self._p + (f - 1), sizeof(self._p, l - (f - 1)))
        v:set(self, range(f, l))
        return v
      end
    else
      error("unexpected selector")
    end
  end,
  __len = function(self)
    return self._n
  end, 
  __newindex = function(self, i, v)
    if not (1 <= i and i <= self._n) then
      error("out of bounds indexing: i="..i..", #x="..self._n)
    end
    self._p[i-1] = v
  end,  
  totable = function(self)
    local o = { }
    for i=1,self._n do 
      o[i] = self[i]
    end
    return o
  end,
  __tostring = vec_tostring,
  pretty = function(self, chars)
    return vec_tostring(self, function(x) return width(x, chars) end)
  end,
  set    = all_set,
  fill   = vec_fill,
  _id    = "V",
  _set   = vec_set,
  _ker   = vec_ker,
  _to    = array_to,
  _stack = array_stack,
}

set_arith_ops(vec_mt, "V")
set_order_ops(vec_mt, "V")
set_reductions(vec_mt, "V")

local vec_copy_mt = merge(vec_mt)
vec_mt._at = array_at
vec_copy_mt._at = array_copy_at

-- Matrices --------------------------------------------------------------------
local function mat_new_std(ct, n, m, c)
 if not (n >= 0 and m >=0) then
   error("matrix size must be non-negative: n="..n..", m="..m)
 end
  -- PERF: Default initialization (compiled)!
  local v = new(ct, n*m)
  v._n, v._m = n, m
  if (n*m > 0) and c then
    if type(c) == "function" then
      for i=1,n do
        for j=1,m do
          v[i][j] = c(i, j)
        end
      end
    else
      v:fill(c)
    end      
  end -- VLS are automatically zero-filled for default initializer case.
  return v
end

-- TODO: review/improve:
local function mat_dim_any(x)
  local n, m = x._n, x._m
  if n then
    check_m(x)
    return n, m
  else
    return #x, #x[1], false
  end
end
-- TODO: review/improve:
local function mat_set_any(x, t, fn, ln)
  local xn, xm = x._n, x._m  
  -- local tn, tm = mat_dim_any(t)
  -- assert(ln - fn + 1 == tn)
  -- assert(xm == tm)
  if t._at then
    local fi, li = (fn - 1)*xm + 1, ln*xm
    for i=fi,li do
      x:_to(i-1, t:_at(i-fi))
    end
  else
    for r=fn,ln do
      for c=1,xm do
        x[r][c] = t[r-fn+1][c]
      end
    end
  end
end
-- TODO: review/improve:
local function mat_new_aggregate(ct, ...)
  local narg = select("#", ...)
  local a1 = ...
  local n1, m1 = mat_dim_any(a1)
  local n = n1
  for i=2,narg do
    local ai = select(i, ...)
    local ni, mi = mat_dim_any(ai)
    if not (m1 == mi) then
      error("all arguments must have the same number of columns")
    end
    n = n + ni
  end
  -- PERF: Default initialization (compiled)!
  local v = new(ct, n*m1)
  v._n, v._m = n, m1  
  local fn, ln = 1, n1
  if fn <= ln then
    mat_set_any(v, a1, fn, ln)    
  end
  for i=2,narg do
    local ai = select(i, ...)
    local ni, mi = mat_dim_any(ai)
    fn, ln = ln + 1, ln + ni
    if fn <= ln then
      mat_set_any(v, ai, fn, ln)      
    end
  end
  return v
end

local function mat_tostring(self, tostringel)
  tostringel = tostringel or tostring
  local o, maxlencol = { }, { }
  for c=1,self._m do
    maxlencol[c] = 0
  end
  for r=1,self._n do
    local oo = { }
    for c=1,self._m do
      oo[c] = tostringel(self[r][c])
      maxlencol[c] = max(maxlencol[c], #oo[c])
    end
    o[r] = oo
  end
  for r=1,self._n do
    for c=1,self._m do
      local pre  = c == 1       and "{" or ""
      local post = c == self._m and "}" or ","
      o[r][c] = pre..o[r][c]..(" "):rep(maxlencol[c] - #o[r][c])..post
    end
    local pre  = r == 1       and "{" or " "
    local post = r == self._n and "}" or ","
    o[r] = pre..concat(o[r], "")..post
  end
  return concat(o, "\n")
end

local row_mt = {
  __index = function(self, i)
    if not (1 <= i and i <= self._m) then
      error("out of bounds col indexing: i="..i..", #cols="..self._m)
    end
    return self._p[i-1]
  end,
  __newindex = function(self, i, v)
    if not (1 <= i and i <= self._m) then
      error("out of bounds col indexing: i="..i..", #cols="..self._m)
    end
    self._p[i-1] = v
  end,
}

local row_copy_mt = {
  __index = function(self, i)
    if not (1 <= i and i <= self._m) then
      error("out of bounds col indexing: i="..i..", #cols="..self._m)
    end
    return self._p[i-1]:copy()
  end,
  __newindex = function(self, i, v)
    if not (1 <= i and i <= self._m) then
      error("out of bounds col indexing: i="..i..", #cols="..self._m)
    end
    self._p[i-1] = v 
  end,
}

local mat_mt = {
  __new = function(ct, ...)
    local n, m, c = ...
    if type(n) == "number" then
      return mat_new_std(ct, n, m, c)
    else
      return mat_new_aggregate(ct, ...)
    end
  end,
  copy = function(self, sel)
    if not sel then
      -- PERF: Default initialization (compiled)!
      local v = new(typeof(self), self._n*self._m)
      v._n, v._m = self._n, self._m
      -- copy(v._p, self._p, sizeof(self._p, self._n*self._m))
      v:set(self)
      return v
    else
      if istype(row, sel) then
        check_row(self, sel)
        local o = (sel._r - 1)*self._m
        local v = self:_new_vec(self._m)
        ker_set_yo(self._m, v, self, o)
        return v
      elseif istype(col, sel) then
        check_col(self, sel)
        local o, s = sel._c - 1, self._m
        local v = self:_new_vec(self._n)
        ker_set_yos(self._n, v, self, o, s)
        return v
      elseif istype(diag, sel) then
        check_diag(self, sel)
        local d = sel._d
        local o = step(d)*(d + d*self._n) - d*self._n
        local s = self._n + 1
        ker_set_yos(self._n - d, v, self, o, s)
        return v
      else
        error("unexpected selector")
      end
    end
  end,
  nrow = function(self)
    return self._n
  end,
  ncol = function(self)
    return self._m
  end, 
  totable = function(self)
    local o = { }
    for r=1,self._n do
      local oo = { }
      for c=1,self._m do
        oo[c] = self[r][c]
      end
      o[r] = oo
    end
    return o
  end,  
  __tostring = mat_tostring,
  pretty = function(self, chars)
    return mat_tostring(self, function(x) return width(x, chars) end)
  end,
  set    = all_set,
  fill   = mat_fill,
  _id    = "M",
  _set   = mat_set,
  _ker   = mat_ker,
  _to    = array_to,
  _stack = array_stack,
}

set_arith_ops(mat_mt, "M")
set_order_ops(mat_mt, "M")
set_reductions(mat_mt, "M")

local mat_copy_mt = merge(mat_mt)
mat_mt._at = array_at
mat_copy_mt._at = array_copy_at

-- Typeof ----------------------------------------------------------------------
local function atypeof(ct, copyonidx)
  ct = typeof(ct)
  local ctn = tonumber(ct)
  if atypeof_cache[ctn] then
    return atypeof_cache[ctn]
  else    
    local row_ct = typeof("struct { int32_t _m; $* _p; }", ct)
    local vec_ct = typeof("struct { int32_t _n; $ _p[?]; }", ct)
    local mat_ct = typeof("struct { int32_t _n, _m; $ _p[?]; }", ct)

    local row = metatype(row_ct, copyonidx and row_copy_mt or row_mt)
    
    local vec_mt_ct = merge(copyonidx and vec_copy_mt or vec_mt, { ct = ct })
    vec_mt_ct.__index = copyonidx and function(self, k)
      if type(k) == "number" then
        if not (1 <= k and k <= self._n) then
          error("out of bounds indexing: i="..k..", #x="..self._n)
        end
        return self._p[k-1]:copy()
      else    
        return vec_mt_ct[k]
      end
    end or function(self, k)
      if type(k) == "number" then
        if not (1 <= k and k <= self._n) then
          error("out of bounds indexing: i="..k..", #x="..self._n)
        end
        return self._p[k-1]
      else    
        return vec_mt_ct[k]
      end
    end
         
    local vec = metatype(vec_ct, vec_mt_ct)
    alg_ct_id_cache[tonumber(typeof(vec))] = "V"

    local mat_mt_ct = merge(copyonidx and mat_copy_mt or mat_mt, { ct = ct })
    mat_mt_ct.__index = function(self, k)
      if type(k) == "number" then
        if not (1 <= k and k <= self._n) then
          error("out of bounds row indexing: i="..k..", #rows="..self._n)
        end
        return row(self._m, self._p + (k-1)*self._m)
      else    
        return mat_mt_ct[k]
      end
    end
    mat_mt_ct._new_vec = function(self, ...)
      return vec(...)
    end
    
    local mat = metatype(mat_ct, mat_mt_ct)
    alg_ct_id_cache[tonumber(typeof(mat))] = "M"      
    
    local vep_ct = typeof("struct { int32_t _n; $* _p; }", ct)
    local map_ct = typeof("struct { int32_t _n, _m; $* _p; }", ct)
    local vep = metatype(vep_ct, vec_mt_ct)
    local map = metatype(map_ct, mat_mt_ct)
    alg_ct_id_cache[tonumber(typeof(vep))] = "V"
    alg_ct_id_cache[tonumber(typeof(map))] = "M"
    
    local elsize = sizeof(ct)
    local stack_n = floor(BUFFER/elsize)
    local stack_ct = typeof("struct { int32_t _use; $ _ptr[?]; }", ct)
    
    local stack_mt = {
      clear = function(self)
        self._use = 0
      end,        
      vec = function(self, n)
        if self._use + n <= stack_n then -- Use stack.
          local v = new(vep, n, self._ptr + self._use)
          self._use = self._use + n
          return v
        else
          return vec(n)
        end      
      end,
      mat = function(self, n, m)
        if self._use + n*m <= stack_n then -- Fast path.
          local v = new(map, n, m, self._ptr + self._use)
          self._use = self._use + n*m
          return v
        else
          return mat(n, m)
        end      
      end
    }
    stack_mt.__index = stack_mt
    
    local stack_ct = ffi.metatype(stack_ct, stack_mt)
    local stack = stack_ct(stack_n)
        
    local o = {
      vec       = vec,
      mat       = mat,
      _stack    = stack,
    }
    atypeof_cache[ctn] = o
    return o
  end
end

local function amaxel(x)
  return x:maxel()
end
local function aminel(x)
  return x:minel()
end
local function asum(x)
  return x:sum()
end
local function aprod(x)
  return x:prod()
end

local adouble = atypeof("double")

return {
  typeof = atypeof,
  elwf   = una_op,
  
  vec    = adouble.vec,
  mat    = adouble.mat,
  
  range  = range,
  row    = row,
  col    = col,
  diag   = diag,
  
  maxel  = amaxel,
  minel  = aminel,
  sum    = asum,
  prod   = aprod,
  
  math   = amath,
  
  blas   = HAS_BLAS and BLAS or false,
}

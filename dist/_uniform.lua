--------------------------------------------------------------------------------
-- Uniform statistical distribution.
--
-- Copyright (C) 2011-2014 Stefano Peluchetti. All rights reserved.
--
-- Features, documentation and more: http://www.scilua.org .
--
-- This file is part of the SciLua library, which is released under the MIT 
-- license: full text in file LICENSE.TXT in the library's root folder.
--------------------------------------------------------------------------------

local ffi   = require "ffi"

local log = math.log

local uni_mt = {
  __new = function(ct, a, b)
    if not a or not b then
      error("distribution parameters must be set at construction")
    end
    if not (a < b) then
      error("a < b is required, a is "..a..", b is "..b)
    end
    return ffi.new(ct, a, b)
  end,
  copy = function(self)
    return ffi.new(ffi.typeof(self), self)
  end,
  range = function(self)
    return self._a, self._b
  end,
  pdf = function(self, x)
    return 1/(self._b - self._a)
  end,
  logpdf = function(self, x)
    return -log(self._b - self._a)
  end,
  mean = function(self)
    return 0.5*(self._a + self._b)
  end,
  var = function(self)
    return (self._b - self._a)^2/12
  end,
  sample = function(self, rng)
    return self._a + (self._b - self._a)*rng:sample()
  end,
}
uni_mt.__index = uni_mt

local dist = ffi.metatype("struct { double _a, _b; }", uni_mt)

-- Multi variate uniform distribution:
local mvuni_mt = {
  sample = function(self, rng, x)
    for i=1,#x do x[i] = rng:sample() end    
    x:set(self._a + (self._b - self._a)*x)
  end,
}
mvuni_mt.__index = mvuni_mt

local function mvdist(a, b)
  if not a or not b then
    error("distribution parameters must be set at construction")
  end
  if #a ~= #b then
    error("a and b must have the same size, #a is "..#a..", #b is "..#b)
  end
  if not (a < b) then 
    error("a < b is required, a is "..a..", b is "..b)
  end
  return setmetatable({ _a = a:copy(), _b = b:copy() }, mvuni_mt)
end

return {
  dist   = dist, 
  mvdist = mvdist,
}

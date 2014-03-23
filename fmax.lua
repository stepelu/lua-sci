--------------------------------------------------------------------------------
-- Function maximization module.
--
-- Copyright (C) 2011-2014 Stefano Peluchetti. All rights reserved.
--
-- Features, documentation and more: http://www.scilua.org .
--
-- This file is part of the SciLua library, which is released under the MIT 
-- license: full text in file LICENSE.TXT in the library's root folder.
--------------------------------------------------------------------------------

-- TODO: Make sure in each function minimization specific module that errors
-- TODO: are not specific to minimization/maximization.

local fmin = require "sci.fmin"

local function unm1arg(f)
  return function(x)
    return -f(x)
  end
end

-- Results are always xmin, fmin, [optional results, like xval, fval].
local function tofmaxresults(...)
  local ret, n = { ... }, select("#", ...)
  if type(ret[1]) == "nil" then -- Handle nil, errorstring case.
    return nil, ret[2]
  else
    ret[2] = - ret[2]
    return unpack(ret, 1, n)
  end
end

-- Minimize -f(x) to maximize f(x).
local function tofmaxalgo(fminalgo)
  return function(f, ...) -- Objective function is always first argument.
    local unmf = unm1arg(f)
    return tofmaxresults(fminalgo(unmf, ...))
  end
end

local fmax = { }

for k,v in pairs(fmin) do
  if type(v) ~= "function" then
    fmax[k] = v
  else
    fmax[k] = tofmaxalgo(v)
  end
end

return fmax

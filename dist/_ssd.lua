--------------------------------------------------------------------------------
-- Shifted and scaled distributions module.
--
-- Copyright (C) 2011-2014 Stefano Peluchetti. All rights reserved.
--
-- Features, documentation and more: http://www.scilua.org .
--
-- This file is part of the SciLua library, which is released under the MIT 
-- license: full text in file LICENSE.TXT in the library's root folder.
--------------------------------------------------------------------------------

local log, abs = math.log, math.abs

-- Template for shifted and scaled distributions:
-- Distribution, shift, scale = _d, _s, _m.
local ssd_mt = {
  range = function(self)
    local xl, xu = _d:range()
    return xl*self._m + self._s, xu*self._m + self._s
  end,
  pdf = function(self, x)
    return self._d:pdf((x - self._s)/self._m)/self._m
  end,
  logpdf = function(self, x)
    return -log(self._m) + self._d:logpdf((x - self._s)/self._m)
  end,
  mean = function(self)
    return self._d:mean()*self._m + self._s
  end,
  var = function(self)
    return self._d:var()*(self._m^2)
  end,
  absmoment = function(self, mm)
    return self._d:absmoment(mm)*abs(self._m)^mm
  end,
  sample = function(self, rng)
    return self._d:sample(rng)*self._m + self._s
  end,
}

return {
  mt = ssd_mt,
}
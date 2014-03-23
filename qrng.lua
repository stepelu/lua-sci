--------------------------------------------------------------------------------
-- Quasi random number generators module.
--
-- Copyright (C) 2011-2014 Stefano Peluchetti. All rights reserved.
--
-- Features, documentation and more: http://www.scilua.org .
--
-- This file is part of the SciLua library, which is released under the MIT 
-- license: full text in file LICENSE.TXT in the library's root folder.
--------------------------------------------------------------------------------

local sobol = require "sci.qrng._sobol"

return {
  std   = sobol.qrng,
  sobol = sobol.qrng,
}

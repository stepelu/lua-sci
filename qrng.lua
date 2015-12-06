--------------------------------------------------------------------------------
-- Quasi random number generators module.
--
-- Copyright (C) 2011-2015 Stefano Peluchetti. All rights reserved.
--------------------------------------------------------------------------------

local sobol = require "sci.qrng._sobol"

return {
  std   = sobol.qrng,
  sobol = sobol.qrng,
}

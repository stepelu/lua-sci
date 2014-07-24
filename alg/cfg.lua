--------------------------------------------------------------------------------
-- Configuration for sci.alg module.
--
-- Copyright (C) 2011-2014 Stefano Peluchetti. All rights reserved.
--
-- Features, documentation and more: http://www.scilua.org .
--
-- This file is part of the SciLua library, which is released under the MIT 
-- license: full text in file LICENSE.TXT in the library's root folder.
--------------------------------------------------------------------------------

return {
  unroll = 16,  -- Non-negative number, 0 to disable.
  buffer = 2e6, -- Bytes, non-negative number, 0 to disable. 
  blas = {
    enable = true,
    banner = false,
  },
}
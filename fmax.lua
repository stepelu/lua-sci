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

local function tofmax(optim)
  return function(...)
    return optim(-1, ...)
  end
end

return {
  de = tofmax(require("sci.fmin._de").optim),
}
--------------------------------------------------------------------------------
-- Root finding module.
--
-- Copyright (C) 2011-2014 Stefano Peluchetti. All rights reserved.
--
-- Features, documentation and more: http://www.scilua.org .
--
-- This file is part of the SciLua library, which is released under the MIT 
-- license: full text in file LICENSE.TXT in the library's root folder.
--------------------------------------------------------------------------------

local newtonm = require "sci.root._newtonm"

return {
  newton  = newtonm.newton,
  halley  = newtonm.halley,
  ridders = require("sci.root._ridders").root,
}

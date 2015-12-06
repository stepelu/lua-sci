--------------------------------------------------------------------------------
-- Root finding module.
--
-- Copyright (C) 2011-2015 Stefano Peluchetti. All rights reserved.
--------------------------------------------------------------------------------

local newtonm = require "sci.root._newtonm"

return {
  newton  = newtonm.newton,
  halley  = newtonm.halley,
  ridders = require("sci.root._ridders").root,
}

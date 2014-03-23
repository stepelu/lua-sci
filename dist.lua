--------------------------------------------------------------------------------
-- Statistical distributions module.
--
-- Copyright (C) 2011-2014 Stefano Peluchetti. All rights reserved.
--
-- Features, documentation and more: http://www.scilua.org .
--
-- This file is part of the SciLua library, which is released under the MIT 
-- license: full text in file LICENSE.TXT in the library's root folder.
--------------------------------------------------------------------------------

return {
  exponential = require("sci.dist._exponential").dist,
  normal      = require("sci.dist._normal").dist,
  lognormal   = require("sci.dist._lognormal").dist,
  gamma       = require("sci.dist._gamma").dist,
  beta        = require("sci.dist._beta").dist,
  student     = require("sci.dist._student").dist,
  uniform     = require("sci.dist._uniform").dist,
  
  mvuniform   = require("sci.dist._uniform").mvdist,
}

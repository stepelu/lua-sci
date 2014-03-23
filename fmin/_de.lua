--------------------------------------------------------------------------------
-- Differential evolution algorithm module.
--
-- Copyright (C) 2011-2014 Stefano Peluchetti. All rights reserved.
--
-- Features, documentation and more: http://www.scilua.org .
--
-- This file is part of the SciLua library, which is released under the MIT 
-- license: full text in file LICENSE.TXT in the library's root folder.
--------------------------------------------------------------------------------

-- Here implemented is the differential evolution algorithm presented in the 
-- paper: "Differential evolution algorithm with strategy adaptation for global 
-- numerical optimization", 2009.
-- The following modifications have been performed: 
-- + in the paper a moving window of LP generations is used to update the 
--   parameters; we simply re-update the parameters every LP generations using 
--   the last LP generations (no overlapping), provided that at least 100 
--   successful mutations have been obtained (i.e. 100 samples on which to base
--   the estimates)
-- + only one CR vector (common for all strategies), instead of 3 as in the 
--   paper
-- + different adaptation algorithm for CR, see sample_CR and initialization of 
--   CRmu, CRsigma

local xsys = require "xsys"
local alg  = require "sci.alg"
local prng = require "sci.prng"
local dist = require "sci.dist"
local math = require "sci.math"
local stat = require "sci.stat"

local min, max, abs, floor, ceil, step, sqrt = xsys.from(math,
     "min, max, abs, floor, ceil, step, sqrt")
     
local normald = dist.normal

local alg32 = alg.typeof("int32_t")
local row = alg.row

local function rand_1_bin(v, j, rj, xmin, x, F, K)
  local j1, j2, j3 = rj[j][1], rj[j][2], rj[j][3]
  v:set(x[j1] + (x[j2] - x[j3])*F[j])
end

local function rand_2_bin(v, j, rj, xmin, x, F, K)
  local j1, j2, j3, j4, j5 = rj[j][1], rj[j][2], rj[j][3], rj[j][4], rj[j][5]
  v:set(x[j1] + (x[j2] - x[j3])*F[j] + (x[j4] - x[j5])*F[j])
end

local function randtobest_2_bin(v, j, rj, xmin, x, F, K)
  local j1, j2, j3, j4 = rj[j][1], rj[j][2], rj[j][3], rj[j][4]
  v:set(x[j] + (xmin - x[j])*F[j] + (x[j1] - x[j2])*F[j] 
                                  + (x[j3] - x[j4])*F[j])
end

local function currenttorand_1(v, j, rj, xmin, x, F, K)
  local j1, j2, j3 = rj[j][1], rj[j][2], rj[j][3]
  v:set(x[j] + (x[j1] - x[j])*K[j] + (x[j2] - x[j3])*F[j])
end

local strategies = { 
  rand_1_bin, 
  rand_2_bin,
  randtobest_2_bin,
  currenttorand_1,
}

-- Branch-free sampling of strategy according to the four probabilities in p.
local function sample_strategy_indices(rng, rs, p)
  assert(#p == #strategies)
  for j=1,#rs do
    local u = rng:sample()
    rs[j] = 1 
          + step(u - (p[1]))               
          + step(u - (p[1] + p[2]))
          + step(u - (p[1] + p[2] + p[3]))
  end
end

-- Sample an integer uniformly distributed on the interval from, ... to.
local function sample_int(rng, from, to)
  return floor(from + (to + 1 - from)*rng:sample())
end

-- Branch-free version, each row of rj contains three distinct j uniformly 
-- distributed on 1, ..., NP.
local function sample_distinct_indices(rng, rj)
  local NP = rj:nrow()
  local j = 1
  while j <= NP do
    local j1, j2, j3, j4, j5 = sample_int(rng, 1, NP), sample_int(rng, 1, NP), 
       sample_int(rng, 1, NP), sample_int(rng, 1, NP), sample_int(rng, 1, NP)
    rj[j][1], rj[j][2], rj[j][3], rj[j][4], rj[j][5] = j1, j2, j3, j4, j5
    -- Zero if any pairwise match, integer otherwise.
    local m = (j1 - j)
             *(j2 - j1)*(j2 - j)
             *(j3 - j2)*(j3 - j1)*(j3 - j)
             *(j4 - j3)*(j4 - j2)*(j4 - j1)*(j4 - j)
             *(j5 - j4)*(j5 - j3)*(j5 - j2)*(j5 - j1)*(j5 - j) 
    j = j + min(abs(m), 1)
  end
end

local function sample_F(rng, F, Fmu, Fsigma)
  for i=1,#F do F[i] = rng:sample() end
end

local function sample_K(rng, K)
  for i=1,#K do K[i] = rng:sample() end
end

-- MODIFICATION: Our algorithm performs flooring at 0 and 1.
-- CRm has column 1 set at 100: always CR = 1.
local function sample_CR(rng, CR, CRmu, CRsigma)
  for j=1,#CR do
    local v = normald(CRmu, CRsigma):sample(rng)
    CR[j] = max(0, min(v, 1))
  end
end

-- Element equal to one means 
local function sample_mutations(rng, rz, rs, CR)
  local NP, dim = #rz, #rz[1]
  for j=1,NP do
    if rs[j] == 4 then
      rz[j]:fill(1)
    else
      for d=1,dim do
        rz[j][d] = step(CR[j] - rng:sample())
      end
      -- Always move at least among one dimension:
      rz[j][sample_int(rng, 1, dim)] = 1
    end
  end
end

local function updatemin(xmin, fmin, xval, fval)
  if fval ~= fval then
    error("nan value encountered, check objective function definition")
  end
  if fval < fmin then
    return xval, fval
  else
    return xmin, fmin
  end
end

local function stop_iter(n)
  local c = 0 
  return function()
    c = c + 1
    return c >= n
  end
end

local function rows_as_vec(x)
  local o =  {}
  for i=1,x:nrow() do
    o[i] = x:copy(row(i))
  end
  return o
end

local function fmin(f, o)
  local stop = o.stop
  if type(stop) == "number" then
    if stop < 1 then
      error("number of required iterations must be strictly positive")
    end
    stop = stop_iter(stop)
  end
    
  local rng = o.rng or prng.std()
  local bound = o.bound or "reflect"
  local LP = o.LP or 20
  
  local dim = o.xval and o.xval:ncol() or #o.xl
  local NP = o.NP or min(10, 8*dim)
  if NP < 10 then
    error("NP >= 10 required")
  end
 
  -- Can be present or not, depending on whether a new population is randomly
  -- generated or not and whether bounds are applied or not:
  local xl, xu = o.xl, o.xu
  if xl and xu then
    if not (xl < xu) then
      error("xl < xu required")
    end
  end
  
  local xval, fval, x
  local xmin, fmin = nil, 1/0 -- Reference to minimum xval.
  if o.xval and o.fval then -- Initialize with argument population.
    xval = alg.mat(o.xval)
    fval = alg.vec(o.fval)
    x = rows_as_vec(xval)
    for j=1,NP do
      xmin, fmin = updatemin(xmin, fmin, x[j], fval[j])
    end
  else -- Initialize with randomly generated population.
    xval = alg.mat(NP, dim)
    fval = alg.vec(NP)
    x = rows_as_vec(xval)
    local maxinit = o.maxinit or 100*NP -- At least 1% in support zone.
    local iter = 0
    local popd = dist.mvuniform(xl, xu)
    for j=1,NP do
      repeat
        iter = iter + 1
        if iter > maxinit then
          return nil, "failed to initialize population, too many infinite "
                    .."evaluation of f, try to increase maxinit"
        end
        popd:sample(rng, x[j])
        fval[j] = f(x[j])
      until fval[j] < 1/0
      xmin, fmin = updatemin(xmin, fmin, x[j], fval[j])
    end
  end
     
  -- Equal probability to each strategy:
  local Pmu = alg.vec(4, 1/4)
  -- Centred around 0.5 with good dispersion: 68.2% of mass in (0, 1).
  local CRmu, CRsigma = 0.5, 0.5
  
  local Pstat = { } 
    for i=1,4 do Pstat[i] = stat.olmean(0) 
  end
  local CRstat = stat.olvar(0)
  local nsuccess = 0

  local F = alg.vec(NP)
  local K = alg.vec(NP)
  local CR = alg.vec(NP)
  
  local rs = alg32.vec(NP)      -- Random strategies.
  local rj = alg32.mat(NP, 5)   -- Random j-indices.
  
  local rz = { } -- Dimensions which mutate.
  local u = { } -- Mutated particles.
  for i=1,NP do
    rz[i], u[i] = alg.vec(dim), alg.vec(dim)
  end
  local v = alg.vec(dim)        -- Potential mutation particle.
      
  local generation = 0
  -- io.write("GENE SUCC CRMU   CRSI   P1     P2     P3     P4\n")
  while not stop(xmin, fmin, xval, fval) do
    generation = generation + 1
    -- Update meta-parameters.
    if generation % LP == 0 and nsuccess >= 100 then      
      for i=1,4 do
        Pmu[i] = Pstat[i]:mean() + 0.01
        Pstat[i]:clear()
      end
      Pmu:set(Pmu/Pmu:sum()) 
      CRmu, CRsigma = CRstat:mean(), max(0.1, sqrt(CRstat:var()))
      CRstat:clear()
      nsuccess = 0
    end
    -- Sample required quantities:
    sample_strategy_indices(rng, rs, Pmu)
    sample_distinct_indices(rng, rj)
    sample_F(rng, F)
    sample_K(rng, K)
    sample_CR(rng, CR, CRmu, CRsigma)
    sample_mutations(rng, rz, rs, CR)
    -- Evolve the population:
    for j=1,NP do
      -- Potential mutation:
      strategies[rs[j]](v, j, rj, xmin, x, F, K)
      -- Mutated particle:
      u[j]:set(rz[j]*v + (1 - rz[j])*x[j])
      -- Apply the bounds:
      for i=1,dim do
        if bound == "reflect" then        
          local b = max(min(u[j][i], xu[i]), xl[i])
          -- Value of b must be xl if x < xl, xu if x > xu and x otherwise.
          u[j][i] = 2*b - u[j][i]
        elseif bound == "absorb" then
          u[j][i] = max(xl[i], min(u[j][i], xu[i]))
        elseif bound == "no" then
          local _;
        else
          error("bound option '"..bound.."' not recognized")
        end
      end        
    end    
    -- Selection:
    for j=1,NP do
      local s = rs[j]
      local fuj = f(u[j])
      if fuj < fval[j] then -- It's an improvement --> select.
        nsuccess = nsuccess + 1
        Pstat[s]:push(1)
        CRstat:push(CR[j])
        fval[j] = fuj
        x[j]:set(u[j])
        xmin, fmin = updatemin(xmin, fmin, x[j], fuj)
      else
        Pstat[s]:push(0)
      end
      -- Update xval for next generation (used only in stop()):
      xval:set(x[j], row(j))
    end
  end
  
  return xmin:copy(), fmin, xval, fval
end

return { 
  fmin = fmin,
}
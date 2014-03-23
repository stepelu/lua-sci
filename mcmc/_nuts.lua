--------------------------------------------------------------------------------
-- NUTS MCMC sampler.
--
-- Copyright (C) 2011-2014 Stefano Peluchetti. All rights reserved.
--
-- Features, documentation and more: http://www.scilua.org .
--
-- This file is part of the SciLua library, which is released under the MIT 
-- license: full text in file LICENSE.TXT in the library's root folder.
--------------------------------------------------------------------------------

-- TODO: Implement non-diagonal mass matrix.

-- TODO: Allow for passing mass vector / matrix and epsilon to avoid adaptation
-- TODO: and continue simulation if needed (in the same spirit of API for 
-- TODO: fmin,fmax).

local alg  = require "sci.alg"
local dist = require "sci.dist"
local stat = require "sci.stat"
local math = require "sci.math"
local xsys = require "xsys"

local vec, sqrtel = alg.vec, alg.math.sqrt
local normal = dist.normal
local uniform = dist.uniform
local exp, log, sqrt, sign, step, min, abs, floor = xsys.from(math, 
     "exp, log, sqrt, sign, step, min, abs, floor")

local cache = setmetatable({}, { __mode = "k" })

-- NOTE: remove comments to verify each theta passed here is never mutated!
local function evalfgrad(fgrad, theta, gradv)
  local found = cache[theta]
  -- if found then assert(theta == found[3]) then    
  if found then
    if gradv then 
      gradv:set(found[2])
    end
    return found[1]
  else
    local hasg = gradv
    gradv = gradv or vec(#theta)
    local val = fgrad(theta, gradv)
    if not (abs(val) < 1/0) then -- If val is nan or not finite.
      val = - 1/0   -- No nan allowed.
      gradv:fill(0) -- Gradient is almost surely nan or not finite.
    end
    cache[theta] = { val, hasg and gradv:copy() or gradv --[[,theta:copy()]] }
    return val
  end
end

local function logpdf(fgrad, theta, r, m)
  return evalfgrad(fgrad, theta) - 0.5*(r^2/m):sum()
end

-- Gradv must be precomputed on third argument.
-- NOTE: remove comments to verify it is always the case.
local function leapfrog(fgrad, eps, th0, r0, th1, r1, gradv, m)
  -- evalfgrad(fgrad, th0, gradv)
  r1:set(r0 + gradv*0.5*eps)
  th1:set(th0 + r1/m*eps)
  evalfgrad(fgrad, th1, gradv)
  r1:set(r1 + gradv*0.5*eps)
end

local function heuristiceps(rng, fgrad, th0, gradv, m)
  local dim = #th0
  local r0, rt, tht = vec(dim), vec(dim), vec(dim)
  local eps = 1
  for i=1,#r0 do r0[i] = normal(0, 1):sample(rng) end
  local logpdf0 = logpdf(fgrad, th0, r0, m)
  evalfgrad(fgrad, th0, gradv)
  leapfrog(fgrad, eps, th0, r0, tht, rt, gradv, m)
  local alpha = sign((logpdf(fgrad, tht, rt, m) - logpdf0) - log(0.5))
  while alpha*(logpdf(fgrad, tht, rt, m) - logpdf0) > alpha*log(0.5) do
    eps = eps*2^alpha
    rt, tht = vec(dim), vec(dim)
    evalfgrad(fgrad, th0, gradv)
    leapfrog(fgrad, eps, th0, r0, tht, rt, gradv, m)
  end
  return eps
end

-- Shorts: m = minus, p = plus, 1 = 1 prime, 2 = 2 primes.
-- This function does not modify any of its arguments.
-- Also all of the returned arguments are not modified in the recursion.
-- As long as the input or returned vectors are *not* modified it's fine to
-- work with references that may alias each other (see also caching of fgrad).
-- This means that all vectors here are effectively immutable after 
-- "initialization", which in this context means after being passed to the 
-- leapfrog function which modifies them.
local function buildtree(rng, fgrad, th, r, logu, v, j, eps, th0, r0, gradv,
    m, deltamax)
  deltamax = deltamax or 1000
  local dim = #th
  if j == 0 then
    local th1, r1 = vec(dim), vec(dim)
    -- Only place where a vec is modified in buildtree (th1, r1, newly created):
    evalfgrad(fgrad, th, gradv)
    leapfrog(fgrad, v*eps, th, r, th1, r1, gradv, m)
    local logpdfv1 = logpdf(fgrad, th1, r1, m)
    local n1 = step(logpdfv1 - logu)
    local s1 = step(deltamax + logpdfv1 - logu)
    return th1, r1, th1, r1, th1, n1, s1,
      min(1, exp(logpdfv1 - logpdf(fgrad, th0, r0, m))), 1
  else
    local thm, rm, thp, rp, th1, n1, s1, a1, na1 =
    buildtree(rng, fgrad, th, r, logu, v, j - 1, eps, th0, r0, gradv, m,
      deltamax)
    local _, th2, n2, s2, a2, na2
    if s1 == 1 then
      if v == -1 then
        thm, rm, _, _, th2, n2, s2, a2, na2 =
        buildtree(rng, fgrad, thm, rm, logu, v, j - 1, eps, th0, r0, gradv,
          m, deltamax)
      else
        _, _, thp, rp, th2, n2, s2, a2, na2 =
        buildtree(rng, fgrad, thp, rp, logu, v, j - 1, eps, th0, r0, gradv,
          m, deltamax)
      end
      if rng:sample() < n2/(n1 + n2) then
        th1 = th2
      end
      a1 = a1 + a2
      na1 = na1 + na2
      s1 = s2*step((thp - thm):t()*rm)*step((thp - thm):t()*rp)
      n1 = n1 + n2
    end
    return thm, rm, thp, rp, th1, n1, s1, a1, na1
  end
end

local function stop_iter(n)
  local c = 0 
  return function()
    c = c + 1
    return c >= n
  end
end

-- Work by references on the vectors, only newly created one are modified in
-- the leapforg function. Use vectors as read-only.
local function nuts(rng, fgrad, theta0, o)
  local stop = o.stop
  local stopadapt = o.stopadapt or 1024
  if type(stop) == "number" then
    stop = stop_iter(stop)
  end
  if log(stopadapt)/log(2) ~= floor(log(stopadapt)/log(2)) then
    error("stopadapt must be a power of 2")
  end
  local olstat = o.olstat
  local delta    = o.delta    or 0.8
  local gamma    = o.gamma    or 0.05
  local t0       = o.t0       or 10
  local k        = o.k        or 0.75
  local deltamax = o.deltamax or 1000

  local dim = #theta0
  local thr0, thr1 = vec(theta0)
  local r0, gradv = vec(dim), vec(dim)
  local vare = stat.olvar(dim)
  local currentvar = vec(dim)
  local varu = 16
  -- M must be precision matrix (Cov^-1) or Var^-1 for diagonal mass case.
  local M = vec(dim, 1e3)
  local currentvar = vec(1/M)
  local eps = heuristiceps(rng, fgrad, thr0, gradv, M)
  local mu = log(10*eps)
  local madapt, Ht, lepst, leps = 0, 0, 0
  local totadapt = 0
  while true do
    for i=1,#r0 do r0[i] = normal(0, 1):sample(rng) end
    r0:set(r0*sqrtel(M))
    local logpdfr0 = logpdf(fgrad, thr0, r0, M)
    assert(logpdfr0 > -1/0)
    local logu = log(rng:sample()) + logpdfr0
    local thm, thp = thr0, thr0
    local rm, rp = r0, r0
    local j, n, s = 0, 1, 1
    local a, na
    thr1 = thr0
    while s == 1 do
      local v = sign(uniform(-1, 1):sample(rng))
      local _, th1, n1, s1
      if v == -1 then
        thm, rm, _, _, th1, n1, s1, a, na =
        buildtree(rng, fgrad, thm, rm, logu, v, j, eps, thr0, r0, gradv, M,
          deltamax)
      else
        _, _, thp, rp, th1, n1, s1, a, na =
        buildtree(rng, fgrad, thp, rp, logu, v, j, eps, thr0, r0, gradv, M,
          deltamax)
      end
      if s1 == 1 then
        if rng:sample() < min(1, n1/n) then
          thr1 = th1
        end
      end
      n = n + n1
      s = s1*step((thp - thm):t()*rm)*step((thp - thm):t()*rp)
      j = j + 1
      -- Limit tree depth during adaptation phase:
      if totadapt <= stopadapt/2 and j >= 10 then break end
    end
    local alpha = a/na
    if totadapt < stopadapt then
      totadapt = totadapt + 1 -- Last possible totadapt is stopadapt.
      madapt = madapt + 1
      Ht = (1 - 1/(madapt + t0))*Ht + 1/(madapt + t0)*(delta - alpha)
      leps = mu - sqrt(madapt)/gamma*Ht -- Log-eps used in adaptation.
      lepst = (madapt^-k)*leps + (1 - madapt^-k)*lepst -- Optimal log-eps.
      if totadapt == stopadapt then
        eps = exp(lepst)
      else
        eps = exp(leps)
      end
      vare:push(thr1)
      if totadapt == varu then
        -- Set mass using current var estimates.
        vare:var(currentvar)
        M:set(1/currentvar)
        -- Re-initialize var estimators.
        vare:clear()
        varu = varu*2
        mu = log(10*eps)
        madapt, Ht, lepst = 1, 0, 0
      end
    else
      olstat:push(thr1)
      if stop(thr1) then break end
    end
    thr0 = thr1
  end
  return thr1:copy()
end

return {
  mcmc = nuts
}